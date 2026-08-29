//! Bounded neural evaluation (GP-477).
//!
//! Inference is CPU work that takes milliseconds at 1-ply and seconds at
//! 2-ply. Run on the async runtime it would stall every other request --
//! `/health` included -- for its whole duration, and run without a bound a
//! burst of requests would each get a slice of the same cores and all of
//! them would be late. So every evaluation runs on the blocking pool under a
//! semaphore of `BG_ENGINE_MAX_CONCURRENT_EVALS` permits (default 1, at most
//! 4: the nets are not the kind of work that benefits from more threads than
//! cores, and the sidecar's instance is small).
//!
//! A request waits at most `QUEUE_WAIT` for a permit. Past that the caller
//! gets `Saturated` (HTTP 429 with `Retry-After`) and no work is started: a
//! live table has a 1.5 s budget and falls back to its own heuristic, so an
//! answer that is late is worth less than a refusal that is prompt.
//!
//! The permit is moved INTO the blocking closure. Dropping the request
//! future -- a client that timed out and closed the socket -- cannot release
//! the permit while inference is still running on the blocking thread; the
//! permit goes back only when that work returns. Without this a cancelled
//! request would let the next one start and the bound would be a fiction.

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;

/// The environment variable that sets the bound.
pub const MAX_CONCURRENT_ENV: &str = "BG_ENGINE_MAX_CONCURRENT_EVALS";
pub const DEFAULT_MAX_CONCURRENT_EVALS: usize = 1;
pub const MAX_CONCURRENT_EVALS_CEILING: usize = 4;
/// How long a request may wait for a permit before it is refused.
pub const QUEUE_WAIT: Duration = Duration::from_millis(100);
/// What a refused request is told to wait, in the `Retry-After` header.
pub const RETRY_AFTER_SECONDS: u64 = 1;

/// Read the bound from the environment: unset means the default, anything
/// that is not an integer in `1..=4` refuses to start.
pub fn max_concurrent_from_env() -> Result<usize, String> {
    match std::env::var(MAX_CONCURRENT_ENV) {
        Err(std::env::VarError::NotPresent) => Ok(DEFAULT_MAX_CONCURRENT_EVALS),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(format!("{MAX_CONCURRENT_ENV} is not valid UTF-8"))
        }
        Ok(raw) => parse_max_concurrent(&raw),
    }
}

pub fn parse_max_concurrent(raw: &str) -> Result<usize, String> {
    let value: usize = raw.trim().parse().map_err(|_| {
        format!(
            "{MAX_CONCURRENT_ENV}={raw:?} is not a whole number; use 1..={MAX_CONCURRENT_EVALS_CEILING} (default {DEFAULT_MAX_CONCURRENT_EVALS})"
        )
    })?;
    if (1..=MAX_CONCURRENT_EVALS_CEILING).contains(&value) {
        Ok(value)
    } else {
        Err(format!(
            "{MAX_CONCURRENT_ENV}={value} is outside 1..={MAX_CONCURRENT_EVALS_CEILING} (default {DEFAULT_MAX_CONCURRENT_EVALS})"
        ))
    }
}

/// The evaluation bound: a semaphore, and the number it started with.
#[derive(Clone)]
pub struct Capacity {
    evals: Arc<Semaphore>,
    bound: usize,
}

/// Why an evaluation was not started.
#[derive(Debug)]
pub enum Refused {
    /// No permit came free within `QUEUE_WAIT`: `in_use` of `bound` slots
    /// were still busy when the wait ran out.
    Saturated {
        bound: usize,
        in_use: usize,
        waited: Duration,
    },
    /// The blocking task did not return (a panic in the evaluator).
    Failed(String),
}

impl Capacity {
    pub fn new(bound: usize) -> Self {
        Self {
            evals: Arc::new(Semaphore::new(bound)),
            bound,
        }
    }

    pub fn bound(&self) -> usize {
        self.bound
    }

    /// Permits not currently held by running (or cancelled-but-still-running) work.
    pub fn available(&self) -> usize {
        self.evals.available_permits()
    }

    /// Run `work` on the blocking pool under one permit, waiting at most
    /// `QUEUE_WAIT` for it. The permit lives inside the closure: see the
    /// module note on cancellation.
    pub async fn run_blocking<T, F>(&self, work: F) -> Result<T, Refused>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let started = std::time::Instant::now();
        let permit =
            match tokio::time::timeout(QUEUE_WAIT, self.evals.clone().acquire_owned()).await {
                Ok(Ok(permit)) => permit,
                Ok(Err(_closed)) => {
                    return Err(Refused::Failed("evaluation semaphore closed".to_string()));
                }
                Err(_elapsed) => {
                    return Err(Refused::Saturated {
                        bound: self.bound,
                        in_use: self.bound - self.available(),
                        waited: started.elapsed(),
                    });
                }
            };
        tokio::task::spawn_blocking(move || {
            // Held until the work returns, whatever happens to the request
            // future that started it.
            let _permit = permit;
            work()
        })
        .await
        .map_err(|join| Refused::Failed(format!("evaluation task did not complete: {join}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::time::Instant;

    #[test]
    fn the_bound_comes_from_the_environment_within_1_to_4() {
        assert_eq!(parse_max_concurrent("1"), Ok(1));
        assert_eq!(parse_max_concurrent(" 4 "), Ok(4));
        assert!(parse_max_concurrent("0").is_err());
        assert!(parse_max_concurrent("5").is_err());
        assert!(parse_max_concurrent("-1").is_err());
        assert!(parse_max_concurrent("two").is_err());
        assert!(parse_max_concurrent("").is_err());
        assert_eq!(DEFAULT_MAX_CONCURRENT_EVALS, 1);
        assert_eq!(MAX_CONCURRENT_EVALS_CEILING, 4);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrency_never_exceeds_the_bound() {
        let capacity = Capacity::new(2);
        let running = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();
        for _ in 0..12 {
            let capacity = capacity.clone();
            let running = running.clone();
            let peak = peak.clone();
            tasks.push(tokio::spawn(async move {
                // Keep trying past saturation: this test is about the bound,
                // not the queue wait.
                loop {
                    let running = running.clone();
                    let peak = peak.clone();
                    let outcome = capacity
                        .run_blocking(move || {
                            let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                            peak.fetch_max(now, Ordering::SeqCst);
                            std::thread::sleep(Duration::from_millis(20));
                            running.fetch_sub(1, Ordering::SeqCst);
                        })
                        .await;
                    if outcome.is_ok() {
                        return;
                    }
                }
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        assert_eq!(peak.load(Ordering::SeqCst), 2);
        assert_eq!(running.load(Ordering::SeqCst), 0);
        assert_eq!(capacity.available(), 2);
    }

    #[tokio::test]
    async fn a_cancelled_request_keeps_its_permit_until_the_blocking_work_ends() {
        let capacity = Capacity::new(1);
        let (started_tx, started_rx) = mpsc::channel::<()>();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let (done_tx, done_rx) = mpsc::channel::<()>();

        let request = tokio::spawn({
            let capacity = capacity.clone();
            async move {
                capacity
                    .run_blocking(move || {
                        started_tx.send(()).unwrap();
                        // Inference in progress: block until the test lets go.
                        release_rx.recv().unwrap();
                        done_tx.send(()).unwrap();
                    })
                    .await
                    .unwrap();
            }
        });

        // The work is running on the blocking pool...
        tokio::task::spawn_blocking(move || started_rx.recv().unwrap())
            .await
            .unwrap();
        assert_eq!(capacity.available(), 0);

        // ...when the client goes away. The request future is dropped, the
        // permit is not.
        request.abort();
        let _ = request.await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(
            capacity.available(),
            0,
            "cancellation must not free capacity that is still in use"
        );

        // A newcomer is refused while the orphaned inference is still running.
        let refused = capacity.run_blocking(|| ()).await;
        assert!(
            matches!(refused, Err(Refused::Saturated { bound: 1, .. })),
            "{refused:?}"
        );

        // Only the end of the work returns the permit.
        release_tx.send(()).unwrap();
        tokio::task::spawn_blocking(move || done_rx.recv().unwrap())
            .await
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while capacity.available() == 0 && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(capacity.available(), 1);
        assert!(capacity.run_blocking(|| 7).await.is_ok());
    }

    #[tokio::test]
    async fn saturation_is_refused_after_the_queue_wait_and_the_refusal_says_how_long() {
        let capacity = Capacity::new(1);
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let holder = tokio::spawn({
            let capacity = capacity.clone();
            async move {
                capacity
                    .run_blocking(move || release_rx.recv().unwrap())
                    .await
                    .unwrap()
            }
        });
        let deadline = Instant::now() + Duration::from_secs(2);
        while capacity.available() != 0 && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(capacity.available(), 0);

        let started = Instant::now();
        let refused = capacity.run_blocking(|| ()).await;
        let elapsed = started.elapsed();
        match refused {
            Err(Refused::Saturated {
                bound,
                in_use,
                waited,
            }) => {
                assert_eq!(bound, 1);
                assert_eq!(in_use, 1);
                assert!(waited >= QUEUE_WAIT, "{waited:?}");
            }
            other => panic!("expected Saturated, got {other:?}"),
        }
        assert!(elapsed >= QUEUE_WAIT, "{elapsed:?}");
        assert!(
            elapsed < Duration::from_secs(1),
            "{elapsed:?}: the wait must be bounded"
        );

        release_tx.send(()).unwrap();
        holder.await.unwrap();
        assert_eq!(capacity.available(), 1);
    }

    #[tokio::test]
    async fn the_result_of_the_work_comes_back_and_the_permit_is_returned() {
        let capacity = Capacity::new(3);
        assert_eq!(capacity.bound(), 3);
        let value = capacity.run_blocking(|| 40 + 2).await.unwrap();
        assert_eq!(value, 42);
        assert_eq!(capacity.available(), 3);
    }

    #[tokio::test]
    async fn a_panic_in_the_work_is_reported_not_propagated_and_frees_the_permit() {
        let capacity = Capacity::new(1);
        let outcome: Result<(), Refused> =
            capacity.run_blocking(|| panic!("evaluator exploded")).await;
        assert!(matches!(outcome, Err(Refused::Failed(_))), "{outcome:?}");
        assert_eq!(capacity.available(), 1);
    }
}
