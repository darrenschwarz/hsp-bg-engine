//! `bg-engine --health [url]`: the container's health probe (GP-476).
//!
//! A real HTTP GET against the running server's `/health`, over a plain TCP
//! socket so the runtime image needs no curl. Exit status 0 when the server
//! answered HTTP 200 with `"ok":true`, 1 otherwise -- which is exactly the
//! contract `/health` itself keeps (200 when ready, 503 when not), so Docker's
//! HEALTHCHECK exercises the same check an operator or the HSP server would.
//!
//! The old `--health` flag was never implemented: the binary ignored it and
//! tried to start a second server on the occupied port.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_secs(3);

/// Run the probe against `url` (default `http://127.0.0.1:<PORT>/health`).
/// Returns the process exit code and prints what it saw, for the container log.
pub fn run(url: Option<&str>, default_port: u16) -> i32 {
    let target = url
        .map(str::to_string)
        .unwrap_or_else(|| format!("http://127.0.0.1:{default_port}/health"));
    match probe(&target) {
        Ok(outcome) => {
            println!("{target}: {}", outcome.summary());
            if outcome.healthy { 0 } else { 1 }
        }
        Err(e) => {
            println!("{target}: {e}");
            1
        }
    }
}

pub struct Outcome {
    pub status: u16,
    pub healthy: bool,
    pub body: String,
}

impl Outcome {
    fn summary(&self) -> String {
        let body = self.body.trim();
        let shown = if body.len() > 300 { &body[..300] } else { body };
        format!(
            "HTTP {} -> {} {shown}",
            self.status,
            if self.healthy { "healthy" } else { "UNHEALTHY" }
        )
    }
}

fn probe(url: &str) -> Result<Outcome, String> {
    let (host, port, path) = parse_url(url)?;
    let address = (host.as_str(), port)
        .to_socket_addrs()
        .map_err(|e| format!("cannot resolve {host}:{port}: {e}"))?
        .next()
        .ok_or_else(|| format!("no address for {host}:{port}"))?;
    let mut stream = TcpStream::connect_timeout(&address, TIMEOUT)
        .map_err(|e| format!("connect failed: {e}"))?;
    stream.set_read_timeout(Some(TIMEOUT)).ok();
    stream.set_write_timeout(Some(TIMEOUT)).ok();
    stream
        .write_all(
            format!("GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\nAccept: application/json\r\n\r\n")
                .as_bytes(),
        )
        .map_err(|e| format!("request failed: {e}"))?;
    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .map_err(|e| format!("reading the response failed: {e}"))?;
    parse_response(&String::from_utf8_lossy(&raw))
}

/// `http://host[:port][/path]` only -- the probe talks to itself.
pub fn parse_url(url: &str) -> Result<(String, u16, String), String> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| format!("only http:// URLs are supported, got {url:?}"))?;
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/health"),
    };
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) => (
            h,
            p.parse::<u16>()
                .map_err(|_| format!("bad port in {url:?}"))?,
        ),
        None => (authority, 80),
    };
    if host.is_empty() {
        return Err(format!("no host in {url:?}"));
    }
    Ok((host.to_string(), port, path.to_string()))
}

/// Healthy means HTTP 200 and a body that says `"ok":true`.
pub fn parse_response(raw: &str) -> Result<Outcome, String> {
    let status_line = raw.lines().next().unwrap_or("");
    let status = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u16>().ok())
        .ok_or_else(|| format!("not an HTTP response: {status_line:?}"))?;
    let body = raw
        .split_once("\r\n\r\n")
        .map(|(_, b)| b)
        .unwrap_or("")
        .to_string();
    let compact: String = body.chars().filter(|c| !c.is_whitespace()).collect();
    let healthy = status == 200 && compact.contains("\"ok\":true");
    Ok(Outcome {
        status,
        healthy,
        body,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_default_and_explicit_urls() {
        assert_eq!(
            parse_url("http://127.0.0.1:8090/health").unwrap(),
            ("127.0.0.1".into(), 8090, "/health".into())
        );
        assert_eq!(
            parse_url("http://localhost").unwrap(),
            ("localhost".into(), 80, "/health".into())
        );
        assert!(parse_url("https://x/health").is_err());
        assert!(parse_url("http://:8090/health").is_err());
    }

    #[test]
    fn only_200_with_ok_true_is_healthy() {
        let ok = parse_response("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\r\n{\"ok\":true,\"apiVersion\":1}").unwrap();
        assert!(ok.healthy);
        assert_eq!(ok.status, 200);
        let unhealthy = parse_response(
            "HTTP/1.1 503 Service Unavailable\r\n\r\n{\"ok\":false,\"error\":\"x\"}",
        )
        .unwrap();
        assert!(!unhealthy.healthy);
        assert_eq!(unhealthy.status, 503);
        let lying = parse_response("HTTP/1.1 200 OK\r\n\r\n{\"ok\":false}").unwrap();
        assert!(!lying.healthy);
        assert!(parse_response("garbage").is_err());
    }
}
