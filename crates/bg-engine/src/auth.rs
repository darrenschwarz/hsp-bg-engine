//! Bearer authentication for the evaluation routes (GP-477).
//!
//! `/rank`, `/evaluate` and `/cube` require `Authorization: Bearer <token>`
//! where the token is `BG_ENGINE_AUTH_TOKEN` from the sidecar's environment.
//! `/health` stays public: the container health check, Render's health check
//! and an operator's curl carry no credentials, and a readiness report leaks
//! nothing an attacker can use.
//!
//! The check runs as middleware in front of the handlers, so a request that
//! fails it is refused before its body is read as JSON and long before any
//! neural evaluation. Refusals are cheap and uniform: HTTP 401 with a fixed
//! body, whether the header is missing, malformed, or carries the wrong
//! token -- the reply never says which.
//!
//! Comparison is constant time. The presented credential and the configured
//! token are both hashed with SHA-256 and the two fixed-size digests are
//! compared with a branch-free fold, so the time taken does not depend on
//! how many leading bytes of the guess were right, nor on the length of the
//! real token.

use axum::http::HeaderMap;
use axum::http::header::AUTHORIZATION;
use sha2::{Digest, Sha256};

/// The environment variable the token is read from.
pub const TOKEN_ENV: &str = "BG_ENGINE_AUTH_TOKEN";

/// The shortest token accepted at startup. A deploy that "fixes" a failed
/// start by setting the variable to `x` has not configured authentication.
pub const MIN_TOKEN_LEN: usize = 16;

/// The token grammar, shared verbatim with the HSP client
/// (`src/backgammon/bot/core/wildbgEngine.ts`, `validateAuthToken`): at least
/// `MIN_TOKEN_LEN` characters, every one of them visible ASCII (`!`..`~`,
/// 0x21..=0x7E). No spaces, no control characters, nothing outside ASCII.
/// That is the alphabet an `Authorization` header value can carry and be read
/// back as a `str` on this side (`HeaderValue::to_str`), so a token outside
/// it could never authenticate -- it is refused at startup, on both sides,
/// rather than at the first request.
pub fn is_visible_ascii(c: char) -> bool {
    ('!'..='~').contains(&c)
}

/// Check a token against the grammar. The message names the variable and
/// the rule; never the value.
pub fn validate_token(secret: &str) -> Result<(), String> {
    let chars = secret.chars().count();
    if chars < MIN_TOKEN_LEN {
        return Err(format!(
            "{TOKEN_ENV} is too short ({chars} characters): use at least {MIN_TOKEN_LEN}, \
             for example `openssl rand -hex 32`"
        ));
    }
    if let Some(bad) = secret.chars().find(|c| !is_visible_ascii(*c)) {
        let what = if bad.is_ascii() {
            format!("a control or whitespace character (0x{:02x})", bad as u32)
        } else {
            format!("a non-ASCII character (U+{:04X})", bad as u32)
        };
        return Err(format!(
            "{TOKEN_ENV} contains {what}: a token must be visible ASCII only (0x21..0x7E), \
             the alphabet an Authorization header can carry; use for example `openssl rand -hex 32`"
        ));
    }
    Ok(())
}

/// The configured token, kept only as its SHA-256 digest.
#[derive(Clone)]
pub struct BearerToken {
    digest: [u8; 32],
}

/// Not even the digest is printed: nothing about the token reaches a log.
impl std::fmt::Debug for BearerToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BearerToken([redacted])")
    }
}

/// A request that failed the check. Carries nothing about the request.
#[derive(Debug, PartialEq, Eq)]
pub struct Unauthorized;

impl BearerToken {
    /// Read `BG_ENGINE_AUTH_TOKEN`, refusing to start without a usable one.
    pub fn from_env() -> Result<Self, String> {
        match std::env::var(TOKEN_ENV) {
            Ok(value) => Self::new(&value),
            Err(std::env::VarError::NotPresent) => Err(format!(
                "{TOKEN_ENV} is not set -- the evaluation routes require a bearer token; \
                 set it to a random secret of at least {MIN_TOKEN_LEN} characters \
                 (for example `openssl rand -hex 32`) on this service and on the HSP server"
            )),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err(format!("{TOKEN_ENV} is not valid UTF-8"))
            }
        }
    }

    /// Validate and digest a token: see `validate_token` for the grammar.
    pub fn new(secret: &str) -> Result<Self, String> {
        validate_token(secret)?;
        Ok(Self {
            digest: digest_of(secret.as_bytes()),
        })
    }

    /// Accept the request only if it carries `Authorization: Bearer <token>`
    /// with exactly the configured token. The scheme is matched
    /// case-insensitively (RFC 7235); the credential is matched exactly.
    pub fn check(&self, headers: &HeaderMap) -> Result<(), Unauthorized> {
        let presented = headers
            .get(AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(bearer_credential)
            .ok_or(Unauthorized)?;
        if constant_time_eq(&digest_of(presented.as_bytes()), &self.digest) {
            Ok(())
        } else {
            Err(Unauthorized)
        }
    }
}

/// The credential of a `Bearer <credential>` header value, or None when the
/// value is not of that shape. Exactly one space, no surrounding whitespace
/// on the credential: what the HSP client sends and nothing looser.
fn bearer_credential(value: &str) -> Option<&str> {
    let (scheme, credential) = value.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("bearer") || credential.is_empty() {
        return None;
    }
    if credential.starts_with(' ') || credential.ends_with(' ') {
        return None;
    }
    Some(credential)
}

fn digest_of(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Branch-free equality over two fixed-size digests. Every byte is visited
/// whatever the result, and the accumulator passes through `black_box` so the
/// optimiser cannot turn the fold back into an early exit.
pub fn constant_time_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    std::hint::black_box(diff) == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    const TOKEN: &str = "0123456789abcdef0123456789abcdef";

    fn headers(value: Option<&str>) -> HeaderMap {
        let mut headers = HeaderMap::new();
        if let Some(v) = value {
            headers.insert(AUTHORIZATION, HeaderValue::from_str(v).unwrap());
        }
        headers
    }

    #[test]
    fn a_usable_token_is_at_least_16_visible_ascii_characters() {
        assert!(BearerToken::new(TOKEN).is_ok());
        assert!(BearerToken::new("sixteen-chars-ok").is_ok());
        // The whole visible range is fine, punctuation included.
        assert!(BearerToken::new("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~").is_ok());

        // Too short, counted in characters.
        for short in ["", "x", "fifteen-chars--"] {
            let err = BearerToken::new(short).unwrap_err();
            assert!(err.contains("too short"), "{short:?}: {err}");
            assert!(err.contains(TOKEN_ENV));
        }

        // Non-visible ASCII: space, tab, newline, DEL, other controls.
        for (name, bad) in [
            ("space", "has a space in it token"),
            ("trailing newline", "trailing-newline-token\n"),
            ("tab", "tab\tinside-the-token"),
            ("DEL", "del\u{7f}inside-the-token"),
            ("NUL", "nul\u{0}inside-the-token-"),
            ("CR", "cr\rinside-the-token-"),
            ("leading space", " leading-space-token"),
        ] {
            let err = BearerToken::new(bad).unwrap_err();
            assert!(err.contains("control or whitespace"), "{name}: {err}");
            assert!(err.contains("visible ASCII"), "{name}: {err}");
            assert!(
                !err.contains(bad),
                "{name}: the message must not carry the value"
            );
        }

        // Outside ASCII: emoji, Latin-1, a non-breaking space, wide letters.
        // Long enough in bytes -- the byte count is not the grammar.
        for (name, bad) in [
            ("emoji", "🔐🔐🔐🔐🔐🔐🔐🔐🔐🔐🔐🔐🔐🔐🔐🔐"),
            ("emoji among ascii", "0123456789abcde🔐"),
            (
                "Latin-1 e-acute",
                "caf\u{e9}-caf\u{e9}-caf\u{e9}-caf\u{e9}-1234",
            ),
            ("Latin-1 n-tilde", "ma\u{f1}ana-ma\u{f1}ana-ma\u{f1}ana-1"),
            ("non-breaking space", "nbsp\u{a0}inside-the-token"),
            (
                "full-width letters",
                "\u{ff41}\u{ff42}\u{ff43}\u{ff44}\u{ff45}\u{ff46}\u{ff47}\u{ff48}\u{ff49}\u{ff4a}\u{ff4b}\u{ff4c}\u{ff4d}\u{ff4e}\u{ff4f}\u{ff50}",
            ),
        ] {
            assert!(bad.len() >= MIN_TOKEN_LEN, "{name}: {} bytes", bad.len());
            let err = BearerToken::new(bad).unwrap_err();
            assert!(err.contains("non-ASCII"), "{name}: {err}");
            assert!(err.contains("visible ASCII"), "{name}: {err}");
            assert!(
                !err.contains(bad),
                "{name}: the message must not carry the value"
            );
        }

        // The grammar is the header alphabet: an accepted token reads back
        // from a header value as the same str, while a non-ASCII one can be
        // put into a header (opaque bytes) but never read back -- so it could
        // never match, which is why it is refused at startup instead.
        assert_eq!(
            HeaderValue::from_str(TOKEN).unwrap().to_str().unwrap(),
            TOKEN
        );
        let latin1 = HeaderValue::from_str("caf\u{e9}-caf\u{e9}-caf\u{e9}-caf\u{e9}-1234").unwrap();
        assert!(latin1.to_str().is_err());
    }

    #[test]
    fn startup_reads_the_environment_through_the_same_grammar() {
        // The environment is process-wide: one test owns it, and restores it.
        let previous = std::env::var_os(TOKEN_ENV);
        let outcome = std::panic::catch_unwind(|| {
            // SAFETY (test only): no other thread in this test binary reads or
            // writes BG_ENGINE_AUTH_TOKEN; every other test uses `new`.
            unsafe { std::env::remove_var(TOKEN_ENV) };
            assert!(BearerToken::from_env().unwrap_err().contains("is not set"));
            for (name, bad, rule) in [
                ("short", "abc", "too short"),
                ("emoji", "🔐🔐🔐🔐🔐🔐🔐🔐🔐🔐🔐🔐🔐🔐🔐🔐", "non-ASCII"),
                (
                    "latin-1",
                    "caf\u{e9}-caf\u{e9}-caf\u{e9}-caf\u{e9}-1234",
                    "non-ASCII",
                ),
                ("space", "has a space in it token", "control or whitespace"),
                (
                    "trailing newline",
                    "trailing-newline-token\n",
                    "control or whitespace",
                ),
            ] {
                unsafe { std::env::set_var(TOKEN_ENV, bad) };
                let err = BearerToken::from_env().unwrap_err();
                assert!(err.contains(rule), "{name}: {err}");
                assert!(
                    !err.contains(bad.trim()),
                    "{name}: the message must not carry the value"
                );
            }
            unsafe { std::env::set_var(TOKEN_ENV, TOKEN) };
            assert!(BearerToken::from_env().is_ok());
        });
        match previous {
            Some(value) => unsafe { std::env::set_var(TOKEN_ENV, value) },
            None => unsafe { std::env::remove_var(TOKEN_ENV) },
        }
        if let Err(panic) = outcome {
            std::panic::resume_unwind(panic);
        }
    }

    #[test]
    fn a_presented_credential_outside_the_grammar_is_unauthorized() {
        let token = BearerToken::new(TOKEN).unwrap();
        let mut headers = HeaderMap::new();
        // Non-ASCII bytes in the header: unreadable, so unauthorized.
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_bytes("Bearer caf\u{e9}".as_bytes()).unwrap(),
        );
        assert_eq!(token.check(&headers), Err(Unauthorized));
        // A raw tab inside the credential is a readable header value, and
        // still not a token.
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}\t{}", &TOKEN[..8], &TOKEN[8..])).unwrap(),
        );
        assert_eq!(token.check(&headers), Err(Unauthorized));
    }

    #[test]
    fn exactly_the_configured_credential_is_accepted() {
        let token = BearerToken::new(TOKEN).unwrap();
        assert_eq!(
            token.check(&headers(Some(&format!("Bearer {TOKEN}")))),
            Ok(())
        );
        // The scheme is case-insensitive per RFC 7235; the credential is not.
        assert_eq!(
            token.check(&headers(Some(&format!("bearer {TOKEN}")))),
            Ok(())
        );
        assert_eq!(
            token.check(&headers(Some(&format!("BEARER {TOKEN}")))),
            Ok(())
        );
    }

    #[test]
    fn missing_malformed_and_wrong_credentials_are_all_refused_alike() {
        let token = BearerToken::new(TOKEN).unwrap();
        let refused = [
            None,
            Some(""),
            Some("Bearer"),
            Some("Bearer "),
            Some(&format!("Basic {TOKEN}")),
            Some(&format!("Token {TOKEN}")),
            Some(TOKEN),
            Some(&format!("Bearer  {TOKEN}")),
            Some(&format!("Bearer {TOKEN} ")),
            Some(&format!("Bearer {}", &TOKEN[..31])),
            Some(&format!("Bearer {TOKEN}0")),
            Some(&format!("Bearer {}", TOKEN.to_uppercase())),
            Some("Bearer 0123456789abcdef0123456789abcdeg"),
        ];
        for value in refused {
            assert_eq!(token.check(&headers(value)), Err(Unauthorized), "{value:?}");
        }
    }

    #[test]
    fn the_digest_comparison_is_exact_and_symmetric() {
        let a = digest_of(b"alpha");
        let b = digest_of(b"alpha");
        let c = digest_of(b"alphb");
        assert!(constant_time_eq(&a, &b));
        assert!(!constant_time_eq(&a, &c));
        assert!(!constant_time_eq(&c, &a));
        let mut flipped = a;
        flipped[31] ^= 0x01;
        assert!(!constant_time_eq(&a, &flipped));
    }

    #[test]
    fn the_token_itself_is_not_kept() {
        // Only a digest is stored: the struct is exactly 32 bytes and the
        // configured secret cannot be read back out of it.
        assert_eq!(std::mem::size_of::<BearerToken>(), 32);
        let token = BearerToken::new(TOKEN).unwrap();
        assert_eq!(token.digest, digest_of(TOKEN.as_bytes()));
    }
}
