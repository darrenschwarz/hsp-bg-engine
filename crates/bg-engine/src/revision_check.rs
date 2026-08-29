// What counts as a source revision (GP-476).
//
// Shared by build.rs (via `include!`) and the crate itself, so the rule the
// build enforces is the rule the tests check. No dependencies on purpose.

/// A full git commit hash: 40 hex digits (SHA-1) or 64 (a SHA-256
/// repository), normalised to lowercase. Anything else -- a branch name, an
/// abbreviated hash, "unversioned", an empty string -- is refused: the
/// identity baked into the binary must name exactly the source it was built
/// from, and there is deliberately no fallback label.
pub fn validate_revision(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("the source revision is empty".to_string());
    }
    let lower = trimmed.to_ascii_lowercase();
    let is_hex = lower.bytes().all(|b| b.is_ascii_hexdigit());
    if !is_hex || !(lower.len() == 40 || lower.len() == 64) {
        return Err(format!(
            "expected a full 40- or 64-hex-digit commit hash, got {trimmed:?}"
        ));
    }
    Ok(lower)
}
