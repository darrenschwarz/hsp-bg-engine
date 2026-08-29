//! Bakes the sidecar's identity into the binary (GP-476).
//!
//! Three facts become `env!()` constants the crate reads at compile time:
//!
//!   BG_ENGINE_SOURCE_REVISION  the full commit hash of the source being built
//!   BG_ENGINE_CONTACT_SHA256   SHA-256 of neural-nets/contact.onnx
//!   BG_ENGINE_RACE_SHA256      SHA-256 of neural-nets/race.onnx
//!
//! The revision is REQUIRED. It comes from `BG_ENGINE_SOURCE_REVISION` (any
//! builder), else `RENDER_GIT_COMMIT` (Render's Docker builds), else
//! `git rev-parse HEAD` when building from a checkout with git available.
//! A missing or malformed revision fails the build: there is no branch name
//! or "unversioned" fallback, because the identity in every reply must name
//! exactly the source that produced it.
//!
//! The net hashes are taken over the very files `crates/engine/src/onnx.rs`
//! embeds with `include_bytes!`, and the build re-runs when either file, the
//! revision variables, or the checkout's HEAD change, so the constants can
//! never describe a different binary than the one they are compiled into.
//! `contract::tests` re-hashes the embedded bytes and checks they agree.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

include!("src/revision_check.rs");

fn main() {
    println!("cargo:rerun-if-env-changed=BG_ENGINE_SOURCE_REVISION");
    println!("cargo:rerun-if-env-changed=RENDER_GIT_COMMIT");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir.join("..").join("..");

    // Local builds: re-run when HEAD moves.
    let git_head = workspace.join(".git").join("HEAD");
    println!("cargo:rerun-if-changed={}", git_head.display());
    if let Ok(head) = fs::read_to_string(&git_head)
        && let Some(reference) = head.trim().strip_prefix("ref: ")
    {
        println!(
            "cargo:rerun-if-changed={}",
            workspace.join(".git").join(reference.trim()).display()
        );
    }

    let revision = match resolve_revision(&workspace) {
        Ok(revision) => revision,
        Err(reason) => panic!(
            "bg-engine needs the full commit hash of the source it is built from, and none was usable: {reason}. \
             Set BG_ENGINE_SOURCE_REVISION=$(git rev-parse HEAD) (Render sets RENDER_GIT_COMMIT), \
             or build from a git checkout with git installed. There is no fallback label."
        ),
    };
    println!("cargo:rustc-env=BG_ENGINE_SOURCE_REVISION={revision}");

    for (name, file) in [("CONTACT", "contact.onnx"), ("RACE", "race.onnx")] {
        let path = workspace.join("neural-nets").join(file);
        println!("cargo:rerun-if-changed={}", path.display());
        let bytes = fs::read(&path)
            .unwrap_or_else(|e| panic!("cannot read {} to hash it: {e}", path.display()));
        let digest = Sha256::digest(&bytes);
        println!(
            "cargo:rustc-env=BG_ENGINE_{name}_SHA256={}",
            hex_lower(&digest)
        );
    }
}

fn resolve_revision(workspace: &Path) -> Result<String, String> {
    if let Ok(value) = env::var("BG_ENGINE_SOURCE_REVISION") {
        return validate_revision(&value).map_err(|e| format!("BG_ENGINE_SOURCE_REVISION: {e}"));
    }
    if let Ok(value) = env::var("RENDER_GIT_COMMIT") {
        return validate_revision(&value).map_err(|e| format!("RENDER_GIT_COMMIT: {e}"));
    }
    if workspace.join(".git").exists() {
        let output = Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(workspace)
            .output()
            .map_err(|e| format!("git rev-parse HEAD could not run: {e}"))?;
        if !output.status.success() {
            return Err(format!(
                "git rev-parse HEAD failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        return validate_revision(&String::from_utf8_lossy(&output.stdout))
            .map_err(|e| format!("git rev-parse HEAD: {e}"));
    }
    Err("no BG_ENGINE_SOURCE_REVISION, no RENDER_GIT_COMMIT, and no git checkout".to_string())
}

fn hex_lower(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    out
}
