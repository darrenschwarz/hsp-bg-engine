//! The sidecar's wire contract (GP-476): who this binary is, what it can do,
//! and what "healthy" means.
//!
//! Every reply -- `/health` and each `/rank`, `/evaluate` and `/cube` batch --
//! carries the same `ContractMeta`, built once at startup from constants the
//! build script baked in (see `build.rs`). Nothing at runtime can change it:
//! no environment variable, no branch label, no "unpinned".
//!
//! Identity format, which the HSP client validates structurally:
//!
//!     wildbg@<full source commit hash>+contact@<sha256>+race@<sha256>
//!
//! The hashes are over the exact bytes of `neural-nets/contact.onnx` and
//! `neural-nets/race.onnx` that `crates/engine` compiles in with
//! `include_bytes!`, so a different net is a different identity.

use std::ops::RangeInclusive;

use serde::Serialize;

pub mod revision_check {
    include!("revision_check.rs");
}

/// Bumped only when the wire shape changes incompatibly. Clients require an
/// exact match; they must not treat a newer version as compatible.
pub const API_VERSION: u32 = 1;

/// What this build implements. Each name is a promise the handlers keep:
/// `rank.v1`, `evaluate.v1`, `cube.money.v1` are the three POST endpoints
/// with their v1 shapes (cube is money-game only), and `plies.1` / `plies.2`
/// are the search depths `/rank` will actually run. `rank.match.v1` is the
/// additive 1-ply match-context scorer; match requests at 2-ply are refused
/// with typed `unsupported_checker_context` until every reply ply is MWC-aware.
pub const CAPABILITIES: [&str; 6] = [
    "rank.v1",
    "rank.match.v1",
    "evaluate.v1",
    "cube.money.v1",
    "plies.1",
    "plies.2",
];

/// Where the side to move's win probability from the opening position must
/// land for the loaded evaluator to be believed. The true value is a shade
/// over 0.5 (the mover is on roll); anything outside this band means the
/// wrong nets, a broken load, or a mis-oriented board -- not a usable engine.
pub const OPENING_WIN_RANGE: RangeInclusive<f32> = 0.45..=0.55;

/// The three facts an identity is made of.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    pub source_revision: &'static str,
    pub contact_sha256: &'static str,
    pub race_sha256: &'static str,
}

/// This binary's identity, as baked in by `build.rs`.
pub const BUILD_IDENTITY: Identity = Identity {
    source_revision: env!("BG_ENGINE_SOURCE_REVISION"),
    contact_sha256: env!("BG_ENGINE_CONTACT_SHA256"),
    race_sha256: env!("BG_ENGINE_RACE_SHA256"),
};

pub fn engine_id(identity: &Identity) -> String {
    format!(
        "wildbg@{}+contact@{}+race@{}",
        identity.source_revision, identity.contact_sha256, identity.race_sha256
    )
}

/// The metadata every reply carries. Immutable once built; cloned into each
/// response so every reply says exactly the same thing.
#[derive(Serialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ContractMeta {
    pub api_version: u32,
    pub capabilities: Vec<&'static str>,
    pub engine_id: String,
}

impl ContractMeta {
    pub fn for_identity(identity: &Identity) -> Self {
        Self {
            api_version: API_VERSION,
            capabilities: CAPABILITIES.to_vec(),
            engine_id: engine_id(identity),
        }
    }

    pub fn from_build() -> Self {
        // The build script already refused anything but a full hash; checking
        // again here means a binary can never START with a malformed identity
        // either, whatever produced it.
        revision_check::validate_revision(BUILD_IDENTITY.source_revision).unwrap_or_else(|e| {
            panic!("the source revision baked into this binary is unusable: {e}")
        });
        Self::for_identity(&BUILD_IDENTITY)
    }
}

/// The health verdict for an opening-position win probability computed by
/// the loaded evaluator. `Ok` means ready; `Err` carries the reason `/health`
/// reports alongside HTTP 503.
pub fn health_check(opening_win: f32) -> Result<(), String> {
    if !opening_win.is_finite() {
        return Err(format!(
            "opening win probability is not finite ({opening_win})"
        ));
    }
    if !OPENING_WIN_RANGE.contains(&opening_win) {
        return Err(format!(
            "opening win probability {opening_win} outside plausible range {}..={}",
            OPENING_WIN_RANGE.start(),
            OPENING_WIN_RANGE.end()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn hex_lower(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn is_lower_hex(text: &str, len: usize) -> bool {
        text.len() == len && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    }

    #[test]
    fn the_build_identity_is_a_full_revision_and_two_full_net_hashes() {
        let revision = BUILD_IDENTITY.source_revision;
        assert!(
            is_lower_hex(revision, 40) || is_lower_hex(revision, 64),
            "source revision is not a full lowercase commit hash: {revision:?}"
        );
        assert!(is_lower_hex(BUILD_IDENTITY.contact_sha256, 64));
        assert!(is_lower_hex(BUILD_IDENTITY.race_sha256, 64));
    }

    #[test]
    fn the_baked_hashes_are_over_the_exact_bytes_include_bytes_compiles_in() {
        // The same paths crates/engine/src/onnx.rs embeds.
        let contact = include_bytes!("../../../neural-nets/contact.onnx");
        let race = include_bytes!("../../../neural-nets/race.onnx");
        assert_eq!(
            hex_lower(&Sha256::digest(contact)),
            BUILD_IDENTITY.contact_sha256
        );
        assert_eq!(hex_lower(&Sha256::digest(race)), BUILD_IDENTITY.race_sha256);
    }

    #[test]
    fn the_engine_id_has_the_documented_structure_and_no_mutable_labels() {
        let id = ContractMeta::from_build().engine_id;
        let rest = id.strip_prefix("wildbg@").expect("wildbg@ prefix");
        let mut parts = rest.split('+');
        let revision = parts.next().unwrap();
        let contact = parts
            .next()
            .unwrap()
            .strip_prefix("contact@")
            .expect("contact@");
        let race = parts.next().unwrap().strip_prefix("race@").expect("race@");
        assert!(parts.next().is_none());
        assert!(is_lower_hex(revision, 40) || is_lower_hex(revision, 64));
        assert!(is_lower_hex(contact, 64));
        assert!(is_lower_hex(race, 64));
        for forbidden in ["main", "unpinned", "nets/", "unversioned"] {
            assert!(
                !id.contains(forbidden),
                "{id} carries the mutable label {forbidden:?}"
            );
        }
    }

    #[test]
    fn the_metadata_is_the_same_object_every_time_and_advertises_only_what_exists() {
        let a = ContractMeta::from_build();
        let b = ContractMeta::from_build();
        assert_eq!(a, b);
        assert_eq!(a.api_version, 1);
        assert_eq!(
            a.capabilities,
            vec![
                "rank.v1",
                "rank.match.v1",
                "evaluate.v1",
                "cube.money.v1",
                "plies.1",
                "plies.2"
            ]
        );
    }

    #[test]
    fn revisions_are_full_hashes_or_nothing() {
        let sha1 = "f1d2d2f924e986ac86fdf7b36c94bcdf32beec15";
        assert_eq!(revision_check::validate_revision(sha1).unwrap(), sha1);
        assert_eq!(
            revision_check::validate_revision(&format!("  {}\n", sha1.to_ascii_uppercase()))
                .unwrap(),
            sha1
        );
        let sha256 = "a".repeat(64);
        assert_eq!(revision_check::validate_revision(&sha256).unwrap(), sha256);
        for bad in [
            "",
            "   ",
            "main",
            "nets",
            "unpinned",
            "f1d2d2f",
            "f1d2d2f924e986ac86fdf7b36c94bcdf32beec1g",
            "main+nets/unpinned",
        ] {
            assert!(
                revision_check::validate_revision(bad).is_err(),
                "{bad:?} was accepted"
            );
        }
    }

    #[test]
    fn health_accepts_the_plausible_band_and_nothing_else() {
        assert!(health_check(0.45).is_ok());
        assert!(health_check(0.5).is_ok());
        assert!(health_check(0.52).is_ok());
        assert!(health_check(0.55).is_ok());
        assert_eq!(
            health_check(0.875).unwrap_err(),
            "opening win probability 0.875 outside plausible range 0.45..=0.55"
        );
        assert!(health_check(0.3).is_err());
        assert!(health_check(0.0).is_err());
        assert!(health_check(1.0).is_err());
        assert!(health_check(f32::NAN).is_err());
        assert!(health_check(f32::INFINITY).is_err());
    }
}
