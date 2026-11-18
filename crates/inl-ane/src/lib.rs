//! Dense and learned-sparse text embeddings on the Apple Neural Engine
//! through Core ML.
//!
//! The model runs as a precompiled Core ML package (fp16 graphs, one function
//! per sequence length; texts shorter than a function are packed together
//! into one call). Known models resolve to a Hub package pinned by revision
//! and manifest sha256 ([`KNOWN_PACKAGES`]); each function is loaded by the
//! first request that needs it and self-tested against reference embeddings
//! before it serves anything, so a cold start pays for the one or two
//! functions it uses, not for all of them.
//!
//! The package contract and packing rules (`layout`), the package files
//! (`package`) and the token table (`table`) are plain Rust; Core ML itself
//! (`coreml`, [`AneEmbedder`], [`AneSparseEncoder`]) exists on macOS only.

mod layout;
mod package;
mod table;

#[cfg(target_os = "macos")]
mod coreml;
#[cfg(target_os = "macos")]
mod embedder;

pub use layout::Family;
pub use package::{KnownPackage, KNOWN_PACKAGES};

#[cfg(target_os = "macos")]
pub use embedder::{
    AneEmbedder, AneOptions, AneSparseEncoder, DEFAULT_MAX_LENGTH, DEFAULT_TIMEOUT_SECONDS,
};

/// The pinned package of `model` (a checkpoint id or its package repo).
pub fn known_package(model: &str) -> Option<&'static KnownPackage> {
    package::known(model)
}

/// The family of the package `model` names (a known model, its package
/// repo, or a local package directory) without downloading anything; `None`
/// when it names no package.
pub fn package_family(model: &str) -> inl_core::Result<Option<Family>> {
    package::family_of(model)
}
