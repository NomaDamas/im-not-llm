//! Dense text embeddings on the Apple Neural Engine through Core ML.
//!
//! The model runs as a precompiled Core ML package (fp16 graphs, one function
//! per sequence length; texts shorter than a function are packed together
//! into one call). Known models resolve to a Hub package pinned by revision
//! and manifest sha256 ([`KNOWN_PACKAGES`]); every load self-tests each
//! function against reference embeddings before serving.
//!
//! The package contract and packing rules (`layout`), the package files
//! (`package`) and the token table (`table`) are plain Rust; Core ML itself
//! (`coreml`, [`AneEmbedder`]) exists on macOS only.

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
pub use embedder::{AneEmbedder, AneOptions, DEFAULT_MAX_LENGTH, DEFAULT_TIMEOUT_SECONDS};

/// The pinned package of `model` (a checkpoint id or its package repo).
pub fn known_package(model: &str) -> Option<&'static KnownPackage> {
    package::known(model)
}
