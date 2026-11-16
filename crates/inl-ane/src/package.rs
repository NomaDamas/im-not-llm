//! Model packages: where they come from, what they hold, and turning one into
//! a compiled Core ML model on this Mac.
//!
//! A package is a directory with `manifest.json` (function signatures and
//! every file's size and sha256), the text `.mlpackage`, the token table, the
//! tokenizer and a self-test. Known models resolve to a Hub repo pinned at a
//! commit and to the sha256 of that commit's manifest, so every byte that runs
//! is checked against a hash compiled into this crate.

use crate::layout::Family;
use inl_core::hub::hf_repo_at;
use inl_core::{InlError, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

/// A model this crate has a pinned package for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KnownPackage {
    /// The original checkpoint id callers name (`--model`).
    pub model: &'static str,
    /// Hub repo of the Core ML package.
    pub repo: &'static str,
    pub revision: &'static str,
    pub manifest_sha256: &'static str,
}

/// Packages by model; the package repo id resolves too.
pub const KNOWN_PACKAGES: &[KnownPackage] = &[KnownPackage {
    model: "google/embeddinggemma-2",
    // ANEEmbed's package (https://github.com/junmo-kim/ANEEmbed).
    repo: "junmo-k/embeddinggemma-2-ane",
    revision: "e97c928a49cbc0cd61974ca628b14a1cd9b03d66",
    manifest_sha256: "b21e57adf6931e3838d5a1e346befcf2e13adfb43906cb7bad1dff1274912d8a",
}];

pub(crate) fn known(id: &str) -> Option<&'static KnownPackage> {
    KNOWN_PACKAGES
        .iter()
        .find(|package| package.model == id || package.repo == id)
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Manifest {
    pub name: String,
    pub embedding: EmbeddingSpec,
    pub token_table: TokenTableSpec,
    pub packages: BTreeMap<String, PackageSpec>,
    pub files: Vec<FileSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct EmbeddingSpec {
    pub dim: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct TokenTableSpec {
    pub file: String,
    pub shape: [usize; 2],
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct PackageSpec {
    pub mlpackage: String,
    pub functions: BTreeMap<String, FunctionSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct FunctionSpec {
    pub inputs: Vec<TensorSpec>,
    pub outputs: Vec<TensorSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

impl TensorSpec {
    pub(crate) fn elements(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct FileSpec {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

/// A resolved package: its files are present under `root`.
pub(crate) struct Package {
    pub root: PathBuf,
    pub manifest: Manifest,
    pub manifest_sha256: String,
    pub family: Family,
}

impl Package {
    /// The text package of the manifest.
    pub(crate) fn text(&self) -> &PackageSpec {
        &self.manifest.packages[self.family.package_name()]
    }

    pub(crate) fn path(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    /// The files the text engine reads: the text `.mlpackage`, token table,
    /// tokenizer and self-test (not the vision tower or a prebuilt
    /// `.mlmodelc`, which is compiled for another OS build).
    fn needed_files(&self) -> Vec<&FileSpec> {
        let mlpackage = format!("{}/", self.text().mlpackage);
        let singles = [
            self.manifest.token_table.file.as_str(),
            "tokenizer.json",
            "selftest.json",
            "selftest.bin",
        ];
        self.manifest
            .files
            .iter()
            .filter(|file| {
                file.path.starts_with(&mlpackage) || singles.contains(&file.path.as_str())
            })
            .collect()
    }
}

fn parse_manifest(path: &Path) -> Result<(Manifest, String, Family)> {
    let bytes = std::fs::read(path)?;
    let sha = hex::encode(Sha256::digest(&bytes));
    let manifest: Manifest = serde_json::from_slice(&bytes).map_err(|error| {
        InlError::Inference(format!(
            "{}: invalid package manifest: {error}",
            path.display()
        ))
    })?;
    let family = manifest
        .packages
        .keys()
        .find_map(|name| Family::from_package_name(name))
        .ok_or_else(|| {
            InlError::Config(format!(
                "{}: no text package this engine runs (packages: {})",
                path.display(),
                manifest
                    .packages
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;
    Ok((manifest, sha, family))
}

/// Find the package for `id` (a known model, its package repo, or a local
/// package directory) and make its files present. Blocking: may download.
pub(crate) fn resolve(id: &str, cache_dir: PathBuf) -> Result<Package> {
    let local = Path::new(id);
    if local.is_dir() {
        let (manifest, manifest_sha256, family) = parse_manifest(&local.join("manifest.json"))?;
        let package = Package {
            root: local.to_path_buf(),
            manifest,
            manifest_sha256,
            family,
        };
        for file in package.needed_files() {
            if !package.path(&file.path).is_file() {
                return Err(InlError::Inference(format!(
                    "package file {} is missing from {}",
                    file.path,
                    local.display()
                )));
            }
        }
        return Ok(package);
    }
    let Some(known) = known(id) else {
        return Err(InlError::Config(format!(
            "no Apple Neural Engine package for {id}: use one of {} or a local package directory",
            KNOWN_PACKAGES
                .iter()
                .map(|package| package.model)
                .collect::<Vec<_>>()
                .join(", ")
        )));
    };
    let repo = hf_repo_at(known.repo, known.revision, cache_dir)?;
    let fetch = |name: &str| {
        repo.get(name)
            .map_err(|error| InlError::Inference(format!("ANE package {error}")))
    };
    let manifest_path = fetch("manifest.json")?;
    let (manifest, manifest_sha256, family) = parse_manifest(&manifest_path)?;
    if manifest_sha256 != known.manifest_sha256 {
        return Err(InlError::Inference(format!(
            "{}@{}: manifest.json sha256 {manifest_sha256} is not the pinned {}",
            known.repo, known.revision, known.manifest_sha256
        )));
    }
    let root = manifest_path
        .parent()
        .expect("a cached file has a snapshot directory")
        .to_path_buf();
    let package = Package {
        root,
        manifest,
        manifest_sha256,
        family,
    };
    let names: Vec<String> = package
        .needed_files()
        .iter()
        .map(|file| file.path.clone())
        .collect();
    for name in names {
        tracing::debug!(repo = known.repo, file = %name, "fetching ANE package file");
        fetch(&name)?;
    }
    Ok(package)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Check every needed file against the manifest: the size always, the
/// sha256 unless `sizes_only` (a stamp says this manifest was hashed before).
pub(crate) fn verify(package: &Package, sizes_only: bool) -> Result<()> {
    for file in package.needed_files() {
        let path = package.path(&file.path);
        let size = std::fs::metadata(&path)?.len();
        if size != file.size {
            return Err(InlError::Inference(format!(
                "package file {} is {size} bytes, the manifest says {}",
                file.path, file.size
            )));
        }
        if !sizes_only {
            let sha = sha256_file(&path)?;
            if sha != file.sha256 {
                return Err(InlError::Inference(format!(
                    "package file {} has sha256 {sha}, the manifest says {}",
                    file.path, file.sha256
                )));
            }
        }
    }
    Ok(())
}

/// Where this package's compiled model lives: one directory per manifest, so
/// a new package revision never reuses an old compile.
pub(crate) fn compiled_dir(package: &Package, cache_dir: &Path) -> PathBuf {
    cache_dir.join("ane").join(format!(
        "{}-{}",
        package.manifest.name,
        &package.manifest_sha256[..16]
    ))
}

/// Copy `from` (a directory whose files may be Hub-cache symlinks) to `to`
/// with real files: Core ML's compiler rejects symlinked weights. Hard links
/// where the volume allows, else copies.
pub(crate) fn materialize(from: &Path, to: &Path) -> Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let source = entry.path();
        let target = to.join(entry.file_name());
        let resolved = std::fs::canonicalize(&source)?;
        if resolved.is_dir() {
            materialize(&resolved, &target)?;
        } else if std::fs::hard_link(&resolved, &target).is_err() {
            std::fs::copy(&resolved, &target)?;
        }
    }
    Ok(())
}

/// Recursive copy of a plain directory tree.
pub(crate) fn copy_dir(from: &Path, to: &Path) -> Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_package(dir: &Path, weights: &[u8], claimed: &[u8]) {
        let mlpackage = dir.join("EmbeddingGemma2Text.mlpackage");
        std::fs::create_dir_all(&mlpackage).unwrap();
        std::fs::write(mlpackage.join("weight.bin"), weights).unwrap();
        for name in [
            "embeddings.bf16",
            "tokenizer.json",
            "selftest.json",
            "selftest.bin",
        ] {
            std::fs::write(dir.join(name), b"x").unwrap();
        }
        let entry = |path: &str, bytes: &[u8]| serde_json::json!({"path": path, "size": bytes.len(), "sha256": hex::encode(Sha256::digest(bytes))});
        let mut files = vec![entry("EmbeddingGemma2Text.mlpackage/weight.bin", claimed)];
        for name in [
            "embeddings.bf16",
            "tokenizer.json",
            "selftest.json",
            "selftest.bin",
        ] {
            files.push(entry(name, b"x"));
        }
        let manifest = serde_json::json!({
            "name": "test-ane",
            "embedding": {"dim": 768},
            "token_table": {"file": "embeddings.bf16", "shape": [1, 512]},
            "packages": {"EmbeddingGemma2Text": {"mlpackage": "EmbeddingGemma2Text.mlpackage", "functions": {}}},
            "files": files,
        });
        std::fs::write(dir.join("manifest.json"), manifest.to_string()).unwrap();
    }

    #[test]
    fn a_local_package_is_verified_against_its_manifest() {
        let dir = tempfile::tempdir().unwrap();
        write_package(dir.path(), b"weights", b"weights");
        let package = resolve(dir.path().to_str().unwrap(), dir.path().join("cache")).unwrap();
        assert_eq!(package.family, Family::Gemma2);
        verify(&package, false).unwrap();
    }

    #[test]
    fn a_file_that_differs_from_the_manifest_is_rejected_by_name() {
        let dir = tempfile::tempdir().unwrap();
        write_package(dir.path(), b"weights", b"WEIGHTS");
        let package = resolve(dir.path().to_str().unwrap(), dir.path().join("cache")).unwrap();
        // Same size: only the hash catches it.
        verify(&package, true).unwrap();
        let error = verify(&package, false).unwrap_err().to_string();
        assert!(error.contains("weight.bin"), "{error}");
    }

    #[test]
    fn an_unknown_model_names_the_supported_ones() {
        let dir = tempfile::tempdir().unwrap();
        let error = resolve("org/not-a-package", dir.path().to_path_buf())
            .err()
            .unwrap();
        assert!(matches!(&error, InlError::Config(_)), "{error}");
        assert!(
            error.to_string().contains("google/embeddinggemma-2"),
            "{error}"
        );
    }

    #[test]
    fn a_model_resolves_by_checkpoint_and_by_package_repo() {
        assert_eq!(
            known("google/embeddinggemma-2"),
            known("junmo-k/embeddinggemma-2-ane")
        );
        assert!(known("google/embeddinggemma-2").is_some());
    }

    #[test]
    fn materialize_replaces_symlinks_with_real_files() {
        let dir = tempfile::tempdir().unwrap();
        let blob = dir.path().join("blob");
        std::fs::write(&blob, b"data").unwrap();
        let source = dir.path().join("pkg/Data");
        std::fs::create_dir_all(&source).unwrap();
        std::os::unix::fs::symlink(&blob, source.join("weight.bin")).unwrap();
        let target = dir.path().join("out");
        materialize(&dir.path().join("pkg"), &target).unwrap();
        let copied = target.join("Data/weight.bin");
        assert!(!std::fs::symlink_metadata(&copied)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(std::fs::read(copied).unwrap(), b"data");
    }
}
