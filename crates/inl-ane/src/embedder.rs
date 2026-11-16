//! [`AneEmbedder`]: a [`DenseEngine`] over a model package on the Neural
//! Engine.
//!
//! Load: resolve the package (download and check every byte against the
//! pinned manifest), compile it for this Mac once, load each function the
//! context length needs, and run the package's self-test on every one of
//! them (the Neural Engine compiler has miscompiled graphs before; a package
//! that does not reproduce its reference embeddings is never served).
//!
//! Embed: tokenize, plan calls that pack short texts together, fill each
//! call's inputs and run it. Calls run one at a time: the Neural Engine is a
//! single device, so concurrency would only interleave them.

use crate::coreml::{self, Function};
use crate::layout::{self, Call, Family, Segment};
use crate::package::{self, Package};
use crate::table::TokenTable;
use async_trait::async_trait;
use inl_core::deadline::BatchDeadline;
use inl_core::device::{model_cache_dir, LoadSlot};
use inl_core::{DenseEngine, EngineInfo, InlError, Modality, Result};
use parking_lot::Mutex;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokenizers::{PostProcessor, Tokenizer, TruncationDirection};

/// Context length when the options name none (the native engines' default).
pub const DEFAULT_MAX_LENGTH: usize = 2048;
/// Deadline of one Neural Engine call.
pub const DEFAULT_TIMEOUT_SECONDS: u64 = 120;

/// Settings of an [`AneEmbedder`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AneOptions {
    /// A known model id (see [`crate::KNOWN_PACKAGES`]), its package repo,
    /// or a local package directory.
    pub id: String,
    /// Model cache directory; `None` uses the shared cache.
    pub cache_dir: Option<PathBuf>,
    /// Context length in tokens; capped at the longest function the package
    /// has. Longer texts are truncated.
    pub max_length: usize,
    /// Deadline of one Neural Engine call, in seconds.
    pub timeout_seconds: u64,
}

impl Default for AneOptions {
    fn default() -> Self {
        Self {
            id: String::new(),
            cache_dir: None,
            max_length: DEFAULT_MAX_LENGTH,
            timeout_seconds: DEFAULT_TIMEOUT_SECONDS,
        }
    }
}

/// A Neural Engine call that never returns strands its thread; every
/// embedder in the process then fails fast.
static ANE_DEADLINE: BatchDeadline = BatchDeadline::new();

pub struct AneEmbedder {
    options: AneOptions,
    slot: LoadSlot<Arc<Loaded>>,
    /// Serializes requests ahead of the per-call deadline.
    gate: tokio::sync::Mutex<()>,
}

struct Loaded {
    family: Family,
    tokenizer: Tokenizer,
    /// Tokens the tokenizer adds around a text (bos/eos).
    specials: usize,
    max_length: usize,
    lengths: Vec<usize>,
    dimension: usize,
    table: TokenTable,
    functions: Mutex<BTreeMap<usize, Function>>,
}

impl AneEmbedder {
    pub fn new(options: AneOptions) -> Result<Self> {
        if options.id.is_empty() {
            return Err(InlError::Config(
                "ANE embedder id is missing a model name".into(),
            ));
        }
        if options.max_length == 0 {
            return Err(InlError::Config("max_length must be greater than 0".into()));
        }
        Ok(Self {
            options,
            slot: LoadSlot::new(),
            gate: tokio::sync::Mutex::new(()),
        })
    }

    async fn loaded(&self) -> Result<Arc<Loaded>> {
        self.slot
            .get_or_load(|| async {
                let options = self.options.clone();
                tokio::task::spawn_blocking(move || Loaded::load(&options).map(Arc::new))
                    .await
                    .map_err(|error| {
                        InlError::Inference(format!("ANE load join failed: {error}"))
                    })?
            })
            .await
    }

    fn budget(&self) -> Duration {
        Duration::from_secs(self.options.timeout_seconds)
    }

    async fn embed_texts(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        ANE_DEADLINE.ensure_healthy("ANE embedding")?;
        let loaded = self.loaded().await?;
        let _gate = self.gate.lock().await;
        let tokenizing = Arc::clone(&loaded);
        let ids = Arc::new(
            ANE_DEADLINE
                .run("ANE tokenization", self.budget(), move || {
                    tokenizing.token_ids(&texts)
                })
                .await?,
        );
        let counts: Vec<usize> = ids.iter().map(Vec::len).collect();
        let mut vectors: Vec<Vec<f32>> = vec![Vec::new(); ids.len()];
        for call in layout::plan(loaded.family, &loaded.lengths, &counts) {
            let running = Arc::clone(&loaded);
            let call_ids = Arc::clone(&ids);
            let rows = ANE_DEADLINE
                .run("ANE embedding", self.budget(), move || {
                    running.embed_call(&call, &call_ids)
                })
                .await?;
            for (item, vector) in rows {
                vectors[item] = vector;
            }
        }
        Ok(vectors)
    }
}

impl Loaded {
    fn load(options: &AneOptions) -> Result<Self> {
        if !cfg!(target_arch = "aarch64") {
            return Err(InlError::Config(
                "the Apple Neural Engine needs an Apple silicon Mac".into(),
            ));
        }
        let cache = model_cache_dir(options.cache_dir.as_deref());
        let package = package::resolve(&options.id, cache.clone())?;
        let mlmodelc = compiled_model(&package, &cache)?;
        let family = package.family;
        let mut tokenizer = Tokenizer::from_file(package.path("tokenizer.json"))
            .map_err(|error| InlError::Inference(format!("ANE tokenizer parse failed: {error}")))?;
        tokenizer
            .with_truncation(None)
            .map_err(|error| InlError::Inference(format!("ANE tokenizer: {error}")))?;
        tokenizer.with_padding(None);
        let specials = tokenizer
            .get_post_processor()
            .map_or(0, |processor| processor.added_tokens(false));
        let longest = *family
            .lengths_for(usize::MAX)
            .last()
            .expect("a family has lengths");
        let max_length = options.max_length.min(longest);
        let lengths = family.lengths_for(max_length);
        let [_, hidden] = package.manifest.token_table.shape;
        let table = TokenTable::open(
            &package.path(&package.manifest.token_table.file),
            hidden,
            family.row_scale(),
        )?;
        let mut functions = BTreeMap::new();
        for &length in &lengths {
            let name = family.function_name(length);
            let spec = package.text().functions.get(&name).ok_or_else(|| {
                InlError::Inference(format!(
                    "package {} has no function {name}",
                    package.manifest.name
                ))
            })?;
            let names: Vec<&str> = spec
                .inputs
                .iter()
                .map(|input| input.name.as_str())
                .collect();
            if names != family.input_names(length) {
                return Err(InlError::Inference(format!(
                    "function {name} takes {names:?}, expected {:?}",
                    family.input_names(length)
                )));
            }
            let started = std::time::Instant::now();
            functions.insert(length, Function::load(&mlmodelc, &name, spec)?);
            tracing::debug!(function = %name, seconds = started.elapsed().as_secs_f32(), "ANE function loaded");
        }
        let loaded = Self {
            family,
            tokenizer,
            specials,
            max_length,
            lengths,
            dimension: package.manifest.embedding.dim,
            table,
            functions: Mutex::new(functions),
        };
        loaded.self_test(&package)?;
        Ok(loaded)
    }

    /// Token ids per text, truncated to `max_length` with the special tokens
    /// kept (so a last-token or eos-sensitive pooling still sees them).
    fn token_ids(&self, texts: &[String]) -> Result<Vec<Vec<u32>>> {
        let room = self.max_length.saturating_sub(self.specials).max(1);
        texts
            .iter()
            .map(|text| {
                let mut content = self
                    .tokenizer
                    .encode(text.as_str(), false)
                    .map_err(|error| InlError::Inference(format!("ANE tokenizer: {error}")))?;
                content.truncate(room, 0, TruncationDirection::Right);
                let encoding = self
                    .tokenizer
                    .post_process(content, None, true)
                    .map_err(|error| InlError::Inference(format!("ANE tokenizer: {error}")))?;
                let ids = encoding.get_ids().to_vec();
                if ids.is_empty() {
                    return Err(InlError::InvalidInput("a text with no tokens".into()));
                }
                Ok(ids)
            })
            .collect()
    }

    fn token_counts(&self, texts: &[String]) -> Result<Vec<usize>> {
        texts
            .iter()
            .map(|text| {
                self.tokenizer
                    .encode(text.as_str(), false)
                    .map(|encoding| encoding.len() + self.specials)
                    .map_err(|error| InlError::Inference(format!("ANE tokenizer: {error}")))
            })
            .collect()
    }

    /// Run one call; returns the whole `[K, dim]` output, row k = segment k.
    fn run(&self, call: &Call, ids: &[Vec<u32>]) -> Result<Vec<f32>> {
        let mut functions = self.functions.lock();
        let function = functions
            .get_mut(&call.length)
            .expect("calls are planned over loaded lengths");
        let hidden = self.table.hidden();
        let embeds = function.input("inputs_embeds");
        for segment in &call.segments {
            let rows =
                &mut embeds[segment.offset * hidden..(segment.offset + segment.len) * hidden];
            self.table.gather(&ids[segment.item], rows)?;
        }
        let used: usize = call.segments.iter().map(|segment| segment.len).sum();
        embeds[used * hidden..].fill(half::f16::ZERO);
        for &name in &self.family.input_names(call.length)[1..] {
            layout::fill_input(
                self.family,
                call.length,
                name,
                &call.segments,
                function.input(name),
            );
        }
        let mut out = Vec::new();
        function.predict(&mut out)?;
        Ok(out)
    }

    fn embed_call(&self, call: &Call, ids: &[Vec<u32>]) -> Result<Vec<(usize, Vec<f32>)>> {
        let out = self.run(call, ids)?;
        Ok(call
            .segments
            .iter()
            .enumerate()
            .map(|(slot, segment)| {
                let row = out[slot * self.dimension..(slot + 1) * self.dimension].to_vec();
                (segment.item, row)
            })
            .collect())
    }

    /// Every loaded function reproduces the package's reference embeddings.
    fn self_test(&self, package: &Package) -> Result<()> {
        let raw = std::fs::read(package.path("selftest.json"))?;
        let spec: SelfTest = serde_json::from_slice(&raw)?;
        let blob = std::fs::read(package.path(&spec.bin))?;
        let names: BTreeMap<String, usize> = self
            .lengths
            .iter()
            .map(|&length| (self.family.function_name(length), length))
            .collect();
        let mut tested = 0;
        let mut worst = f64::INFINITY;
        for test in &spec.tests {
            let Some(&length) = names.get(&test.function) else {
                continue;
            };
            let sequences: Vec<Vec<u32>> = match (&test.sequences, test.doc_rows) {
                (Some(sequences), _) => sequences.clone(),
                (None, Some(rows)) => {
                    let doc = spec.doc_ids.as_ref().ok_or_else(|| {
                        InlError::Inference("self-test has doc_rows but no doc_ids".into())
                    })?;
                    vec![doc[..rows].to_vec()]
                }
                (None, None) => continue,
            };
            let mut offset = 0;
            let segments = sequences
                .iter()
                .enumerate()
                .map(|(item, ids)| {
                    let segment = Segment {
                        item,
                        offset,
                        len: ids.len(),
                    };
                    offset += ids.len();
                    segment
                })
                .collect();
            let call = Call { length, segments };
            let out = self.run(&call, &sequences)?;
            let expected = test.expected.read(&blob)?;
            let tolerance = spec.tolerance.get(&test.kind).ok_or_else(|| {
                InlError::Inference(format!("self-test has no tolerance for {}", test.kind))
            })?;
            let dim = self.dimension;
            for (k, want) in expected.chunks_exact(dim).enumerate() {
                let cos = cosine(&out[k * dim..(k + 1) * dim], want);
                worst = worst.min(cos);
                if cos < tolerance.min_cos {
                    return Err(InlError::Inference(format!(
                        "ANE self-test failed: {} text {k} has cosine {cos:.6} to the reference (minimum {})",
                        test.function, tolerance.min_cos
                    )));
                }
            }
            if let Some(limit) = tolerance.zero_rows_max_abs {
                let spare = out[sequences.len() * dim..]
                    .iter()
                    .fold(0f32, |max, value| max.max(value.abs()));
                if f64::from(spare) > limit {
                    return Err(InlError::Inference(format!(
                        "ANE self-test failed: {} unused rows reach {spare} (maximum {limit})",
                        test.function
                    )));
                }
            }
            tested += 1;
        }
        if tested != self.lengths.len() {
            return Err(InlError::Inference(format!(
                "ANE self-test covers {tested} of {} loaded functions",
                self.lengths.len()
            )));
        }
        tracing::info!(
            functions = tested,
            worst_cosine = worst,
            "ANE self-test passed"
        );
        Ok(())
    }
}

/// The compiled model of `package`, compiling it on first use. The package
/// files are hashed once per manifest; later loads check their sizes.
fn compiled_model(package: &Package, cache: &Path) -> Result<PathBuf> {
    let dir = package::compiled_dir(package, cache);
    let name = package.family.package_name();
    let mlmodelc = dir.join(format!("{name}.mlmodelc"));
    let stamp = dir.join("verified");
    let fresh = mlmodelc.is_dir()
        && std::fs::read_to_string(&stamp).is_ok_and(|sha| sha == package.manifest_sha256);
    package::verify(package, fresh)?;
    if fresh {
        return Ok(mlmodelc);
    }
    tracing::info!(
        package = %package.manifest.name,
        into = %dir.display(),
        "compiling the Core ML package for this Mac (first use; the Neural Engine compile of each function follows)"
    );
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    let mlpackage = dir.join(format!("{name}.mlpackage"));
    package::materialize(&package.path(&package.text().mlpackage), &mlpackage)?;
    let compiled = coreml::compile(&mlpackage)?;
    let partial = dir.join(format!("{name}.mlmodelc.partial"));
    package::copy_dir(&compiled, &partial)?;
    std::fs::rename(&partial, &mlmodelc)?;
    let _ = std::fs::remove_dir_all(&compiled);
    std::fs::remove_dir_all(&mlpackage)?;
    std::fs::write(&stamp, &package.manifest_sha256)?;
    Ok(mlmodelc)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE)
}

#[derive(Deserialize)]
struct SelfTest {
    bin: String,
    tolerance: BTreeMap<String, Tolerance>,
    doc_ids: Option<Vec<u32>>,
    tests: Vec<SelfTestCase>,
}

#[derive(Deserialize)]
struct Tolerance {
    min_cos: f64,
    zero_rows_max_abs: Option<f64>,
}

#[derive(Deserialize)]
struct SelfTestCase {
    function: String,
    kind: String,
    sequences: Option<Vec<Vec<u32>>>,
    doc_rows: Option<usize>,
    expected: BlobRef,
}

#[derive(Deserialize)]
struct BlobRef {
    offset: usize,
    shape: Vec<usize>,
}

impl BlobRef {
    fn read(&self, blob: &[u8]) -> Result<Vec<f32>> {
        let count: usize = self.shape.iter().product();
        let bytes = blob
            .get(self.offset..self.offset + count * 4)
            .ok_or_else(|| InlError::Inference("self-test blob is too short".into()))?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }
}

#[async_trait]
impl DenseEngine for AneEmbedder {
    async fn load(&self) -> Result<EngineInfo> {
        ANE_DEADLINE.ensure_healthy("ANE embedding")?;
        let loaded = self.loaded().await?;
        Ok(EngineInfo {
            model_id: self.options.id.clone(),
            dimension: loaded.dimension,
            dtype: "f16".to_string(),
            device: "ane".to_string(),
            attention: "packed".to_string(),
            max_length: loaded.max_length,
            modalities: vec![Modality::Text],
        })
    }

    async fn token_counts(&self, texts: &[String]) -> Result<Vec<usize>> {
        let loaded = self.loaded().await?;
        let texts = texts.to_vec();
        tokio::task::spawn_blocking(move || loaded.token_counts(&texts))
            .await
            .map_err(|error| InlError::Inference(format!("ANE tokenizer join failed: {error}")))?
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        self.embed_texts(texts).await
    }

    async fn unload(&self) -> bool {
        self.slot.unload().await
    }

    fn is_loaded(&self) -> bool {
        self.slot.is_loaded()
    }
}
