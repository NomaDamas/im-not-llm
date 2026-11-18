//! [`AneEmbedder`] (a [`DenseEngine`]) and [`AneSparseEncoder`] (a
//! [`SparseEngine`]) over a model package on the Neural Engine.
//!
//! Load: resolve the package (download and check every byte against the
//! pinned manifest), compile it for this Mac once, load each function the
//! context length needs, and run the package's self-test on every one of
//! them (the Neural Engine compiler has miscompiled graphs before; a package
//! that does not reproduce its reference embeddings is never served).
//!
//! Embed: tokenize, plan calls that pack short texts together, fill each
//! call's inputs and run it. Calls run one at a time: the Neural Engine is a
//! single device, so concurrency would only interleave them. A SPLADE call
//! returns the per-block max of every text's shifted logits; the host
//! finishes each text's term weights from its own blocks.

use crate::coreml::{self, Function};
use crate::layout::{self, Call, Family, Segment, SPLADE_BLOCK};
use crate::package::{self, FunctionSpec, Package};
use crate::table::TokenTable;
use async_trait::async_trait;
use inl_core::deadline::BatchDeadline;
use inl_core::device::{model_cache_dir, LoadSlot};
use inl_core::{DenseEngine, EngineInfo, InlError, Modality, Result};
use inl_sparse::modernbert::SpladePooling;
use inl_sparse::{SparseEngine, SparseVector};
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

/// A dense text embedder over a Neural Engine package.
pub struct AneEmbedder {
    ane: Ane,
}

impl AneEmbedder {
    pub fn new(options: AneOptions) -> Result<Self> {
        Ok(Self {
            ane: Ane::new(options, false)?,
        })
    }
}

/// A learned-sparse (SPLADE) encoder over a Neural Engine package. Texts
/// are encoded exactly as given: the caller adds `[Q] ` / `[D] `.
pub struct AneSparseEncoder {
    ane: Ane,
}

impl AneSparseEncoder {
    pub fn new(options: AneOptions) -> Result<Self> {
        Ok(Self {
            ane: Ane::new(options, true)?,
        })
    }
}

/// What both engines share: one package, loaded once, its calls serialized.
struct Ane {
    options: AneOptions,
    /// The engine serves a sparse package (else a dense one).
    sparse: bool,
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
    /// Width of a text's vector: the dense dimension, or the sparse term
    /// space.
    dimension: usize,
    table: TokenTable,
    head: Head,
    /// Functions loaded so far, by length. A function is created on the first
    /// call that needs it: loading all of them (eight for SPLADE) costs about
    /// 2 s of a cold start, and a query only ever touches the smallest.
    functions: Mutex<BTreeMap<usize, Function>>,
    /// What loading and checking any one function needs.
    source: FunctionSource,
}

/// Everything needed to load one function after `Loaded::load` returned.
struct FunctionSource {
    mlmodelc: PathBuf,
    specs: BTreeMap<usize, FunctionSpec>,
    /// The package's self-test, kept to check each function as it loads.
    selftest: SelfTestData,
}

/// How a call's output becomes one vector per text.
enum Head {
    /// Row k of the output is text k's embedding.
    Dense,
    Splade(Splade),
}

/// The host half of SPLADE pooling: the graph returns, per 8-row block, the
/// max of `relu(logit - shift) * keep`; a text's weights are the max over
/// its blocks with the masked vocabulary zeroed, `ln(1 + x)`, max-folded onto
/// canonical terms (all monotone or per term, so they commute with the max).
struct Splade {
    prefix: Vec<u32>,
    masked: Vec<usize>,
    canonical: Option<Vec<u32>>,
    vocab: usize,
}

impl Splade {
    fn load(package: &Package) -> Result<Self> {
        let pooling: SpladePooling =
            serde_json::from_slice(&std::fs::read(package.path(package::SPLADE_POOLING_FILE))?)?;
        let [vocab, _] = package.manifest.token_table.shape;
        pooling.validate(vocab)?;
        if pooling.dimension(vocab) != package.manifest.embedding.dim {
            return Err(InlError::Inference(format!(
                "package {}: the pooling has {} terms, the manifest {}",
                package.manifest.name,
                pooling.dimension(vocab),
                package.manifest.embedding.dim
            )));
        }
        Ok(Self {
            prefix: pooling.prefix_token_ids,
            masked: pooling
                .masked_vocab_ids
                .iter()
                .map(|&id| id as usize)
                .collect(),
            canonical: pooling.canonical_ids,
            vocab,
        })
    }

    /// Term weights (`dimension` wide) of `segment` from a call's
    /// `[S / 8, vocab]` block maxima.
    fn weights(&self, out: &[f32], segment: &Segment, dimension: usize) -> Vec<f32> {
        let vocab = self.vocab;
        let first = segment.offset / SPLADE_BLOCK;
        let end = (segment.offset + segment.len).div_ceil(SPLADE_BLOCK);
        let mut pooled = out[first * vocab..(first + 1) * vocab].to_vec();
        for block in out[(first + 1) * vocab..end * vocab].chunks_exact(vocab) {
            for (max, &value) in pooled.iter_mut().zip(block) {
                if value > *max {
                    *max = value;
                }
            }
        }
        for &id in &self.masked {
            pooled[id] = 0.0;
        }
        let Some(canonical) = &self.canonical else {
            pooled
                .iter_mut()
                .for_each(|weight| *weight = weight.ln_1p());
            return pooled;
        };
        let mut folded = vec![0f32; dimension];
        for (&max, &term) in pooled.iter().zip(canonical) {
            let slot = &mut folded[term as usize];
            // ln_1p is monotone: fold the maxima, take the log once per term.
            if max > *slot {
                *slot = max;
            }
        }
        folded
            .iter_mut()
            .for_each(|weight| *weight = weight.ln_1p());
        folded
    }
}

impl Ane {
    fn new(options: AneOptions, sparse: bool) -> Result<Self> {
        if options.id.is_empty() {
            return Err(InlError::Config(
                "ANE engine id is missing a model name".into(),
            ));
        }
        if options.max_length == 0 {
            return Err(InlError::Config("max_length must be greater than 0".into()));
        }
        Ok(Self {
            options,
            sparse,
            slot: LoadSlot::new(),
            gate: tokio::sync::Mutex::new(()),
        })
    }

    async fn loaded(&self) -> Result<Arc<Loaded>> {
        self.slot
            .get_or_load(|| async {
                let options = self.options.clone();
                let sparse = self.sparse;
                tokio::task::spawn_blocking(move || Loaded::load(&options, sparse).map(Arc::new))
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

    async fn info(&self) -> Result<EngineInfo> {
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

    /// One vector per text, in order.
    async fn vectors(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
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
        let calls = layout::plan(loaded.family, &loaded.lengths, &counts);
        tracing::debug!(
            functions = ?layout::used_lengths(&calls),
            of = loaded.lengths.len(),
            "ANE request uses these functions; the others stay unloaded"
        );
        for call in calls {
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
    fn load(options: &AneOptions, sparse: bool) -> Result<Self> {
        if !cfg!(target_arch = "aarch64") {
            return Err(InlError::Config(
                "the Apple Neural Engine needs an Apple silicon Mac".into(),
            ));
        }
        let cache = model_cache_dir(options.cache_dir.as_deref());
        let package = package::resolve(&options.id, cache.clone())?;
        let family = package.family;
        if family.is_sparse() != sparse {
            let (is, wanted) = if sparse {
                ("dense", "sparse")
            } else {
                ("sparse", "dense")
            };
            return Err(InlError::Config(format!(
                "{} is a {is} package, not a {wanted} one",
                options.id
            )));
        }
        let mlmodelc = compiled_model(&package, &cache)?;
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
        // Validate every function's declared inputs now (cheap, no Core ML), but
        // create none: each is loaded by the first call that needs it.
        let mut specs = BTreeMap::new();
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
            specs.insert(length, spec.clone());
        }
        let head = if family.is_sparse() {
            Head::Splade(Splade::load(&package)?)
        } else {
            Head::Dense
        };
        let loaded = Self {
            family,
            tokenizer,
            specials,
            max_length,
            lengths,
            dimension: package.manifest.embedding.dim,
            table,
            head,
            functions: Mutex::new(BTreeMap::new()),
            source: FunctionSource {
                mlmodelc,
                specs,
                selftest: SelfTestData::read(&package)?,
            },
        };
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
        self.ensure_function(call.length)?;
        self.run_loaded(call, ids)
    }

    /// [`Self::run`] for a function that is already in `functions`.
    fn run_loaded(&self, call: &Call, ids: &[Vec<u32>]) -> Result<Vec<f32>> {
        let mut functions = self.functions.lock();
        let function = functions
            .get_mut(&call.length)
            .expect("the function is loaded");
        let hidden = self.table.hidden();
        let embeds = function.input("inputs_embeds");
        // Alignment gaps between texts are padding rows too.
        embeds.fill(half::f16::ZERO);
        for segment in &call.segments {
            let rows =
                &mut embeds[segment.offset * hidden..(segment.offset + segment.len) * hidden];
            self.table.gather(&ids[segment.item], rows)?;
        }
        for &name in &self.family.input_names(call.length)[1..] {
            match (&self.head, name) {
                (Head::Splade(splade), "keep") => {
                    layout::fill_keep(&call.segments, ids, &splade.prefix, function.input(name))
                }
                _ => layout::fill_input(
                    self.family,
                    call.length,
                    name,
                    &call.segments,
                    function.input(name),
                ),
            }
        }
        let mut out = Vec::new();
        function.predict(&mut out)?;
        Ok(out)
    }

    /// Load function `length` if no call has used it yet, then check it against
    /// the package's reference embeddings before it serves anything. A function
    /// that fails its check is not kept: the next call retries, and fails again.
    fn ensure_function(&self, length: usize) -> Result<()> {
        if self.functions.lock().contains_key(&length) {
            return Ok(());
        }
        let name = self.family.function_name(length);
        let spec = self.source.specs.get(&length).ok_or_else(|| {
            InlError::Inference(format!("function {name} is not part of this load"))
        })?;
        let started = std::time::Instant::now();
        let function = Function::load(&self.source.mlmodelc, &name, spec)?;
        tracing::debug!(function = %name, seconds = started.elapsed().as_secs_f32(), "ANE function loaded");
        self.functions.lock().insert(length, function);
        if let Err(error) = self.self_test_function(length) {
            self.functions.lock().remove(&length);
            return Err(error);
        }
        Ok(())
    }

    fn embed_call(&self, call: &Call, ids: &[Vec<u32>]) -> Result<Vec<(usize, Vec<f32>)>> {
        let out = self.run(call, ids)?;
        Ok(self.rows(call, &out))
    }

    /// Each text's vector from a call's output: a dense function's row k, or
    /// a SPLADE function's blocks of text k turned into term weights.
    fn rows(&self, call: &Call, out: &[f32]) -> Vec<(usize, Vec<f32>)> {
        call.segments
            .iter()
            .enumerate()
            .map(|(slot, segment)| {
                let row = match &self.head {
                    Head::Dense => out[slot * self.dimension..(slot + 1) * self.dimension].to_vec(),
                    Head::Splade(splade) => splade.weights(out, segment, self.dimension),
                };
                (segment.item, row)
            })
            .collect()
    }

    /// Every loaded function reproduces the package's reference embeddings.
    /// Check function `length` against the package's reference embeddings. The
    /// run goes through [`Self::run`] without its lazy load, so the function is
    /// exercised exactly as a request will use it.
    fn self_test_function(&self, length: usize) -> Result<()> {
        let data = &self.source.selftest;
        let function_name = self.family.function_name(length);
        let mut tested = 0;
        let mut worst = f64::INFINITY;
        for test in data
            .spec
            .tests
            .iter()
            .filter(|t| t.function == function_name)
        {
            let sequences: Vec<Vec<u32>> = match (&test.sequences, test.doc_rows) {
                (Some(sequences), _) => sequences.clone(),
                (None, Some(rows)) => {
                    let doc = data.spec.doc_ids.as_ref().ok_or_else(|| {
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
                    offset += ids.len().next_multiple_of(self.family.alignment());
                    segment
                })
                .collect();
            let call = Call { length, segments };
            let out = self.run_loaded(&call, &sequences)?;
            let rows = self.rows(&call, &out);
            let expected = test.expected.read(&data.blob)?;
            let tolerance = data.spec.tolerance.get(&test.kind).ok_or_else(|| {
                InlError::Inference(format!("self-test has no tolerance for {}", test.kind))
            })?;
            let dim = self.dimension;
            for (k, want) in expected.chunks_exact(dim).enumerate() {
                let cos = cosine(&rows[k].1, want);
                worst = worst.min(cos);
                if cos < tolerance.min_cos {
                    return Err(InlError::Inference(format!(
                        "ANE self-test failed: {} text {k} has cosine {cos:.6} to the reference (minimum {})",
                        test.function, tolerance.min_cos
                    )));
                }
            }
            if let (Some(limit), Head::Dense) = (tolerance.zero_rows_max_abs, &self.head) {
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
        if tested == 0 {
            return Err(InlError::Inference(format!(
                "ANE self-test has no case for {function_name}"
            )));
        }
        tracing::info!(function = %function_name, worst_cosine = worst, "ANE self-test passed");
        Ok(())
    }
}

/// The package's self-test, read once at load; each function is checked against
/// it when it first loads.
struct SelfTestData {
    spec: SelfTest,
    blob: Vec<u8>,
}

impl SelfTestData {
    fn read(package: &Package) -> Result<Self> {
        let raw = std::fs::read(package.path("selftest.json"))?;
        let spec: SelfTest = serde_json::from_slice(&raw)?;
        let blob = std::fs::read(package.path(&spec.bin))?;
        Ok(Self { spec, blob })
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
        self.ane.info().await
    }

    async fn token_counts(&self, texts: &[String]) -> Result<Vec<usize>> {
        self.ane.token_counts(texts).await
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        self.ane.vectors(texts).await
    }

    async fn unload(&self) -> bool {
        self.ane.slot.unload().await
    }

    fn is_loaded(&self) -> bool {
        self.ane.slot.is_loaded()
    }
}

#[async_trait]
impl SparseEngine for AneSparseEncoder {
    async fn load(&self) -> Result<EngineInfo> {
        self.ane.info().await
    }

    async fn token_counts(&self, texts: &[String]) -> Result<Vec<usize>> {
        self.ane.token_counts(texts).await
    }

    async fn encode(&self, texts: Vec<String>) -> Result<Vec<SparseVector>> {
        self.ane
            .vectors(texts)
            .await?
            .into_iter()
            .map(|row| {
                // `from_pairs` drops non-finite weights silently; fail instead.
                if row.iter().any(|weight| !weight.is_finite()) {
                    return Err(InlError::Inference(
                        "ANE sparse encoder produced non-finite weights".into(),
                    ));
                }
                Ok(SparseVector::from_pairs(
                    row.into_iter()
                        .enumerate()
                        .map(|(term, weight)| (term as u32, weight)),
                ))
            })
            .collect()
    }

    async fn unload(&self) -> bool {
        self.ane.slot.unload().await
    }

    fn is_loaded(&self) -> bool {
        self.ane.slot.is_loaded()
    }
}
