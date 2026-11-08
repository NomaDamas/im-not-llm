//! In-process native embedder (Qwen3 and EmbeddingGemma 2 on candle). The
//! models live in [`crate::qwen3`] and [`crate::gemma2`]; this module owns
//! device/dtype selection, batching and the per-batch deadline.

use crate::gemma2::{self, Gemma2Embedder};
use crate::qwen3::{AttentionBackend, AttentionRequest, LoadOptions, Qwen3Embedder};
use async_trait::async_trait;
use candle_core::{DType, Device};
use inl_core::deadline::BatchDeadline;
use inl_core::device::{
    device_kind, device_label, dtype_name, effective_batch_timeout, model_cache_dir,
    resolve_device, DeviceSpec, DtypeSpec, LoadSlot,
};
use inl_core::engine::{DenseEngine, EngineInfo, Modality};
use inl_core::hub::ModelSource;
use inl_core::{InlError, Result};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};
use tokio::sync::OnceCell;

pub(crate) const DEFAULT_MAX_LENGTH: usize = 2048;

/// Default number of inputs in one forward pass.
pub(crate) const DEFAULT_BATCH_SIZE: usize = 64;

/// Default cap on padded tokens (`inputs x longest input`) per forward pass.
pub(crate) const DEFAULT_MAX_BATCH_TOKENS: usize = 4096;

/// Default deadline for one forward pass, in seconds.
pub(crate) const DEFAULT_TIMEOUT_SECONDS: u64 = 60;

#[cfg(test)]
#[test]
fn tokenized_length_detects_qwen3_limit() {
    let tokenizer = test_tokenizer();
    let text = "a ".repeat(DEFAULT_MAX_LENGTH);
    assert!(tokenized_length(&tokenizer, &text) >= DEFAULT_MAX_LENGTH);
}

#[cfg(test)]
#[test]
fn tokenized_length_short_control_stays_below_limit() {
    let tokenizer = test_tokenizer();
    assert!(tokenized_length(&tokenizer, "short control text") < DEFAULT_MAX_LENGTH);
}

/// Upper bound on the attention-score tensor (`batch x heads x seq x seq`)
/// materialized per forward pass. On Metal, a tensor above 4 GiB makes the
/// forward return NaN or silently wrong (finite) vectors, e.g. 17 inputs
/// padded to 2048 tokens with Qwen3-0.6B's 16 heads is ~4.6 GiB. 1 GiB keeps
/// a wide margin and bounds peak memory; smaller batches of long inputs cost
/// no throughput because attention is already compute-bound there.
const ATTENTION_BYTES_BUDGET: usize = 1 << 30;

/// Qwen3 uses byte-level BPE, so a text never tokenizes to more tokens than
/// its UTF-8 byte length. This slack covers template/special tokens.
const SPECIAL_TOKEN_SLACK: usize = 8;

/// The model family behind a native model id, read from its `config.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Qwen3,
    Gemma2,
}

impl Architecture {
    /// `model` is a Hugging Face repo id or a local model directory. Anything
    /// that is not an EmbeddingGemma 2 checkpoint loads as Qwen3, as before.
    fn detect(model: &str, cache_dir: PathBuf) -> Result<Self> {
        let path = ModelSource::resolve(model, cache_dir)?.fetch("config.json")?;
        let config: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        Ok(if config["model_type"] == gemma2::MODEL_TYPE {
            Self::Gemma2
        } else {
            Self::Qwen3
        })
    }

    /// Matryoshka output dimensions EmbeddingGemma 2 supports.
    fn matryoshka_dims(self) -> Option<&'static [usize]> {
        match self {
            Self::Gemma2 => Some(&[128, 256, 512, 768]),
            Self::Qwen3 => None,
        }
    }
}

/// A loaded native embedding model.
enum NativeModel {
    Qwen3(Box<Qwen3Embedder>),
    Gemma2(Box<Gemma2Embedder>),
}

impl NativeModel {
    fn load(architecture: Architecture, options: &LoadOptions<'_>) -> Result<Self> {
        Ok(match architecture {
            Architecture::Qwen3 => Self::Qwen3(Box::new(Qwen3Embedder::from_hf(options)?)),
            Architecture::Gemma2 => Self::Gemma2(Box::new(Gemma2Embedder::from_hf(options)?)),
        })
    }

    /// Unit-length embeddings, one per input.
    fn embed(&self, texts: &[String]) -> candle_core::Result<Vec<Vec<f32>>> {
        match self {
            Self::Qwen3(model) => model.embed(texts),
            Self::Gemma2(model) => model.embed(texts),
        }
    }

    fn attention_heads(&self) -> usize {
        match self {
            Self::Qwen3(model) => model.config().num_attention_heads,
            Self::Gemma2(model) => model.config().num_attention_heads(),
        }
    }

    /// Output vector length, before any Matryoshka truncation.
    fn dimension(&self) -> usize {
        match self {
            Self::Qwen3(model) => model.config().hidden_size,
            Self::Gemma2(model) => model.config().embedding_dim(),
        }
    }

    fn backend(&self) -> AttentionBackend {
        match self {
            Self::Qwen3(model) => model.backend(),
            Self::Gemma2(model) => model.backend(),
        }
    }
}

#[derive(Clone)]
struct LoadedModel {
    model: Arc<Mutex<NativeModel>>,
    /// Serializes forward passes ahead of the deadline; see [`embed_in_batches`].
    gate: Arc<tokio::sync::Mutex<()>>,
    attention_heads: usize,
    /// Output vector length (after [`resolve_output_dim`]).
    dimension: usize,
    /// `cpu`, `metal:N` or `cuda:N`.
    device: String,
    dtype: DType,
    backend: AttentionBackend,
    /// Deadline for one forward pass on the device the model loaded onto.
    batch_timeout: Duration,
    /// Matryoshka truncation length, validated at load; `None` keeps the model
    /// dimension.
    output_dim: Option<usize>,
}

/// What a host needs to record which embedding space the vectors belong to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextIdentity {
    /// The model id as configured: a hub repo id or local directory.
    pub id: String,
    pub dimension: usize,
    /// Resolved precision: `f32`, `f16` or `bf16`.
    pub dtype: &'static str,
    /// Resolved device: `cpu`, `metal` or `cuda`.
    pub device: &'static str,
    pub max_length: usize,
}

/// Settings for a [`TextEmbedder`], with builder-style setters. Build one with
/// [`TextOptions::default`] and `with_*`; only `id` has no usable default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextOptions {
    /// A Hugging Face repo id or a local model directory.
    pub id: String,
    /// Compute device; `Auto` is Metal on macOS, else CUDA if built, else CPU.
    pub device: DeviceSpec,
    /// Weight precision; `None` picks the device default (see [`default_dtype`]).
    pub dtype: Option<DtypeSpec>,
    pub attention: AttentionRequest,
    /// Model context length in tokens.
    pub max_length: usize,
    /// Inputs per forward pass.
    pub batch_size: usize,
    /// Cap on padded tokens (`inputs x longest input`) per forward pass.
    pub max_batch_tokens: usize,
    /// Deadline for one forward pass, used when `per_batch_timeout_seconds` is unset.
    pub timeout_seconds: u64,
    /// Deadline for one forward pass, overriding `timeout_seconds`. At least
    /// [`inl_core::device::CPU_BATCH_TIMEOUT`] on CPU unless set explicitly.
    pub per_batch_timeout_seconds: Option<u64>,
    /// Directory for cached model files; `None` uses the shared cache.
    pub cache_dir: Option<PathBuf>,
    /// Prefix prepended by [`TextEmbedder::embed_query`].
    pub query_prefix: Option<String>,
    /// Prefix prepended by [`TextEmbedder::embed`].
    pub passage_prefix: Option<String>,
    /// Matryoshka output dimension (EmbeddingGemma 2 only): one of the model's
    /// supported sizes. `None` keeps the full dimension.
    pub output_dim: Option<usize>,
}

impl Default for TextOptions {
    fn default() -> Self {
        Self {
            id: String::new(),
            device: DeviceSpec::Auto,
            dtype: None,
            attention: AttentionRequest::Auto,
            max_length: DEFAULT_MAX_LENGTH,
            batch_size: DEFAULT_BATCH_SIZE,
            max_batch_tokens: DEFAULT_MAX_BATCH_TOKENS,
            timeout_seconds: DEFAULT_TIMEOUT_SECONDS,
            per_batch_timeout_seconds: None,
            cache_dir: None,
            query_prefix: None,
            passage_prefix: None,
            output_dim: None,
        }
    }
}

impl TextOptions {
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    pub fn with_device(mut self, device: DeviceSpec) -> Self {
        self.device = device;
        self
    }

    pub fn with_dtype(mut self, dtype: Option<DtypeSpec>) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn with_attention(mut self, attention: AttentionRequest) -> Self {
        self.attention = attention;
        self
    }

    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_max_batch_tokens(mut self, max_batch_tokens: usize) -> Self {
        self.max_batch_tokens = max_batch_tokens;
        self
    }

    pub fn with_timeout_seconds(mut self, timeout_seconds: u64) -> Self {
        self.timeout_seconds = timeout_seconds;
        self
    }

    pub fn with_per_batch_timeout_seconds(mut self, seconds: Option<u64>) -> Self {
        self.per_batch_timeout_seconds = seconds;
        self
    }

    pub fn with_cache_dir(mut self, cache_dir: Option<PathBuf>) -> Self {
        self.cache_dir = cache_dir;
        self
    }

    pub fn with_query_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.query_prefix = Some(prefix.into());
        self
    }

    pub fn with_passage_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.passage_prefix = Some(prefix.into());
        self
    }

    pub fn with_output_dim(mut self, output_dim: Option<usize>) -> Self {
        self.output_dim = output_dim;
        self
    }
}

pub struct TextEmbedder {
    model_id: String,
    device: DeviceSpec,
    /// `None` picks the device's default (see [`default_dtype`]).
    dtype: Option<DtypeSpec>,
    attention: AttentionRequest,
    max_length: usize,
    batch_size: usize,
    max_batch_tokens: usize,
    /// Deadline from the options (`per_batch_timeout_seconds`, else
    /// `timeout_seconds`); see [`effective_batch_timeout`] for the CPU default.
    batch_timeout: Duration,
    batch_timeout_explicit: bool,
    cache_dir: Option<PathBuf>,
    query_prefix: Option<String>,
    passage_prefix: Option<String>,
    /// Requested Matryoshka dimension, validated against the model at load.
    output_dim: Option<usize>,
    inner: LoadSlot<LoadedModel>,
    tokenizer: OnceCell<Tokenizer>,
    deadline: &'static BatchDeadline,
}

/// Process-wide: a timed-out native forward pass strands a thread (and the
/// model it holds) until exit, so every [`TextEmbedder`] in the process must
/// fail fast instead of loading another model.
pub(crate) static NATIVE_DEADLINE: BatchDeadline = BatchDeadline::new();

impl TextEmbedder {
    pub fn new(options: TextOptions) -> Result<Self> {
        if options.id.is_empty() {
            return Err(InlError::Config(
                "text embedder id is missing a model name".to_string(),
            ));
        }
        let batch_timeout = batch_timeout(&options)?;
        Ok(Self {
            model_id: options.id,
            device: options.device,
            dtype: options.dtype,
            attention: options.attention,
            max_length: options.max_length,
            batch_size: options.batch_size,
            max_batch_tokens: options.max_batch_tokens,
            batch_timeout,
            batch_timeout_explicit: options.per_batch_timeout_seconds.is_some(),
            cache_dir: options.cache_dir,
            query_prefix: options.query_prefix,
            passage_prefix: options.passage_prefix,
            output_dim: options.output_dim,
            inner: LoadSlot::new(),
            tokenizer: OnceCell::new(),
            deadline: &NATIVE_DEADLINE,
        })
    }

    async fn tokenizer(&self) -> Result<&Tokenizer> {
        self.tokenizer
            .get_or_try_init(|| async {
                let repo = self.model_id.clone();
                let cache_dir = self.cache_dir.clone();
                tokio::task::spawn_blocking(move || {
                    let source =
                        ModelSource::resolve(&repo, model_cache_dir(cache_dir.as_deref()))?;
                    let path = source.fetch("tokenizer.json")?;
                    let mut tokenizer = Tokenizer::from_file(path).map_err(|error| {
                        InlError::Inference(format!("native tokenizer parse failed: {error}"))
                    })?;
                    let _ = tokenizer.with_padding(Some(PaddingParams {
                        strategy: PaddingStrategy::BatchLongest,
                        direction: tokenizers::PaddingDirection::Left,
                        ..Default::default()
                    }));
                    Ok(tokenizer)
                })
                .await
                .map_err(|error| {
                    InlError::Inference(format!("native tokenizer join failed: {error}"))
                })?
            })
            .await
    }

    /// How many of `texts` (passage prefix applied) reach `max_length` tokens
    /// and are therefore truncated by the model.
    pub async fn count_truncated(&self, texts: &[String]) -> Result<usize> {
        let texts: Vec<String> = texts
            .iter()
            .map(|text| match &self.passage_prefix {
                Some(prefix) => format!("{prefix}{text}"),
                None => text.clone(),
            })
            .collect();
        let candidates: Vec<_> = texts
            .iter()
            // Qwen3's byte-level BPE cannot produce more tokens than UTF-8
            // bytes; this gate avoids tokenizing clearly short chunks while
            // retaining multibyte text as a conservative candidate.
            .filter(|text| text.len() >= self.max_length)
            .collect();
        if candidates.is_empty() {
            return Ok(0);
        }
        let tokenizer = self.tokenizer().await?;
        candidates
            .iter()
            .map(|text| {
                tokenizer
                    .encode(text.as_str(), true)
                    .map(|encoding| usize::from(encoding.len() >= self.max_length))
                    .map_err(|error| {
                        InlError::Inference(format!("native tokenizer encode failed: {error}"))
                    })
            })
            .sum()
    }

    async fn model(&self) -> Result<LoadedModel> {
        self.inner
            .get_or_load(|| async {
                let repo = self.model_id.clone();
                let device_spec = self.device;
                let dtype_spec = self.dtype;
                let attention = self.attention;
                let max_length = self.max_length;
                let cache_dir = self.cache_dir.clone();
                let batch_timeout = self.batch_timeout;
                let batch_timeout_explicit = self.batch_timeout_explicit;
                let requested_output_dim = self.output_dim;
                tokio::task::spawn_blocking(move || {
                    let device = resolve_device(device_spec)?;
                    let cache = model_cache_dir(cache_dir.as_deref());
                    let architecture = Architecture::detect(&repo, cache.clone())?;
                    let dtype = dtype_spec.map_or_else(
                        || default_dtype(&device, architecture),
                        DtypeSpec::to_candle,
                    );
                    let model = NativeModel::load(
                        architecture,
                        &LoadOptions {
                            repo_id: &repo,
                            default_cache_dir: cache,
                            device: &device,
                            dtype,
                            max_length,
                            attention,
                        },
                    )?;
                    let output_dim =
                        resolve_output_dim(architecture, model.dimension(), requested_output_dim)?;
                    let dimension = output_dim.unwrap_or_else(|| model.dimension());
                    let batch_timeout = effective_batch_timeout(
                        batch_timeout,
                        batch_timeout_explicit,
                        device.is_cpu(),
                    );
                    tracing::debug!(
                        device = ?device.location(),
                        ?dtype,
                        attention = ?model.backend(),
                        batch_timeout_seconds = batch_timeout.as_secs(),
                        "native embedder ready"
                    );
                    Ok(LoadedModel {
                        attention_heads: model.attention_heads().max(1),
                        dimension,
                        device: device_label(&device),
                        dtype,
                        backend: model.backend(),
                        batch_timeout,
                        output_dim,
                        model: Arc::new(Mutex::new(model)),
                        gate: Arc::new(tokio::sync::Mutex::new(())),
                    })
                })
                .await
                .map_err(|error| {
                    InlError::Inference(format!("native model load join failed: {error}"))
                })?
            })
            .await
    }

    /// The embedding space this embedder produces. Loads the model if it is
    /// not resident, because the device, dtype and dimension are only known
    /// once it is.
    pub async fn identity(&self) -> Result<TextIdentity> {
        let loaded = self.model().await?;
        Ok(TextIdentity {
            id: self.model_id.clone(),
            dimension: loaded.dimension,
            dtype: dtype_name(loaded.dtype),
            device: device_kind(&loaded.device),
            max_length: self.max_length,
        })
    }

    /// Release the model weights. The next embed reloads them. A forward pass
    /// already running keeps its own handle and finishes first. Returns true
    /// when a model was resident.
    pub async fn unload(&self) -> bool {
        self.inner.unload().await
    }

    /// True while the model weights are resident.
    pub fn is_loaded(&self) -> bool {
        self.inner.is_loaded()
    }

    /// Tokens per text as the model sees it before truncation.
    pub async fn token_counts(&self, texts: &[String]) -> Result<Vec<usize>> {
        self.token_counts_of(texts).await
    }

    async fn token_counts_of(&self, texts: &[String]) -> Result<Vec<usize>> {
        let tokenizer = self.tokenizer().await?;
        texts
            .iter()
            .map(|text| {
                tokenizer
                    .encode(text.as_str(), true)
                    .map(|encoding| encoding.len())
                    .map_err(|error| {
                        InlError::Inference(format!("native tokenizer encode failed: {error}"))
                    })
            })
            .collect()
    }

    /// Embed passages: the passage prefix is applied.
    pub async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let inputs = texts
            .iter()
            .map(|text| match &self.passage_prefix {
                Some(prefix) => format!("{prefix}{text}"),
                None => text.clone(),
            })
            .collect();
        self.embed_texts(inputs).await
    }

    /// Embed a search query: the query prefix is applied.
    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let input = match &self.query_prefix {
            Some(prefix) => format!("{prefix}{text}"),
            None => text.to_string(),
        };
        self.embed_texts(vec![input])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| InlError::Inference("empty response".to_string()))
    }

    /// Embed `texts` exactly as given (no prefixes applied).
    async fn embed_texts(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if self.batch_size == 0 {
            return Err(InlError::Inference(
                "batch_size must be greater than 0".to_string(),
            ));
        }
        self.deadline.ensure_healthy("native embedding")?;
        let loaded = self.model().await?;
        let per_row_attention_bytes = loaded.attention_heads * loaded.dtype.size_in_bytes();
        let backend = loaded.backend;
        let batch_timeout = loaded.batch_timeout;
        let output_dim = loaded.output_dim;
        let model = loaded.model;
        let gate = loaded.gate;
        let batches = plan_batches(
            &texts,
            self.batch_size,
            self.max_length,
            per_row_attention_bytes,
            ATTENTION_BYTES_BUDGET,
            self.max_batch_tokens,
            backend,
        );
        embed_in_batches(
            self.deadline,
            &gate,
            batch_timeout,
            texts,
            batches,
            move |chunk: &[String]| {
                let model = model.lock().map_err(|error| {
                    InlError::Inference(format!("native model lock poisoned: {error}"))
                })?;
                let mut vectors = model.embed(chunk).map_err(|error| {
                    InlError::Inference(format!("native embedding failed: {error}"))
                })?;
                if let Some(dim) = output_dim {
                    for vector in &mut vectors {
                        truncate_and_normalize(vector, dim);
                    }
                }
                Ok(vectors)
            },
        )
        .await
    }
}

#[async_trait]
impl DenseEngine for TextEmbedder {
    async fn load(&self) -> Result<EngineInfo> {
        self.deadline.ensure_healthy("native embedding")?;
        let loaded = self.model().await?;
        // The tokenizer guards the context per request; load it up front too.
        self.tokenizer().await?;
        Ok(EngineInfo {
            model_id: self.model_id.clone(),
            dimension: loaded.dimension,
            dtype: dtype_name(loaded.dtype).to_string(),
            device: loaded.device,
            attention: backend_name(loaded.backend).to_string(),
            max_length: self.max_length,
            modalities: vec![Modality::Text],
        })
    }

    async fn token_counts(&self, texts: &[String]) -> Result<Vec<usize>> {
        self.token_counts_of(texts).await
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        self.embed_texts(texts).await
    }

    async fn unload(&self) -> bool {
        self.inner.unload().await
    }

    fn is_loaded(&self) -> bool {
        self.inner.is_loaded()
    }
}

/// One of `eager`, `tiled` or `fused`, the vocabulary [`EngineInfo`] uses.
fn backend_name(backend: AttentionBackend) -> &'static str {
    match backend {
        AttentionBackend::Eager => "eager",
        AttentionBackend::Tiled => "tiled",
        AttentionBackend::MetalSdpa | AttentionBackend::CpuFlash => "fused",
    }
}

/// Run `embed_batch` over each planned batch on the blocking pool, each
/// forward pass under its own `batch_timeout` so a stuck kernel surfaces as
/// [`InlError::Timeout`] instead of hanging the caller.
///
/// `gate` admits one forward pass at a time. Callers queue on it *before*
/// the deadline starts, so a pass only times out on its own runtime, not
/// while another caller (e.g. a concurrent server request) holds the model.
async fn embed_in_batches<F>(
    deadline: &BatchDeadline,
    gate: &tokio::sync::Mutex<()>,
    batch_timeout: Duration,
    texts: Vec<String>,
    batches: Vec<std::ops::Range<usize>>,
    embed_batch: F,
) -> Result<Vec<Vec<f32>>>
where
    F: Fn(&[String]) -> Result<Vec<Vec<f32>>> + Send + Sync + 'static,
{
    let texts = Arc::new(texts);
    let embed_batch = Arc::new(embed_batch);
    let mut all = Vec::with_capacity(texts.len());
    for range in batches {
        let batch_texts = Arc::clone(&texts);
        let embed_batch = Arc::clone(&embed_batch);
        let expected = range.len();
        let _turn = gate.lock().await;
        let vectors = deadline
            .run("native embedding batch", batch_timeout, move || {
                embed_batch(&batch_texts[range])
            })
            .await?;
        if vectors.len() != expected {
            return Err(InlError::Inference(format!(
                "native embedder returned {} embeddings for {expected} inputs",
                vectors.len(),
            )));
        }
        all.extend(vectors);
    }
    Ok(all)
}

/// Split `texts` into consecutive batches of at most `batch_size` inputs whose
/// attention scratch stays within `budget_bytes` and whose padded token count
/// stays within `max_batch_tokens`.
///
/// Inputs are left-padded to the longest member, so a batch costs `len * padded`
/// tokens, where `padded` is bounded by the byte length of the longest text
/// (capped at `max_length`), and
/// `len * per_row_bytes * padded * backend.score_rows(padded)` bytes of
/// attention scores, where `per_row_bytes` is `heads * dtype_size`. The
/// eager backend keeps `padded` score rows alive, the tiled one a fixed tile,
/// and the fused kernels none, so they differ in how much the budget binds.
/// A single input always forms a batch on its own even when it alone exceeds
/// either budget.
fn plan_batches(
    texts: &[String],
    batch_size: usize,
    max_length: usize,
    per_row_bytes: usize,
    budget_bytes: usize,
    max_batch_tokens: usize,
    backend: AttentionBackend,
) -> Vec<std::ops::Range<usize>> {
    let token_bound = |text: &String| (text.len() + SPECIAL_TOKEN_SLACK).min(max_length.max(1));
    let mut batches = Vec::new();
    let mut start = 0;
    let mut padded = 0;
    for (index, text) in texts.iter().enumerate() {
        let candidate_padded = padded.max(token_bound(text));
        let candidate_len = index - start + 1;
        let attention_bytes = candidate_len
            .saturating_mul(per_row_bytes)
            .saturating_mul(candidate_padded)
            .saturating_mul(backend.score_rows(candidate_padded));
        let padded_tokens = candidate_len.saturating_mul(candidate_padded);
        if index > start
            && (candidate_len > batch_size
                || attention_bytes > budget_bytes
                || padded_tokens > max_batch_tokens)
        {
            batches.push(start..index);
            start = index;
            padded = token_bound(text);
        } else {
            padded = candidate_padded;
        }
    }
    if start < texts.len() {
        batches.push(start..texts.len());
    }
    batches
}

/// Deadline for one forward pass: `per_batch_timeout_seconds`, else
/// `timeout_seconds`.
fn batch_timeout(options: &TextOptions) -> Result<Duration> {
    let seconds = options
        .per_batch_timeout_seconds
        .unwrap_or(options.timeout_seconds);
    if seconds == 0 {
        return Err(InlError::Config(
            "timeout_seconds / per_batch_timeout_seconds must be greater than 0 for the native embedder".to_string(),
        ));
    }
    Ok(Duration::from_secs(seconds))
}

/// Validate a requested Matryoshka output dimension against the loaded model.
/// Only EmbeddingGemma 2 supports it, only at its trained sizes, and never
/// wider than what the model actually produces.
fn resolve_output_dim(
    architecture: Architecture,
    model_dim: usize,
    requested: Option<usize>,
) -> Result<Option<usize>> {
    let Some(dim) = requested else {
        return Ok(None);
    };
    let Some(supported) = architecture.matryoshka_dims() else {
        return Err(InlError::Config(
            "output_dim is only supported for EmbeddingGemma 2 models".to_string(),
        ));
    };
    if !supported.contains(&dim) {
        return Err(InlError::Config(format!(
            "output_dim {dim} is not supported by EmbeddingGemma 2: use one of {supported:?}"
        )));
    }
    if dim > model_dim {
        return Err(InlError::Config(format!(
            "output_dim {dim} exceeds the model's {model_dim}-dimensional embeddings"
        )));
    }
    Ok(Some(dim))
}

/// Truncate a unit-length vector to `dim` and re-normalize it to unit length,
/// the Matryoshka operator: the leading dimensions of a normalized embedding
/// approximate a lower-dimensional embedding once re-normalized.
fn truncate_and_normalize(vector: &mut Vec<f32>, dim: usize) {
    vector.truncate(dim);
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector.iter_mut() {
            *value /= norm;
        }
    }
}

#[cfg(test)]
fn tokenized_length(tokenizer: &Tokenizer, text: &str) -> usize {
    tokenizer
        .encode(text, true)
        .expect("Qwen3 tokenizer encodes unit test input")
        .len()
}

#[cfg(test)]
fn test_tokenizer() -> Tokenizer {
    Tokenizer::from_bytes(
        br#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"[UNK]": 0, "a": 1},
                "unk_token": "[UNK]"
            }
        }"#,
    )
    .expect("build test tokenizer")
}

/// Weight/activation precision when the options do not name one: half
/// precision on Metal for Qwen3, where it cuts the model's footprint by a
/// third at no measurable retrieval cost; f32 elsewhere, where half-precision
/// kernels are either slow (CPU, pre-Volta CUDA) or not worth the precision.
/// EmbeddingGemma 2 overflows float16, so it runs in f32 everywhere.
fn default_dtype(device: &Device, architecture: Architecture) -> DType {
    if device.is_metal() && architecture == Architecture::Qwen3 {
        DType::F16
    } else {
        DType::F32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inl_core::attention::AttentionBackend;
    use std::path::{Path, PathBuf};

    // Qwen3-Embedding-0.6B: 16 heads, f32.
    const QWEN3_F32_ROW: usize = 16 * 4;

    fn texts(lengths: &[usize]) -> Vec<String> {
        lengths.iter().map(|&len| "a".repeat(len)).collect()
    }

    fn max_attention_bytes(texts: &[String], batches: &[std::ops::Range<usize>]) -> usize {
        batches
            .iter()
            .map(|range| {
                let padded = texts[range.clone()]
                    .iter()
                    .map(|text| (text.len() + SPECIAL_TOKEN_SLACK).min(DEFAULT_MAX_LENGTH))
                    .max()
                    .unwrap();
                range.len() * QWEN3_F32_ROW * padded * padded
            })
            .max()
            .unwrap()
    }

    #[test]
    fn short_texts_keep_configured_batch_size() {
        let inputs = texts(&[40; 130]);
        let batches = plan_batches(
            &inputs,
            64,
            DEFAULT_MAX_LENGTH,
            QWEN3_F32_ROW,
            ATTENTION_BYTES_BUDGET,
            DEFAULT_MAX_BATCH_TOKENS,
            AttentionBackend::Eager,
        );
        assert_eq!(batches, vec![0..64, 64..128, 128..130]);
    }

    #[test]
    fn max_length_inputs_stay_under_budget_and_below_metal_4gib_limit() {
        // Reproduces the failing shape: 20 chunks that all pad to 2048 tokens
        // (~5.4 GiB of attention scores in one batch before the fix).
        let inputs = texts(&[200_000; 20]);
        let batches = plan_batches(
            &inputs,
            64,
            DEFAULT_MAX_LENGTH,
            QWEN3_F32_ROW,
            ATTENTION_BYTES_BUDGET,
            DEFAULT_MAX_BATCH_TOKENS,
            AttentionBackend::Eager,
        );
        assert!(max_attention_bytes(&inputs, &batches) <= ATTENTION_BYTES_BUDGET);
        assert!(max_attention_bytes(&inputs, &batches) < 1 << 32);
        assert_eq!(batches.iter().map(|range| range.len()).sum::<usize>(), 20);
    }

    #[test]
    fn batches_are_contiguous_and_cover_every_input_in_order() {
        let inputs = texts(&[10, 5000, 10, 10, 3000, 10, 90_000, 10, 10]);
        let batches = plan_batches(
            &inputs,
            4,
            DEFAULT_MAX_LENGTH,
            QWEN3_F32_ROW,
            ATTENTION_BYTES_BUDGET,
            DEFAULT_MAX_BATCH_TOKENS,
            AttentionBackend::Eager,
        );
        let mut next = 0;
        for range in &batches {
            assert_eq!(range.start, next);
            assert!(!range.is_empty() && range.len() <= 4);
            next = range.end;
        }
        assert_eq!(next, inputs.len());
        assert!(max_attention_bytes(&inputs, &batches) <= ATTENTION_BYTES_BUDGET);
    }

    #[test]
    fn single_oversized_input_still_forms_its_own_batch() {
        let inputs = texts(&[100_000, 100_000]);
        let batches = plan_batches(
            &inputs,
            64,
            32_768,
            QWEN3_F32_ROW,
            ATTENTION_BYTES_BUDGET,
            DEFAULT_MAX_BATCH_TOKENS,
            AttentionBackend::Eager,
        );
        assert_eq!(batches, vec![0..1, 1..2]);
    }

    #[test]
    fn empty_input_plans_no_batches() {
        let batches = plan_batches(
            &[],
            64,
            DEFAULT_MAX_LENGTH,
            QWEN3_F32_ROW,
            ATTENTION_BYTES_BUDGET,
            DEFAULT_MAX_BATCH_TOKENS,
            AttentionBackend::Eager,
        );
        assert!(batches.is_empty());
    }

    fn max_padded_tokens(texts: &[String], batches: &[std::ops::Range<usize>]) -> usize {
        batches
            .iter()
            .map(|range| {
                let padded = texts[range.clone()]
                    .iter()
                    .map(|text| (text.len() + SPECIAL_TOKEN_SLACK).min(DEFAULT_MAX_LENGTH))
                    .max()
                    .unwrap();
                range.len() * padded
            })
            .max()
            .unwrap()
    }

    #[test]
    fn mid_length_chunks_are_capped_by_padded_token_budget_not_just_batch_size() {
        // 64 chunks of ~1000 tokens fit the 1 GiB attention budget (64 MB per
        // row of attention x 16 rows) yet are 64k padded tokens in one pass.
        let inputs = texts(&[1_000; 64]);
        let batches = plan_batches(
            &inputs,
            64,
            DEFAULT_MAX_LENGTH,
            QWEN3_F32_ROW,
            ATTENTION_BYTES_BUDGET,
            DEFAULT_MAX_BATCH_TOKENS,
            AttentionBackend::Eager,
        );
        assert!(
            max_padded_tokens(&inputs, &batches) <= DEFAULT_MAX_BATCH_TOKENS,
            "{batches:?}"
        );
        assert!(batches.iter().all(|range| range.len() <= 8), "{batches:?}");
        assert_eq!(batches.iter().map(|range| range.len()).sum::<usize>(), 64);
    }

    #[test]
    fn token_budget_is_checked_against_the_padded_longest_member() {
        // One long text in a run of short ones raises the padded length of
        // every row of its batch.
        let mut lengths = vec![40; 20];
        lengths[10] = 4_000;
        let inputs = texts(&lengths);
        let batches = plan_batches(
            &inputs,
            64,
            DEFAULT_MAX_LENGTH,
            QWEN3_F32_ROW,
            ATTENTION_BYTES_BUDGET,
            DEFAULT_MAX_BATCH_TOKENS,
            AttentionBackend::Eager,
        );
        assert!(max_padded_tokens(&inputs, &batches) <= DEFAULT_MAX_BATCH_TOKENS);
    }

    #[test]
    fn single_input_over_token_budget_still_forms_its_own_batch() {
        let inputs = texts(&[100_000, 100_000]);
        let batches = plan_batches(
            &inputs,
            64,
            32_768,
            QWEN3_F32_ROW,
            usize::MAX,
            4_096,
            AttentionBackend::Eager,
        );
        assert_eq!(batches, vec![0..1, 1..2]);
    }

    #[test]
    fn fused_kernels_are_not_limited_by_the_attention_budget() {
        // Four 2048-token inputs: eager attention scores are 256 MiB each, so a
        // 256 MiB budget splits them one per batch; a fused kernel holds none.
        let inputs = texts(&[200_000; 4]);
        let plan = |backend| {
            plan_batches(
                &inputs,
                64,
                DEFAULT_MAX_LENGTH,
                QWEN3_F32_ROW,
                ATTENTION_BYTES_BUDGET / 4,
                usize::MAX,
                backend,
            )
        };
        assert_eq!(plan(AttentionBackend::Eager).len(), 4);
        assert_eq!(plan(AttentionBackend::MetalSdpa), vec![0..4]);
        assert_eq!(plan(AttentionBackend::CpuFlash), vec![0..4]);
    }

    #[test]
    fn tiled_attention_scratch_binds_far_later_than_eager() {
        let inputs = texts(&[200_000; 16]);
        let plan = |backend| {
            plan_batches(
                &inputs,
                64,
                DEFAULT_MAX_LENGTH,
                QWEN3_F32_ROW,
                ATTENTION_BYTES_BUDGET,
                usize::MAX,
                backend,
            )
        };
        let eager = plan(AttentionBackend::Eager);
        let tiled = plan(AttentionBackend::Tiled);
        assert!(eager.iter().all(|range| range.len() <= 4), "{eager:?}");
        // 256 rows of 2048 keys per tile keep scratch at 2048 x 256 x 64 B = 32 MiB per input.
        assert!(tiled.iter().all(|range| range.len() <= 32), "{tiled:?}");
        assert!(tiled.len() < eager.len());
    }

    #[test]
    fn dtype_and_attention_settings_are_validated() {
        assert_eq!(DtypeSpec::parse(None).unwrap(), None);
        assert_eq!(DtypeSpec::parse(Some("auto")).unwrap(), None);
        assert_eq!(DtypeSpec::parse(Some("f16")).unwrap(), Some(DtypeSpec::F16));
        assert!(DtypeSpec::parse(Some("int8")).is_err());
        assert_eq!(
            AttentionRequest::parse(Some("eager")).unwrap(),
            AttentionRequest::Eager
        );
        assert!(AttentionRequest::parse(Some("flash")).is_err());
    }

    #[test]
    fn default_dtype_is_f32_unless_a_qwen3_runs_on_metal() {
        for architecture in [Architecture::Qwen3, Architecture::Gemma2] {
            assert_eq!(default_dtype(&Device::Cpu, architecture), DType::F32);
        }
        #[cfg(target_os = "macos")]
        if let Ok(metal) = Device::new_metal(0) {
            assert_eq!(default_dtype(&metal, Architecture::Qwen3), DType::F16);
            assert_eq!(default_dtype(&metal, Architecture::Gemma2), DType::F32);
        }
    }

    #[test]
    fn architecture_follows_the_model_type_of_the_config() {
        let dir = std::env::temp_dir().join("inl-arch-detect");
        std::fs::create_dir_all(&dir).unwrap();
        let detect = |config: &str| {
            std::fs::write(dir.join("config.json"), config).unwrap();
            Architecture::detect(dir.to_str().unwrap(), dir.clone()).unwrap()
        };
        assert_eq!(
            detect(r#"{"model_type":"embedding_gemma2"}"#),
            Architecture::Gemma2
        );
        assert_eq!(detect(r#"{"model_type":"qwen3"}"#), Architecture::Qwen3);
        assert_eq!(detect("{}"), Architecture::Qwen3);
    }

    fn fixture_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_embeddinggemma2")
    }

    fn tiny_embedder(output_dim: Option<usize>) -> TextEmbedder {
        TextEmbedder::new(
            TextOptions::default()
                .with_id(fixture_dir().to_str().unwrap())
                .with_device(DeviceSpec::Cpu)
                .with_output_dim(output_dim),
        )
        .expect("tiny embedder builds")
    }

    #[tokio::test]
    async fn without_output_dim_the_full_dimension_is_produced() {
        let embedder = tiny_embedder(None);
        let identity = embedder.identity().await.unwrap();
        assert_eq!(identity.dimension, 24);
        let vector = embedder.embed(&["w0 w1 w2".to_string()]).await.unwrap();
        assert_eq!(vector[0].len(), 24);
    }

    #[tokio::test]
    async fn output_dim_is_validated_against_the_model() {
        // 128 is an allowed EmbeddingGemma 2 size, but the tiny checkpoint only
        // produces 24 dimensions; asking for more must be a config error.
        let error = tiny_embedder(Some(128))
            .identity()
            .await
            .expect_err("128 exceeds the tiny model")
            .to_string();
        assert!(error.contains("output_dim 128"), "{error}");
        assert!(error.contains("exceeds"), "{error}");

        // Not one of EmbeddingGemma 2's trained sizes.
        let error = tiny_embedder(Some(96))
            .identity()
            .await
            .expect_err("96 is not a supported size")
            .to_string();
        assert!(error.contains("output_dim 96"), "{error}");
    }

    #[test]
    fn output_dim_is_refused_for_qwen3_and_validated_generically() {
        let qwen3 = Architecture::Qwen3;
        let error = resolve_output_dim(qwen3, 1024, Some(256))
            .expect_err("Qwen3 has no Matryoshka support")
            .to_string();
        assert!(
            error.contains("only supported for EmbeddingGemma 2"),
            "{error}"
        );

        let gemma2 = Architecture::Gemma2;
        assert_eq!(resolve_output_dim(gemma2, 768, None).unwrap(), None);
        assert_eq!(
            resolve_output_dim(gemma2, 768, Some(256)).unwrap(),
            Some(256)
        );
        assert!(resolve_output_dim(gemma2, 768, Some(100)).is_err());
        assert!(resolve_output_dim(gemma2, 128, Some(256)).is_err());
    }

    #[test]
    fn truncation_renormalizes_to_unit_length() {
        // [3, 4, 12] -> keep [3, 4] -> [0.6, 0.8].
        let mut vector = vec![3.0, 4.0, 12.0];
        truncate_and_normalize(&mut vector, 2);
        assert_eq!(vector.len(), 2);
        assert!((vector[0] - 0.6).abs() < 1e-6, "{vector:?}");
        assert!((vector[1] - 0.8).abs() < 1e-6, "{vector:?}");
    }

    #[tokio::test]
    async fn prefixes_are_applied_by_embed_and_embed_query() {
        let embedder = TextEmbedder::new(
            TextOptions::default()
                .with_id(fixture_dir().to_str().unwrap())
                .with_device(DeviceSpec::Cpu)
                .with_query_prefix("task: search result | query: ")
                .with_passage_prefix("title: none | text: "),
        )
        .unwrap();

        let query = embedder.embed_query("w0 w1").await.unwrap();
        let explicit = embedder
            .embed_texts(vec!["task: search result | query: w0 w1".to_string()])
            .await
            .unwrap();
        assert_eq!(query, explicit[0]);

        let passage = embedder.embed(&["w2 w3".to_string()]).await.unwrap();
        let explicit_passage = embedder
            .embed_texts(vec!["title: none | text: w2 w3".to_string()])
            .await
            .unwrap();
        assert_eq!(passage[0], explicit_passage[0]);
    }
}

#[cfg(test)]
mod deadline_tests {
    use super::*;
    use std::time::Instant;

    fn settings(timeout: u64, per_batch: Option<u64>) -> TextOptions {
        TextOptions {
            id: "s".to_string(),
            timeout_seconds: timeout,
            per_batch_timeout_seconds: per_batch,
            ..Default::default()
        }
    }

    #[test]
    fn batch_timeout_defaults_to_timeout_seconds() {
        assert_eq!(
            batch_timeout(&settings(42, None)).unwrap(),
            Duration::from_secs(42)
        );
    }

    #[test]
    fn per_batch_timeout_overrides_timeout_seconds() {
        assert_eq!(
            batch_timeout(&settings(42, Some(3))).unwrap(),
            Duration::from_secs(3)
        );
    }

    #[test]
    fn zero_timeout_is_rejected() {
        assert!(matches!(
            batch_timeout(&settings(0, None)),
            Err(InlError::Config(_))
        ));
        assert!(matches!(
            batch_timeout(&settings(42, Some(0))),
            Err(InlError::Config(_))
        ));
    }

    #[test]
    fn cpu_defaults_to_a_longer_deadline_unless_one_is_configured() {
        let configured = Duration::from_secs(60);
        // CPU without per_batch_timeout_seconds: 900 s.
        assert_eq!(
            effective_batch_timeout(configured, false, true),
            Duration::from_secs(900)
        );
        // A larger timeout_seconds is not shortened.
        assert_eq!(
            effective_batch_timeout(Duration::from_secs(1200), false, true),
            Duration::from_secs(1200)
        );
        // An explicit per_batch_timeout_seconds wins, even when shorter.
        assert_eq!(
            effective_batch_timeout(Duration::from_secs(30), true, true),
            Duration::from_secs(30)
        );
        // GPUs keep the configured value (60 s by default).
        assert_eq!(
            effective_batch_timeout(configured, false, false),
            Duration::from_secs(60)
        );
        assert_eq!(
            effective_batch_timeout(configured, true, false),
            Duration::from_secs(60)
        );
    }

    #[tokio::test]
    async fn stuck_batch_returns_timeout_within_deadline() {
        let deadline = BatchDeadline::new();
        let texts: Vec<String> = (0..4).map(|i| format!("t{i}")).collect();
        let started = Instant::now();

        let outcome = embed_in_batches(
            &deadline,
            &tokio::sync::Mutex::new(()),
            Duration::from_millis(100),
            texts,
            vec![0..2, 2..4],
            |chunk: &[String]| {
                if chunk[0] == "t2" {
                    std::thread::sleep(Duration::from_millis(800));
                }
                Ok(chunk.iter().map(|_| vec![1.0]).collect())
            },
        )
        .await;

        assert!(matches!(outcome, Err(InlError::Timeout(_))), "{outcome:?}");
        assert!(started.elapsed() < Duration::from_millis(600));
    }

    #[tokio::test]
    async fn healthy_batches_return_vectors_in_input_order() {
        let deadline = BatchDeadline::new();
        let texts: Vec<String> = (0..5).map(|i| i.to_string()).collect();

        let vectors = embed_in_batches(
            &deadline,
            &tokio::sync::Mutex::new(()),
            Duration::from_secs(5),
            texts,
            vec![0..2, 2..5],
            |chunk: &[String]| {
                Ok(chunk
                    .iter()
                    .map(|text| vec![text.parse::<f32>().unwrap()])
                    .collect())
            },
        )
        .await
        .unwrap();

        assert_eq!(
            vectors,
            vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]]
        );
    }

    #[tokio::test]
    #[allow(clippy::single_range_in_vec_init)]
    async fn vector_count_mismatch_is_an_embedding_error() {
        let deadline = BatchDeadline::new();

        let outcome = embed_in_batches(
            &deadline,
            &tokio::sync::Mutex::new(()),
            Duration::from_secs(5),
            vec!["a".into(), "b".into()],
            vec![0..2],
            |_: &[String]| Ok(vec![vec![1.0]]),
        )
        .await;

        assert!(matches!(outcome, Err(InlError::Inference(_))));
    }

    /// A server runs several requests at once against one model. Waiting
    /// for another request's forward pass must not count against this
    /// request's per-pass deadline, or a busy server would declare itself
    /// wedged although every pass finished in time.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[allow(clippy::single_range_in_vec_init)]
    async fn waiting_for_another_callers_pass_does_not_count_against_the_deadline() {
        let deadline = Arc::new(BatchDeadline::new());
        let gate = Arc::new(tokio::sync::Mutex::new(()));
        let model = Arc::new(Mutex::new(()));
        let run = |name: &'static str| {
            let deadline = Arc::clone(&deadline);
            let gate = Arc::clone(&gate);
            let model = Arc::clone(&model);
            tokio::spawn(async move {
                embed_in_batches(
                    &deadline,
                    &gate,
                    Duration::from_millis(400),
                    vec![name.to_string()],
                    vec![0..1],
                    move |chunk: &[String]| {
                        let _model = model.lock().unwrap();
                        std::thread::sleep(Duration::from_millis(250));
                        Ok(chunk.iter().map(|_| vec![1.0]).collect())
                    },
                )
                .await
            })
        };
        let handles: Vec<_> = ["a", "b", "c"].into_iter().map(run).collect();
        for handle in handles {
            let outcome = handle.await.unwrap();
            assert!(outcome.is_ok(), "{outcome:?}");
        }
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn concurrent_callers_share_one_load() {
        let slot: LoadSlot<Arc<u32>> = LoadSlot::new();
        let loads = AtomicUsize::new(0);
        let load = || async {
            loads.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(30)).await;
            Ok(Arc::new(7))
        };

        let (a, b, c) = tokio::join!(
            slot.get_or_load(load),
            slot.get_or_load(load),
            slot.get_or_load(load)
        );

        assert_eq!((*a.unwrap(), *b.unwrap(), *c.unwrap()), (7, 7, 7));
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert!(slot.is_loaded());
    }

    #[tokio::test]
    async fn unload_releases_the_value_and_the_next_call_reloads() {
        let slot: LoadSlot<Arc<u32>> = LoadSlot::new();
        let loads = AtomicUsize::new(0);
        let load = || async {
            loads.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(1))
        };

        let first = slot.get_or_load(load).await.unwrap();
        assert_eq!(Arc::strong_count(&first), 2);

        assert!(slot.unload().await);
        assert!(!slot.is_loaded());
        assert_eq!(Arc::strong_count(&first), 1, "slot must drop its handle");
        assert!(!slot.unload().await, "second unload is a no-op");

        slot.get_or_load(load).await.unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 2);
        assert!(slot.is_loaded());
    }

    #[tokio::test]
    async fn failed_load_is_not_cached() {
        let slot: LoadSlot<u32> = LoadSlot::new();

        let failed = slot
            .get_or_load(|| async { Err(InlError::Inference("no weights".to_string())) })
            .await;
        assert!(matches!(failed, Err(InlError::Inference(_))));
        assert!(!slot.is_loaded());

        assert_eq!(slot.get_or_load(|| async { Ok(5) }).await.unwrap(), 5);
    }

    #[test]
    fn identity_names_match_the_config_vocabulary() {
        assert_eq!(dtype_name(DType::F32), "f32");
        assert_eq!(dtype_name(DType::F16), "f16");
        assert_eq!(dtype_name(DType::BF16), "bf16");
        assert_eq!(device_kind(&device_label(&Device::Cpu)), "cpu");
        assert_eq!(device_kind("metal:0"), "metal");
        assert_eq!(device_kind("cuda:1"), "cuda");
        #[cfg(target_os = "macos")]
        if let Ok(metal) = Device::new_metal(0) {
            assert_eq!(device_kind(&device_label(&metal)), "metal");
        }
    }
}
