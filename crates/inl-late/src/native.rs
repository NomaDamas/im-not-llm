//! In-process late-interaction encoder: a Sentence-Transformers
//! `MultiVectorEncoder` checkpoint (Qwen3.5 tower, `Dense` projection,
//! `MultiVectorMask` skiplist, `Normalize`) on candle. Device/dtype selection,
//! the load slot and the per-batch deadline come from `inl-core`.

use super::qwen3_5::{Config, Qwen35Text};
use super::{LateEngine, MultiVector, Task};
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use inl_core::deadline::BatchDeadline;
use inl_core::device::{
    device_label, dtype_name, effective_batch_timeout, model_cache_dir, resolve_device, DeviceSpec,
    DtypeSpec, LoadSlot,
};
use inl_core::hub::ModelSource;
use inl_core::{EngineInfo, InlError, Modality, Result};
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokenizers::Tokenizer;

/// Default model: perplexity-ai/pplx-embed-v2-late-0.6b.
pub const DEFAULT_MODEL_ID: &str = "perplexity-ai/pplx-embed-v2-late-0.6b";

/// Default cap on padded tokens (`inputs x longest input`) per forward pass:
/// two full-length (4096-token) documents.
pub const DEFAULT_MAX_BATCH_TOKENS: usize = 8192;

/// Settings for an in-process late-interaction encoder. `id` is a Hugging
/// Face repo id or a directory laid out like a snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LateOptions {
    pub id: String,
    /// `auto`, `cpu`, `metal`, or `cuda`; `None` is `auto`.
    pub device: Option<String>,
    /// `auto`, `f32`, `f16`, or `bf16`; `None`/`auto` is f32.
    pub dtype: Option<String>,
    /// Caps the checkpoint's query and document lengths; `None` keeps them.
    pub max_length: Option<usize>,
    /// Inputs per forward pass.
    pub batch_size: usize,
    /// Cap on padded tokens (`inputs x longest input`) per forward pass.
    pub max_batch_tokens: usize,
    /// Deadline for one forward pass unless `per_batch_timeout_seconds` is set.
    pub timeout_seconds: u64,
    /// Deadline in seconds for one forward pass; at least 900 on CPU.
    pub per_batch_timeout_seconds: Option<u64>,
    /// Directory for cached model files (`INL_CACHE_DIR` when unset).
    pub cache_dir: Option<PathBuf>,
}

impl Default for LateOptions {
    fn default() -> Self {
        Self {
            id: DEFAULT_MODEL_ID.to_string(),
            device: None,
            dtype: None,
            max_length: None,
            batch_size: 32,
            max_batch_tokens: DEFAULT_MAX_BATCH_TOKENS,
            timeout_seconds: 60,
            per_batch_timeout_seconds: None,
            cache_dir: None,
        }
    }
}

impl LateOptions {
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        self.device = Some(device.into());
        self
    }

    pub fn with_dtype(mut self, dtype: impl Into<String>) -> Self {
        self.dtype = Some(dtype.into());
        self
    }

    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = Some(max_length);
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

    pub fn with_per_batch_timeout_seconds(mut self, seconds: u64) -> Self {
        self.per_batch_timeout_seconds = Some(seconds);
        self
    }

    pub fn with_cache_dir(mut self, cache_dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(cache_dir.into());
        self
    }
}

/// Process-wide like the other engines: a timed-out forward pass strands its
/// thread (and the model) until exit, so later calls must fail fast.
static LATE_DEADLINE: BatchDeadline = BatchDeadline::new();

/// Per-task settings of the checkpoint.
struct TaskSetup {
    prompt: String,
    /// Context in tokens, prompt included.
    length: usize,
    /// Token ids dropped from the output (the document skiplist).
    skip: Vec<u32>,
}

/// Everything resident once the model is loaded.
struct Loaded {
    model: Qwen35Text,
    /// `1_Dense` as `[in, out]`, f32.
    projection: Tensor,
    bias: Option<Tensor>,
    normalize: bool,
    query: TaskSetup,
    document: TaskSetup,
    tokenizer: Tokenizer,
    pad_id: u32,
    device: String,
    dtype: DType,
    batch_timeout: Duration,
}

impl Loaded {
    fn task(&self, task: Task) -> &TaskSetup {
        match task {
            Task::Query => &self.query,
            Task::Document => &self.document,
        }
    }

    fn dimension(&self) -> usize {
        self.projection.dim(1).unwrap_or(0)
    }
}

/// The in-process late-interaction encoder.
pub struct LateEncoder {
    model_id: String,
    device: DeviceSpec,
    dtype: Option<DtypeSpec>,
    max_length: Option<usize>,
    batch_size: usize,
    max_batch_tokens: usize,
    batch_timeout: Duration,
    batch_timeout_explicit: bool,
    cache_dir: Option<PathBuf>,
    inner: LoadSlot<Arc<Loaded>>,
    gate: Arc<tokio::sync::Mutex<()>>,
    forward: Arc<Mutex<()>>,
}

impl LateEncoder {
    /// Build an encoder from `options`. Nothing is loaded until the first
    /// encode (or [`Self::identity`]).
    pub fn new(options: LateOptions) -> Result<Self> {
        if options.id.trim().is_empty() {
            return Err(InlError::Config(
                "late encoder id is missing a model name or directory".to_string(),
            ));
        }
        let seconds = options
            .per_batch_timeout_seconds
            .unwrap_or(options.timeout_seconds);
        if seconds == 0 {
            return Err(InlError::Config(
                "timeout_seconds / per_batch_timeout_seconds must be greater than 0 for the native encoder"
                    .to_string(),
            ));
        }
        if options.batch_size == 0 {
            return Err(InlError::Config(
                "batch_size must be greater than 0".to_string(),
            ));
        }
        if options.max_length == Some(0) {
            return Err(InlError::Config(
                "max_length must be greater than 0".to_string(),
            ));
        }
        Ok(Self {
            model_id: options.id,
            device: DeviceSpec::parse(options.device.as_deref())?,
            dtype: DtypeSpec::parse(options.dtype.as_deref())?,
            max_length: options.max_length,
            batch_size: options.batch_size,
            max_batch_tokens: options.max_batch_tokens,
            batch_timeout: Duration::from_secs(seconds),
            batch_timeout_explicit: options.per_batch_timeout_seconds.is_some(),
            cache_dir: options.cache_dir,
            inner: LoadSlot::new(),
            gate: Arc::new(tokio::sync::Mutex::new(())),
            forward: Arc::new(Mutex::new(())),
        })
    }

    async fn loaded(&self) -> Result<Arc<Loaded>> {
        self.inner
            .get_or_load(|| async {
                let model_id = self.model_id.clone();
                let device = self.device;
                let dtype = self.dtype;
                let max_length = self.max_length;
                let cache_dir = self.cache_dir.clone();
                let (timeout, explicit) = (self.batch_timeout, self.batch_timeout_explicit);
                tokio::task::spawn_blocking(move || {
                    let device = resolve_device(device)?;
                    // f32 unless asked otherwise: the reference runs f32, and
                    // MaxSim ranks on small per-token differences.
                    let dtype = dtype.map_or(DType::F32, DtypeSpec::to_candle);
                    let source =
                        ModelSource::resolve(&model_id, model_cache_dir(cache_dir.as_deref()))?;
                    let loaded = load_model(&source, &device, dtype, max_length)?;
                    let batch_timeout = effective_batch_timeout(timeout, explicit, device.is_cpu());
                    tracing::debug!(
                        device = ?device.location(),
                        ?dtype,
                        batch_timeout_seconds = batch_timeout.as_secs(),
                        "native late encoder ready"
                    );
                    Ok(Arc::new(Loaded {
                        device: device_label(&device),
                        batch_timeout,
                        ..loaded
                    }))
                })
                .await
                .map_err(|error| {
                    InlError::Inference(format!("native late model load join failed: {error}"))
                })?
            })
            .await
    }

    /// The token-vector space this encoder produces. Loads the model if needed.
    /// `max_length` is the document context.
    pub async fn identity(&self) -> Result<EngineInfo> {
        let loaded = self.loaded().await?;
        Ok(EngineInfo {
            model_id: self.model_id.clone(),
            dimension: loaded.dimension(),
            dtype: dtype_name(loaded.dtype).to_string(),
            device: loaded.device.clone(),
            attention: loaded.model.backend().name().to_string(),
            max_length: loaded.document.length,
            modalities: vec![Modality::Text],
        })
    }

    /// Release the model weights; the next call reloads them.
    pub async fn unload(&self) -> bool {
        self.inner.unload().await
    }

    pub fn is_loaded(&self) -> bool {
        self.inner.is_loaded()
    }

    /// Token ids per input for `task`: prompt prepended, cut at the task's
    /// context.
    async fn tokenize(
        &self,
        loaded: &Arc<Loaded>,
        texts: Vec<String>,
        task: Task,
    ) -> Result<Vec<Vec<u32>>> {
        let loaded = Arc::clone(loaded);
        tokio::task::spawn_blocking(move || {
            let setup = loaded.task(task);
            texts
                .iter()
                .map(|text| {
                    let mut ids = loaded
                        .tokenizer
                        .encode(format!("{}{text}", setup.prompt), true)
                        .map_err(|error| {
                            InlError::Inference(format!(
                                "native late tokenizer encode failed: {error}"
                            ))
                        })?
                        .get_ids()
                        .to_vec();
                    ids.truncate(setup.length);
                    Ok(ids)
                })
                .collect()
        })
        .await
        .map_err(|error| {
            InlError::Inference(format!("native late tokenize join failed: {error}"))
        })?
    }

    /// Encode `texts` for `task`: one vector per kept token, in token order.
    pub async fn encode(&self, texts: Vec<String>, task: Task) -> Result<Vec<MultiVector>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        LATE_DEADLINE.ensure_healthy("native late encoding")?;
        let loaded = self.loaded().await?;
        let tokenized = self.tokenize(&loaded, texts, task).await?;
        let lengths: Vec<usize> = tokenized.iter().map(Vec::len).collect();
        // Longest first so a batch holds similar lengths; results go back to
        // input order below.
        let order = longest_first(&lengths);
        let mut slots: Vec<Option<Vec<u32>>> = tokenized.into_iter().map(Some).collect();
        let sorted: Vec<Vec<u32>> = order
            .iter()
            .map(|&index| slots[index].take().unwrap_or_default())
            .collect();
        let sorted_lengths: Vec<usize> = sorted.iter().map(Vec::len).collect();
        let batches = plan_batches(&sorted_lengths, self.batch_size, self.max_batch_tokens);
        let sorted = Arc::new(sorted);
        let mut encoded = Vec::with_capacity(sorted.len());
        for range in batches {
            let expected = range.len();
            let _turn = self.gate.lock().await;
            let forward = Arc::clone(&self.forward);
            let model = Arc::clone(&loaded);
            let rows = Arc::clone(&sorted);
            let vectors = LATE_DEADLINE
                .run(
                    "native late encoding batch",
                    loaded.batch_timeout,
                    move || {
                        let _guard = forward.lock().map_err(|error| {
                            InlError::Inference(format!("native late model lock poisoned: {error}"))
                        })?;
                        encode_batch(&model, &rows[range], task)
                    },
                )
                .await?;
            if vectors.len() != expected {
                return Err(InlError::Inference(format!(
                    "native late encoder returned {} outputs for {expected} inputs",
                    vectors.len()
                )));
            }
            encoded.extend(vectors);
        }
        let mut all: Vec<Option<MultiVector>> = (0..encoded.len()).map(|_| None).collect();
        for (position, vectors) in encoded.into_iter().enumerate() {
            all[order[position]] = Some(vectors);
        }
        all.into_iter()
            .map(|vectors| {
                vectors.ok_or_else(|| {
                    InlError::Inference("native late encoder lost an input".to_string())
                })
            })
            .collect()
    }
}

fn load_error(what: &str, error: &dyn std::fmt::Display) -> InlError {
    InlError::Inference(format!("native late model {what}: {error}"))
}

#[derive(Deserialize)]
struct ModuleEntry {
    #[serde(default)]
    path: String,
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Deserialize)]
struct DenseConfig {
    in_features: usize,
    out_features: usize,
    #[serde(default)]
    bias: bool,
    #[serde(default)]
    activation_function: Option<String>,
    #[serde(default)]
    use_residual: bool,
}

#[derive(Deserialize)]
struct MaskConfig {
    #[serde(default)]
    skiplist_words: Vec<String>,
    #[serde(default = "document_only")]
    skiplist_tasks: Vec<String>,
    #[serde(default)]
    keep_only_token_ids: Option<Vec<u32>>,
}

fn document_only() -> Vec<String> {
    vec!["document".to_string()]
}

#[derive(Deserialize, Default)]
struct StConfig {
    #[serde(default)]
    query_length: Option<usize>,
    #[serde(default)]
    document_length: Option<usize>,
    #[serde(default)]
    max_seq_length: Option<usize>,
    #[serde(default)]
    query_expansion: Option<serde_json::Value>,
}

#[derive(Deserialize, Default)]
struct Prompts {
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    document: Option<String>,
}

#[derive(Deserialize, Default)]
struct StModelConfig {
    #[serde(default)]
    prompts: Prompts,
}

fn read_json<T: serde::de::DeserializeOwned>(source: &ModelSource, name: &str) -> Result<T> {
    serde_json::from_slice(&std::fs::read(source.fetch(name)?)?)
        .map_err(|error| load_error(&format!("{name} is invalid"), &error))
}

fn module_file(dir: &str, name: &str) -> String {
    if dir.is_empty() {
        name.to_string()
    } else {
        format!("{}/{name}", dir.trim_end_matches('/'))
    }
}

/// Read the checkpoint: the module pipeline from `modules.json`, the
/// Qwen3.5 `config.json`, prompts and lengths, weights and `tokenizer.json`.
fn load_model(
    source: &ModelSource,
    device: &Device,
    dtype: DType,
    max_length: Option<usize>,
) -> Result<Loaded> {
    let modules: Vec<ModuleEntry> = read_json(source, "modules.json")?;
    let mut modules = modules.into_iter();
    match modules.next() {
        Some(first) if first.kind.ends_with("Transformer") => {}
        _ => {
            return Err(InlError::Inference(
                "native late model modules.json must start with a Transformer module".to_string(),
            ))
        }
    }
    let (mut dense, mut mask, mut normalize) = (None, None, false);
    for module in modules {
        let kind = module.kind.rsplit('.').next().unwrap_or_default();
        match kind {
            "Dense" if dense.is_none() => dense = Some(module.path),
            "MultiVectorMask" => {
                mask = Some(read_json::<MaskConfig>(
                    source,
                    &module_file(&module.path, "config.json"),
                )?)
            }
            "Normalize" => normalize = true,
            _ => {
                return Err(InlError::Inference(format!(
                    "native late model module {} is not supported",
                    module.kind
                )))
            }
        }
    }
    let dense_dir = dense.ok_or_else(|| {
        InlError::Inference("native late model modules.json has no Dense projection".to_string())
    })?;

    let config = Config::parse(&std::fs::read(source.fetch("config.json")?)?)?;
    let st: StConfig = read_json(source, "sentence_bert_config.json")?;
    if st
        .query_expansion
        .as_ref()
        .is_some_and(|value| !value.is_null())
    {
        return Err(InlError::Inference(
            "native late model query expansion is not supported".to_string(),
        ));
    }
    let prompts = read_json::<StModelConfig>(source, "config_sentence_transformers.json")?.prompts;
    let fallback = st.max_seq_length.unwrap_or(config.max_position_embeddings);
    let cap = |length: Option<usize>| {
        length
            .unwrap_or(fallback)
            .min(max_length.unwrap_or(usize::MAX))
            .min(config.max_position_embeddings)
    };
    let (query_length, document_length) = (cap(st.query_length), cap(st.document_length));

    let weights = source.weight_files()?;
    // SAFETY: the safetensors files are not modified while the model is loaded.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, device) }
        .map_err(|error| load_error("weight load failed", &error))?;
    let root = ["language_model", "model.language_model", "model", ""]
        .into_iter()
        .find(|prefix| {
            let name = if prefix.is_empty() {
                "embed_tokens.weight".to_string()
            } else {
                format!("{prefix}.embed_tokens.weight")
            };
            vb.contains_tensor(&name)
        })
        .ok_or_else(|| {
            InlError::Inference("native late model weights have no embed_tokens".to_string())
        })?;
    let vb = if root.is_empty() { vb } else { vb.pp(root) };
    let model = Qwen35Text::load(&config, vb, query_length.max(document_length))
        .map_err(|error| load_error("load failed", &error))?;

    let dense_config: DenseConfig = read_json(source, &module_file(&dense_dir, "config.json"))?;
    let identity = dense_config
        .activation_function
        .as_deref()
        .is_none_or(|name| name.ends_with("Identity"));
    if !identity || dense_config.use_residual || dense_config.in_features != config.hidden_size {
        return Err(InlError::Inference(format!(
            "native late model Dense must be a plain {} -> n projection without activation or residual",
            config.hidden_size
        )));
    }
    let dense_weights = source.fetch(&module_file(&dense_dir, "model.safetensors"))?;
    // SAFETY: as above.
    let dense_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[dense_weights], DType::F32, device) }
            .map_err(|error| load_error("Dense weight load failed", &error))?;
    let shape = (dense_config.out_features, dense_config.in_features);
    let projection = dense_vb
        .get(shape, "linear.weight")
        .and_then(|weight| weight.t()?.contiguous())
        .map_err(|error| load_error("Dense weight load failed", &error))?;
    let bias = if dense_config.bias {
        Some(
            dense_vb
                .get(dense_config.out_features, "linear.bias")
                .map_err(|error| load_error("Dense bias load failed", &error))?,
        )
    } else {
        None
    };

    let mut tokenizer = Tokenizer::from_file(source.fetch("tokenizer.json")?)
        .map_err(|error| load_error("tokenizer parse failed", &error))?;
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(None)
        .map_err(|error| load_error("tokenizer setup failed", &error))?;
    let pad_id = tokenizer
        .get_padding()
        .map(|padding| padding.pad_id)
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
        .or_else(|| tokenizer.token_to_id("<pad>"))
        .unwrap_or(0);

    let mask = mask.unwrap_or(MaskConfig {
        skiplist_words: Vec::new(),
        skiplist_tasks: Vec::new(),
        keep_only_token_ids: None,
    });
    if mask.keep_only_token_ids.is_some() {
        return Err(InlError::Inference(
            "native late model keep_only_token_ids is not supported".to_string(),
        ));
    }
    // As the reference: a skiplist word counts only when it is one vocabulary
    // token; anything else would drop real unknown tokens.
    let skip: Vec<u32> = mask
        .skiplist_words
        .iter()
        .filter_map(|word| tokenizer.token_to_id(word))
        .collect();
    let skip_for = |task: Task| {
        if mask.skiplist_tasks.iter().any(|name| name == task.as_str()) {
            skip.clone()
        } else {
            Vec::new()
        }
    };
    Ok(Loaded {
        model,
        projection,
        bias,
        normalize,
        query: TaskSetup {
            prompt: prompts.query.unwrap_or_default(),
            length: query_length,
            skip: skip_for(Task::Query),
        },
        document: TaskSetup {
            prompt: prompts.document.unwrap_or_default(),
            length: document_length,
            skip: skip_for(Task::Document),
        },
        tokenizer,
        pad_id,
        device: String::new(),
        dtype,
        batch_timeout: Duration::ZERO,
    })
}

/// Right-pad `rows`, run the tower, project, normalize, and keep each row's
/// real tokens that are not on the task's skiplist.
fn encode_batch(loaded: &Loaded, rows: &[Vec<u32>], task: Task) -> Result<Vec<MultiVector>> {
    let fail = |error: candle_core::Error| {
        InlError::Inference(format!("native late forward failed: {error}"))
    };
    let seq = rows.iter().map(Vec::len).max().unwrap_or(0).max(1);
    let mut flat = Vec::with_capacity(rows.len() * seq);
    for row in rows {
        flat.extend_from_slice(row);
        flat.extend(std::iter::repeat_n(loaded.pad_id, seq - row.len()));
    }
    let lengths: Vec<usize> = rows.iter().map(Vec::len).collect();
    let device = loaded.model.device();
    let vectors = (|| -> candle_core::Result<Vec<Vec<Vec<f32>>>> {
        let ids = Tensor::from_vec(flat, (rows.len(), seq), device)?;
        let hidden = loaded.model.forward(&ids, &lengths)?.to_dtype(DType::F32)?;
        let mut projected = hidden.broadcast_matmul(&loaded.projection)?;
        if let Some(bias) = &loaded.bias {
            projected = projected.broadcast_add(bias)?;
        }
        if loaded.normalize {
            // `F.normalize`: x / max(||x||, 1e-12).
            let norm = projected
                .sqr()?
                .sum_keepdim(D::Minus1)?
                .sqrt()?
                .clamp(1e-12, f64::MAX)?;
            projected = projected.broadcast_div(&norm)?;
        }
        projected.to_vec3::<f32>()
    })()
    .map_err(fail)?;
    let skip = &loaded.task(task).skip;
    rows.iter()
        .zip(vectors)
        .map(|(row, matrix)| {
            let kept: MultiVector = row
                .iter()
                .zip(matrix)
                .filter(|(id, _)| !skip.contains(id))
                .map(|(_, vector)| vector)
                .collect();
            if kept.iter().flatten().any(|value| !value.is_finite()) {
                return Err(InlError::Inference(
                    "native late encoder produced non-finite values".to_string(),
                ));
            }
            Ok(kept)
        })
        .collect()
}

/// Input indices ordered by token length, longest first; ties keep input order.
fn longest_first(lengths: &[usize]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..lengths.len()).collect();
    order.sort_by_key(|&index| std::cmp::Reverse(lengths[index]));
    order
}

/// Consecutive batches of at most `batch_size` inputs and `max_batch_tokens`
/// padded tokens (`inputs x longest`); a single input always forms a batch.
fn plan_batches(
    lengths: &[usize],
    batch_size: usize,
    max_batch_tokens: usize,
) -> Vec<std::ops::Range<usize>> {
    let mut batches = Vec::new();
    let (mut start, mut padded) = (0, 0);
    for (index, &length) in lengths.iter().enumerate() {
        let candidate_padded = padded.max(length.max(1));
        let candidate_len = index - start + 1;
        if index > start
            && (candidate_len > batch_size
                || candidate_len.saturating_mul(candidate_padded) > max_batch_tokens)
        {
            batches.push(start..index);
            start = index;
            padded = length.max(1);
        } else {
            padded = candidate_padded;
        }
    }
    if start < lengths.len() {
        batches.push(start..lengths.len());
    }
    batches
}

#[async_trait]
impl LateEngine for LateEncoder {
    async fn load(&self) -> Result<EngineInfo> {
        self.identity().await
    }

    async fn token_counts(&self, texts: &[String], task: Task) -> Result<Vec<usize>> {
        let loaded = self.loaded().await?;
        Ok(self
            .tokenize(&loaded, texts.to_vec(), task)
            .await?
            .iter()
            .map(Vec::len)
            .collect())
    }

    async fn encode(&self, texts: Vec<String>, task: Task) -> Result<Vec<MultiVector>> {
        LateEncoder::encode(self, texts, task).await
    }

    async fn unload(&self) -> bool {
        self.inner.unload().await
    }

    fn is_loaded(&self) -> bool {
        self.inner.is_loaded()
    }
}
