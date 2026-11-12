//! Late-interaction (multi-vector) text encoders: one L2-normalized vector
//! per token, scored with MaxSim. The in-process engine runs
//! Sentence-Transformers `MultiVectorEncoder` checkpoints over a Qwen3.5 text
//! tower (`perplexity-ai/pplx-embed-v2-late-0.6b`) on candle.

mod native;
pub mod qwen3_5;

pub use native::{LateEncoder, LateOptions, DEFAULT_MODEL_ID};

use async_trait::async_trait;
use inl_core::{EngineInfo, Result};

/// One input's token vectors, in token order (`[tokens][dimension]`).
pub type MultiVector = Vec<Vec<f32>>;

/// Which side of retrieval an input is: it selects the checkpoint's prompt,
/// the context length, and whether the document skiplist applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    Query,
    Document,
}

impl Task {
    /// The Sentence-Transformers task / prompt name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Document => "document",
        }
    }
}

/// What a server needs from a late-interaction engine.
#[async_trait]
pub trait LateEngine: Send + Sync {
    /// Load the model (once) and describe it. `dimension` is the length of
    /// every token vector; `max_length` the document context.
    async fn load(&self) -> Result<EngineInfo>;

    /// Tokens the model sees per input for `task` (prompt included, after
    /// the task's truncation).
    async fn token_counts(&self, texts: &[String], task: Task) -> Result<Vec<usize>>;

    /// Encode raw `texts` for `task`; the checkpoint's prompt is added here.
    async fn encode(&self, texts: Vec<String>, task: Task) -> Result<Vec<MultiVector>>;

    /// Release the weights; the next call reloads. True when a model was
    /// resident.
    async fn unload(&self) -> bool;

    fn is_loaded(&self) -> bool;
}
