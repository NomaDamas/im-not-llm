//! The one error type every im-not-llm crate returns.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum InlError {
    /// A setting is invalid or unsupported on this build / device.
    #[error("config error: {0}")]
    Config(String),

    /// Loading a model or running inference failed.
    #[error("inference failed: {0}")]
    Inference(String),

    /// A forward pass exceeded its deadline. The thread cannot be cancelled:
    /// every later call in the process fails fast; restart the process.
    #[error("inference timeout: {0}")]
    Timeout(String),

    /// The caller's input cannot be processed (bad image, too many inputs).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, InlError>;

impl InlError {
    /// Stable machine-readable class, surfaced by the server and the bindings.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Config(_) => "config",
            Self::Inference(_) | Self::Io(_) | Self::Json(_) => "inference",
            Self::Timeout(_) => "timeout",
            Self::InvalidInput(_) => "invalid_input",
            Self::Other(_) => "other",
        }
    }
}
