//! Error types for VibeVoice operations.

use thiserror::Error;

/// Result type for VibeVoice operations.
pub type Result<T> = std::result::Result<T, VibeVoiceError>;

/// Errors that can occur during VibeVoice operations.
#[derive(Error, Debug)]
pub enum VibeVoiceError {
    #[error("Device error: {0}")]
    DeviceError(String),

    #[error("Model initialization failed: {0}")]
    InitializationError(String),

    #[error("Failed to download model: {0}")]
    DownloadError(String),

    #[error("Voice processing error: {0}")]
    VoiceError(String),

    #[error("Audio processing error: {0}")]
    AudioError(String),

    #[error("Text processing error: {0}")]
    ProcessingError(String),

    #[error("Generation failed: {0}")]
    GenerationError(String),

    #[error("IO error: {0}")]
    IoError(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Invalid configuration: {0}")]
    ConfigError(String),
}

impl From<candle_core::Error> for VibeVoiceError {
    fn from(e: candle_core::Error) -> Self {
        VibeVoiceError::GenerationError(e.to_string())
    }
}

impl From<std::io::Error> for VibeVoiceError {
    fn from(e: std::io::Error) -> Self {
        VibeVoiceError::IoError(e.to_string())
    }
}

impl From<anyhow::Error> for VibeVoiceError {
    fn from(e: anyhow::Error) -> Self {
        VibeVoiceError::GenerationError(e.to_string())
    }
}
