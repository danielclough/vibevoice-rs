//! VibeVoice: High-quality text-to-speech with voice cloning.
//!
//! # Quick Start
//!
//! ```no_run
//! use vibevoice::{VibeVoice, ModelVariant, Device};
//!
//! // Load model
//! let mut vv = VibeVoice::new(ModelVariant::Batch1_5B, Device::auto())?;
//!
//! // Simple TTS
//! let audio = vv.synthesize("Hello, world!", None)?;
//! audio.save_wav("output.wav")?;
//!
//! // Voice cloning
//! let audio = vv.synthesize("Hello!", Some("voice.wav"))?;
//! audio.save_wav("cloned.wav")?;
//! # Ok::<(), vibevoice::VibeVoiceError>(())
//! ```
//!
//! # Builder Pattern
//!
//! For more control, use the builder:
//!
//! ```no_run
//! use vibevoice::{VibeVoice, ModelVariant, Device};
//!
//! let mut vv = VibeVoice::builder()
//!     .variant(ModelVariant::Realtime)
//!     .device(Device::Metal)
//!     .seed(42)
//!     .cfg_scale(1.3)
//!     .diffusion_steps(5)
//!     .build()?;
//! # Ok::<(), vibevoice::VibeVoiceError>(())
//! ```

// Public facade API
mod audio;
mod error;
mod facade;

pub use audio::AudioData;
pub use error::{Result, VibeVoiceError};
pub use facade::{Device, ModelVariant, Progress, ProgressCallback, VibeVoice, VibeVoiceBuilder};

// Internal modules (pub(crate) for use within the crate only)
pub(crate) mod acoustic_connector;
pub(crate) mod config;
pub(crate) mod diffusion;
pub(crate) mod model;
pub(crate) mod processor;
pub(crate) mod pytorch_rng;
pub(crate) mod realtime;
pub(crate) mod semantic_tokenizer;
pub(crate) mod speech_connector;
pub(crate) mod streaming_cache;
pub(crate) mod utils;
pub(crate) mod vae_decoder;
pub(crate) mod vae_encoder;
pub(crate) mod vae_layers;
pub(crate) mod vae_utils;
pub(crate) mod voice_mapper;


// Re-export file logging utility for CLI
pub use utils::init_file_logging;
// Voice path resolution utilities for CLI and server
pub use utils::{VoiceType, detect_voice_type, resolve_voice_path};
// Voice converter is public for CLI use
pub mod voice_converter;

/// Test helpers for Rust-Python parity validation.
///
/// **Note:** This module is primarily for internal testing and debugging.
#[doc(hidden)]
pub mod test_helpers;
