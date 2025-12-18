//! VibeVoice Realtime (0.5B) Streaming Model with Full CFG Support
//!
//! This module implements the streaming variant of VibeVoice with:
//! - Dual Split LLM architecture for CFG (4 lower + 20 upper layers, both pos/neg paths)
//! - Windowed generation (5 text tokens → 6 speech tokens)
//! - Binary classifier for EOS detection
//! - Voice cache for pre-computed speaker embeddings
//!
//! # Architecture Overview
//!
//! ```text
//! Text tokens (5 per window)
//!        ↓
//! ┌──────────────────────────────────────────┐
//! │         language_model (4 layers)        │
//! │   ┌─────────────┐   ┌─────────────┐      │
//! │   │ pos_lm (KV1)│   │ neg_lm (KV2)│      │
//! │   └─────────────┘   └─────────────┘      │
//! └──────────────────────────────────────────┘
//!        ↓ hidden states (both paths)
//! ┌──────────────────────────────────────────┐
//! │       tts_language_model (20 layers)     │
//! │   ┌─────────────┐   ┌─────────────┐      │
//! │   │pos_tts (KV3)│   │neg_tts (KV4)│      │
//! │   └─────────────┘   └─────────────┘      │
//! └──────────────────────────────────────────┘
//!        ↓
//! pos_condition, neg_condition → CFG Diffusion → Audio
//! ```
//!
//! # CFG Formula
//!
//! ```text
//! output = negative + cfg_scale * (positive - negative)
//! ```

pub mod binary_classifier;
pub mod config;
pub mod generation;
pub mod model;
pub mod split_llm;
pub mod voice_cache;

// Re-export main types
pub use binary_classifier::BinaryClassifier;
pub use config::{RealtimeConfig, TTS_SPEECH_WINDOW_SIZE, TTS_TEXT_WINDOW_SIZE};
pub use generation::{GenerationConfig, GenerationState, WindowedGenerator};
pub use model::VibeVoiceRealtimeModel;
pub use split_llm::{DualKvCaches, DualLmOutput, DualSplitLLM, DualTtsLmOutput};
pub use voice_cache::{CacheEntry, VoiceCache};
