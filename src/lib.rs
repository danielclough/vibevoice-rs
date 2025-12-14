pub mod acoustic_connector;
pub mod config;
pub mod diffusion;
pub mod model;
pub mod processor;
pub mod pytorch_rng; // PyTorch-compatible Box-Muller RNG
pub mod semantic_tokenizer;
pub mod speech_connector;
pub mod streaming_cache;
pub mod utils;
pub mod vae_decoder;
pub mod vae_encoder;
pub mod vae_layers;
pub mod vae_utils;
pub mod voice_mapper;

/// Test helpers for Rust-Python parity validation.
///
/// **Note:** This module is primarily for internal testing and debugging.
/// It provides utilities for loading NumPy checkpoints and comparing outputs.
/// The API is not stable and may change without notice.
#[doc(hidden)]
pub mod test_helpers;
