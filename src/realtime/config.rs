//! Configuration for VibeVoice Realtime (0.5B) streaming model.
//!
//! The realtime model uses a split LLM architecture:
//! - `language_model`: Lower 4 layers for text processing
//! - `tts_language_model`: Upper 20 layers for TTS generation

use crate::config::{AcousticTokenizerConfig, DiffusionHeadConfig, LLMConfig};
use anyhow::{Result, anyhow};
use candle_transformers::models::qwen2::Config as Qwen2Config;
use serde::Deserialize;
use std::path::Path;
use tracing::debug;

/// Window size for text tokens per generation step.
/// Python: `TTS_TEXT_WINDOW_SIZE = 5`
pub const TTS_TEXT_WINDOW_SIZE: usize = 5;

/// Number of speech tokens generated per text window.
/// Python: `TTS_SPEECH_WINDOW_SIZE = 6`
pub const TTS_SPEECH_WINDOW_SIZE: usize = 6;

/// Audio sample rate for VibeVoice models.
pub const AUDIO_SAMPLE_RATE: u32 = 24000;

/// Default number of upper layers for TTS (tts_language_model).
fn default_tts_backbone_layers() -> usize {
    20
}

/// Default acoustic VAE dimension.
fn default_acoustic_vae_dim() -> usize {
    64
}

/// Configuration for VibeVoice Realtime streaming model.
///
/// This config describes a split LLM architecture where the total layers
/// are divided between `language_model` (lower) and `tts_language_model` (upper).
#[derive(Debug, Clone, Deserialize)]
pub struct RealtimeConfig {
    /// Model type identifier. Expected: "vibevoice_streaming"
    pub model_type: String,

    /// Base LLM configuration (Qwen2 parameters).
    /// Total layers = `llm_config.num_hidden_layers` (typically 24).
    #[serde(alias = "decoder_config")]
    pub llm_config: LLMConfig,

    /// Diffusion head configuration for speech synthesis.
    pub diffusion_head_config: DiffusionHeadConfig,

    /// Acoustic tokenizer (VAE decoder) configuration.
    pub acoustic_tokenizer_config: AcousticTokenizerConfig,

    /// Number of upper layers for TTS generation.
    /// Python: `tts_backbone_num_hidden_layers = 20`
    ///
    /// The lower layers count is computed as:
    /// `llm_config.num_hidden_layers - tts_backbone_num_hidden_layers`
    #[serde(default = "default_tts_backbone_layers")]
    pub tts_backbone_num_hidden_layers: usize,

    /// Acoustic VAE latent dimension.
    #[serde(default = "default_acoustic_vae_dim")]
    pub acoustic_vae_dim: usize,
}

impl RealtimeConfig {
    /// Load configuration from a JSON file.
    ///
    /// Validates that the model type is "vibevoice_streaming".
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;

        // Validate model type
        if config.model_type != "vibevoice_streaming" {
            return Err(anyhow!(
                "Expected model_type 'vibevoice_streaming', got '{}'. \
                 Use the batch model for '{}'.",
                config.model_type,
                config.model_type
            ));
        }

        config.validate()?;

        debug!("Loaded VibeVoice-Realtime configuration");
        debug!(
            "  Split LLM: {} lower + {} upper layers",
            config.lm_num_layers(),
            config.tts_backbone_num_hidden_layers
        );
        debug!(
            "  Hidden size: {}, VAE dim: {}",
            config.llm_config.hidden_size, config.acoustic_vae_dim
        );

        Ok(config)
    }

    /// Validate configuration parameters.
    fn validate(&self) -> Result<()> {
        let total_layers = self.llm_config.num_hidden_layers;
        let upper_layers = self.tts_backbone_num_hidden_layers;

        if upper_layers >= total_layers {
            return Err(anyhow!(
                "tts_backbone_num_hidden_layers ({}) must be less than total layers ({})",
                upper_layers,
                total_layers
            ));
        }

        // For 0.5B model: expect 24 total layers, 1024 hidden size
        if self.llm_config.hidden_size == 1024 {
            if total_layers != 24 {
                tracing::warn!("0.5B model typically has 24 layers, got {}", total_layers);
            }
            if upper_layers != 20 {
                tracing::warn!(
                    "0.5B model typically has 20 TTS layers, got {}",
                    upper_layers
                );
            }
        }

        Ok(())
    }

    /// Number of layers for the lower language_model.
    ///
    /// Computed as `total_layers - tts_backbone_num_hidden_layers`.
    /// For 0.5B model: 24 - 20 = 4 layers.
    #[inline]
    pub fn lm_num_layers(&self) -> usize {
        self.llm_config.num_hidden_layers - self.tts_backbone_num_hidden_layers
    }

    /// Number of layers for the upper tts_language_model.
    #[inline]
    pub fn tts_lm_num_layers(&self) -> usize {
        self.tts_backbone_num_hidden_layers
    }

    /// Hidden size of the LLM.
    #[inline]
    pub fn hidden_size(&self) -> usize {
        self.llm_config.hidden_size
    }

    /// Create Qwen2Config for the lower language_model (4 layers).
    ///
    /// This config is used to initialize the lower LLM that processes text tokens.
    /// Note: The final norm should be disabled (Identity) when using this model.
    pub fn to_lm_qwen2_config(&self) -> Qwen2Config {
        let lc = &self.llm_config;
        Qwen2Config {
            vocab_size: lc.vocab_size,
            hidden_size: lc.hidden_size,
            intermediate_size: lc.intermediate_size,
            num_hidden_layers: self.lm_num_layers(), // 4 layers
            num_attention_heads: lc.num_attention_heads,
            num_key_value_heads: lc.num_key_value_heads,
            max_position_embeddings: lc.max_position_embeddings,
            sliding_window: lc.sliding_window.unwrap_or(lc.max_position_embeddings),
            max_window_layers: lc.max_window_layers,
            tie_word_embeddings: false,
            rope_theta: lc.rope_theta,
            rms_norm_eps: lc.rms_norm_eps,
            use_sliding_window: lc.use_sliding_window,
            hidden_act: candle_nn::Activation::Silu,
        }
    }

    /// Create Qwen2Config for the upper tts_language_model (20 layers).
    ///
    /// This config is used to initialize the upper LLM that generates TTS conditions.
    pub fn to_tts_lm_qwen2_config(&self) -> Qwen2Config {
        let lc = &self.llm_config;
        Qwen2Config {
            vocab_size: lc.vocab_size,
            hidden_size: lc.hidden_size,
            intermediate_size: lc.intermediate_size,
            num_hidden_layers: self.tts_lm_num_layers(), // 20 layers
            num_attention_heads: lc.num_attention_heads,
            num_key_value_heads: lc.num_key_value_heads,
            max_position_embeddings: lc.max_position_embeddings,
            sliding_window: lc.sliding_window.unwrap_or(lc.max_position_embeddings),
            max_window_layers: lc.max_window_layers,
            tie_word_embeddings: false,
            rope_theta: lc.rope_theta,
            rms_norm_eps: lc.rms_norm_eps,
            use_sliding_window: lc.use_sliding_window,
            hidden_act: candle_nn::Activation::Silu,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_split() {
        // Simulate a config with 24 total layers, 20 TTS layers
        let config = RealtimeConfig {
            model_type: "vibevoice_streaming".to_string(),
            llm_config: LLMConfig {
                hidden_size: 1024,
                num_attention_heads: 16,
                num_key_value_heads: 2,
                num_hidden_layers: 24,
                vocab_size: 151936,
                intermediate_size: 2816,
                max_position_embeddings: 32768,
                rope_theta: 1000000.0,
                rms_norm_eps: 1e-6,
                sliding_window: None,
                use_sliding_window: false,
                max_window_layers: 24,
                hidden_act: "silu".to_string(),
                tie_word_embeddings: false,
            },
            diffusion_head_config: DiffusionHeadConfig {
                ddpm_batch_mul: 1,
                ddpm_beta_schedule: "scaled_linear".to_string(),
                ddpm_num_inference_steps: 5,
                ddpm_num_steps: 1000,
                ddpm_prediction_type: Some("v_prediction".to_string()),
                head_ffn_ratio: 4.0,
                head_layers: 6,
                hidden_size: 768,
                latent_size: 64,
                model_type: "DiffusionHead".to_string(),
                prediction_type: "v_prediction".to_string(),
                rms_norm_eps: 1e-6,
                speech_vae_dim: 64,
            },
            acoustic_tokenizer_config: AcousticTokenizerConfig {
                channels: 1,
                corpus_normalize: 0.01,
                causal: true,
                vae_dim: 64,
                fix_std: 1.0,
                std_dist_type: "constant".to_string(),
                mixer_layer: "depthwise_conv".to_string(),
                conv_norm: "none".to_string(),
                pad_mode: "constant".to_string(),
                disable_last_norm: true,
                layernorm: "RMSNorm".to_string(),
                layernorm_eps: 1e-5,
                layernorm_elementwise_affine: true,
                conv_bias: true,
                layer_scale_init_value: 1e-6,
                weight_init_value: 1.0,
                encoder_n_filters: 32,
                encoder_ratios: vec![8, 5, 5, 4, 2, 2],
                encoder_depths: "3-3-3-3-3-3-8".to_string(),
                decoder_n_filters: 32,
                decoder_ratios: None,
                decoder_depths: None,
            },
            tts_backbone_num_hidden_layers: 20,
            acoustic_vae_dim: 64,
        };

        assert_eq!(config.lm_num_layers(), 4);
        assert_eq!(config.tts_lm_num_layers(), 20);

        let lm_config = config.to_lm_qwen2_config();
        assert_eq!(lm_config.num_hidden_layers, 4);

        let tts_config = config.to_tts_lm_qwen2_config();
        assert_eq!(tts_config.num_hidden_layers, 20);
    }

    #[test]
    fn test_window_constants() {
        assert_eq!(TTS_TEXT_WINDOW_SIZE, 5);
        assert_eq!(TTS_SPEECH_WINDOW_SIZE, 6);
    }
}
