use anyhow::{Error as AnyErr, Result};
use candle_transformers::models::qwen2::Config as Qwen2Config;
use serde::Deserialize;
use std::path::PathBuf;
use tracing::{debug, warn};
/// Top-level VibeVoice config with nested structures
#[derive(Debug, Clone, Deserialize)]
pub struct VibeVoiceConfig {
    pub model_type: String,
    #[serde(alias = "decoder_config")]
    pub llm_config: LLMConfig,
    pub diffusion_head_config: DiffusionHeadConfig,
    pub acoustic_tokenizer_config: AcousticTokenizerConfig,
    pub semantic_tokenizer_config: SemanticTokenizerConfig,
    #[serde(default = "default_acoustic_vae_dim")]
    pub acoustic_vae_dim: usize,
    #[serde(default)]
    pub semantic_vae_dim: usize,
}
fn default_acoustic_vae_dim() -> usize {
    64
}
impl VibeVoiceConfig {
    pub fn from_file(path: &PathBuf) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)
            .map_err(|e| AnyErr::msg(format!("Failed to parse config: {}", e)))?;
        config.validate()?;
        config.llm_config.validate()?;

        debug!("✓ Loaded {} configuration", config.variant_name());
        debug!(
            "  LLM: hidden={}, vocab={}, heads={}, kv_heads={}, layers={}",
            config.llm_config.hidden_size,
            config.llm_config.vocab_size,
            config.llm_config.num_attention_heads,
            config.llm_config.num_key_value_heads,
            config.llm_config.num_hidden_layers
        );
        debug!(
            "  VAE: acoustic={}, semantic={}",
            config.acoustic_tokenizer_config.vae_dim, config.semantic_tokenizer_config.vae_dim
        );
        debug!(
            "  Diffusion: {} layers, {} inference steps",
            config.diffusion_head_config.head_layers,
            config.diffusion_head_config.ddpm_num_inference_steps
        );

        Ok(config)
    }
    fn validate(&self) -> Result<()> {
        let hidden_size = self.llm_config.hidden_size;
        match hidden_size {
            1536 => {
                if self.llm_config.num_attention_heads != 12 {
                    return Err(AnyErr::msg(format!(
                        "1.5B expects 12 attention heads, got {}",
                        self.llm_config.num_attention_heads
                    )));
                }
                if self.llm_config.num_key_value_heads != 2 {
                    return Err(AnyErr::msg(format!(
                        "1.5B expects 2 KV heads, got {}",
                        self.llm_config.num_key_value_heads
                    )));
                }
                if self.llm_config.vocab_size != 151936 {
                    warn!(
                        "  ⚠️  Warning: 1.5B typically has vocab_size 151936, got {}",
                        self.llm_config.vocab_size
                    );
                }
            }
            3584 => {
                if self.llm_config.num_attention_heads != 28 {
                    return Err(AnyErr::msg(format!(
                        "7B expects 28 attention heads, got {}",
                        self.llm_config.num_attention_heads
                    )));
                }
                if self.llm_config.num_key_value_heads != 4 {
                    return Err(AnyErr::msg(format!(
                        "7B expects 4 KV heads, got {}",
                        self.llm_config.num_key_value_heads
                    )));
                }
                if self.llm_config.vocab_size != 152064 {
                    warn!(
                        "  ⚠️  Warning: 7B typically has vocab_size 152064, got {}",
                        self.llm_config.vocab_size
                    );
                }
            }
            _ => {
                return Err(AnyErr::msg(format!(
                    "Unknown hidden_size: {}. Expected 1536 (1.5B) or 3584 (7B)",
                    hidden_size
                )));
            }
        }
        // Validate acoustic dimensions match
        if self.acoustic_vae_dim != 64 {
            warn!(
                "  ⚠️  Warning: Expected acoustic_vae_dim=64, got {}",
                self.acoustic_vae_dim
            );
        }
        Ok(())
    }
    pub fn variant_name(&self) -> &'static str {
        match self.llm_config.hidden_size {
            1536 => "VibeVoice-1.5B",
            3584 => "VibeVoice-7B",
            _ => "Unknown",
        }
    }
    pub fn to_qwen2_config(&self) -> Qwen2Config {
        let lc = &self.llm_config;
        Qwen2Config {
            vocab_size: lc.vocab_size,
            hidden_size: lc.hidden_size,
            intermediate_size: lc.intermediate_size,
            num_hidden_layers: lc.num_hidden_layers,
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
/// Nested decoder config (Qwen2 LLM parameters)
#[derive(Debug, Clone, Deserialize)]
pub struct LLMConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default = "default_max_window_layers")]
    pub max_window_layers: usize,
    pub hidden_act: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}
fn default_max_window_layers() -> usize {
    28
}
impl LLMConfig {
    pub fn validate(&self) -> Result<()> {
        // Check critical values
        if (self.rms_norm_eps - 1e-6).abs() > 1e-9 {
            warn!(
                "⚠️  RMS norm eps is {}, expected 1e-06 for VibeVoice",
                self.rms_norm_eps
            );
        }

        if self.rope_theta < 100000.0 {
            warn!(
                "⚠️  RoPE theta is {}, expected 1000000.0 for VibeVoice",
                self.rope_theta
            );
        }

        if self.hidden_act != "silu" {
            warn!(
                "⚠️  Hidden activation is '{}', expected 'silu'",
                self.hidden_act
            );
        }

        Ok(())
    }
}

/// Nested diffusion head config matching Python VibeVoiceDiffusionHeadConfig
#[derive(Debug, Clone, Deserialize)]
pub struct DiffusionHeadConfig {
    pub ddpm_batch_mul: usize,
    pub ddpm_beta_schedule: String,
    pub ddpm_num_inference_steps: usize,
    pub ddpm_num_steps: usize,
    pub ddpm_prediction_type: Option<String>,
    pub head_ffn_ratio: f64,
    pub head_layers: usize,
    pub hidden_size: usize,
    pub latent_size: usize,
    pub model_type: String,
    pub prediction_type: String,
    pub rms_norm_eps: f64,
    pub speech_vae_dim: usize,
}
/// Acoustic Tokenizer Configuration matching Python VibeVoiceAcousticTokenizerConfig
#[derive(Debug, Clone, Deserialize)]
pub struct AcousticTokenizerConfig {
    pub channels: usize,
    pub corpus_normalize: f64,
    pub causal: bool,
    pub vae_dim: usize,
    pub fix_std: f64,
    pub std_dist_type: String,
    pub mixer_layer: String,
    pub conv_norm: String,
    pub pad_mode: String,
    pub disable_last_norm: bool,
    pub layernorm: String,
    pub layernorm_eps: f64,
    pub layernorm_elementwise_affine: bool,
    pub conv_bias: bool,
    pub layer_scale_init_value: f64,
    pub weight_init_value: f64,
    pub encoder_n_filters: usize,
    pub encoder_ratios: Vec<usize>,
    pub encoder_depths: String,
    pub decoder_n_filters: usize,
    pub decoder_ratios: Option<Vec<usize>>,
    pub decoder_depths: Option<String>,
}

/// Semantic Tokenizer Configuration matching Python VibeVoiceSemanticTokenizerConfig
#[derive(Debug, Clone, Deserialize)]
pub struct SemanticTokenizerConfig {
    pub channels: usize,
    pub corpus_normalize: f64,
    pub causal: bool,
    pub vae_dim: usize,
    pub fix_std: f64,
    pub std_dist_type: String,
    pub mixer_layer: String,
    pub conv_norm: String,
    pub pad_mode: String,
    pub disable_last_norm: bool,
    pub layernorm: String,
    pub layernorm_eps: f64,
    pub layernorm_elementwise_affine: bool,
    pub conv_bias: bool,
    pub layer_scale_init_value: f64,
    pub weight_init_value: f64,
    pub encoder_n_filters: usize,
    pub encoder_ratios: Vec<usize>,
    pub encoder_depths: String,
}

/// VAE Decoder Configuration matching Python VibeVoiceAcousticTokenizerConfig
#[derive(Debug, Clone, Deserialize)]
pub struct VAEDecoderConfig {
    /// Number of audio channels (1 for mono)
    pub channels: usize,
    /// VAE latent dimension
    pub vae_dim: usize,
    /// Base number of filters
    pub n_filters: usize,
    /// Upsampling ratios for each stage
    pub ratios: Vec<usize>,
    /// Number of blocks in each stage
    pub depths: Vec<usize>,
    /// Whether to use causal convolutions
    pub causal: bool,
    /// Convolution kernel size
    pub kernel_size: usize,
    /// Last layer kernel size
    pub last_kernel_size: usize,
    /// Normalization type for convolutions ('none', 'weight_norm', etc.)
    pub conv_norm: String,
    /// Padding mode ('constant', 'reflect', etc.)
    pub pad_mode: String,
    /// Whether convolutions have bias
    pub conv_bias: bool,
    /// Layer normalization type ('RMSNorm' or 'LN')
    pub layernorm: String,
    /// Layer normalization epsilon
    pub layernorm_eps: f64,
    /// Whether to use elementwise affine in layernorm
    pub layernorm_elementwise_affine: bool,
    /// Mixer layer type ('depthwise_conv', 'conv', etc.)
    pub mixer_layer: String,
    /// Layer scale initialization value
    pub layer_scale_init_value: f64,
    /// Whether to disable last normalization
    pub disable_last_norm: bool,
    /// Trim right ratio for transposed convolutions
    pub trim_right_ratio: f64,
    /// FFN expansion factor
    pub ffn_expansion: usize,
}
impl Default for VAEDecoderConfig {
    fn default() -> Self {
        Self {
            channels: 1,
            vae_dim: 64,
            n_filters: 32,
            ratios: vec![8, 5, 5, 4, 2, 2],
            // Decoder depths are reversed from encoder: [8, 3, 3, 3, 3, 3, 3]
            depths: vec![8, 3, 3, 3, 3, 3, 3],
            causal: true,
            kernel_size: 7,
            last_kernel_size: 7,
            conv_norm: "none".to_string(),
            pad_mode: "constant".to_string(),
            conv_bias: true,
            layernorm: "RMSNorm".to_string(),
            layernorm_eps: 1e-5,
            layernorm_elementwise_affine: true,
            mixer_layer: "depthwise_conv".to_string(),
            layer_scale_init_value: 1e-6,
            disable_last_norm: true,
            trim_right_ratio: 1.0,
            ffn_expansion: 4,
        }
    }
}
impl VAEDecoderConfig {
    /// Create VAEDecoderConfig from AcousticTokenizerConfig
    /// This ensures the decoder uses the actual model config instead of hardcoded defaults
    pub fn from_acoustic_config(ac: &AcousticTokenizerConfig) -> Self {
        // Parse encoder depths (e.g., "3-3-3-3-3-3-8" -> [3, 3, 3, 3, 3, 3, 8])
        let encoder_depths: Vec<usize> = ac
            .encoder_depths
            .split('-')
            .filter_map(|s| s.parse().ok())
            .collect();

        // Decoder depths: use explicit config or reverse encoder depths
        let decoder_depths = ac
            .decoder_depths
            .as_ref()
            .map(|s| s.split('-').filter_map(|x| x.parse().ok()).collect())
            .unwrap_or_else(|| encoder_depths.iter().rev().cloned().collect());

        // Decoder ratios: use explicit config or same as encoder (decoder reverses internally)
        let decoder_ratios = ac
            .decoder_ratios
            .clone()
            .unwrap_or_else(|| ac.encoder_ratios.clone());

        Self {
            channels: ac.channels,
            vae_dim: ac.vae_dim,
            n_filters: ac.decoder_n_filters,
            ratios: decoder_ratios,
            depths: decoder_depths,
            causal: ac.causal,
            kernel_size: 7,      // Python default
            last_kernel_size: 7, // Python default
            conv_norm: ac.conv_norm.clone(),
            pad_mode: ac.pad_mode.clone(),
            conv_bias: ac.conv_bias,
            layernorm: ac.layernorm.clone(),
            layernorm_eps: ac.layernorm_eps,
            layernorm_elementwise_affine: ac.layernorm_elementwise_affine,
            mixer_layer: ac.mixer_layer.clone(),
            layer_scale_init_value: ac.layer_scale_init_value,
            disable_last_norm: ac.disable_last_norm,
            trim_right_ratio: 1.0, // Python default
            ffn_expansion: 4,      // Python default
        }
    }

    /// Calculate the number of channels at each stage
    pub fn stage_channels(&self) -> Vec<usize> {
        let num_stages = self.depths.len();
        (0..num_stages)
            .map(|i| self.n_filters * 2_usize.pow((num_stages - 1 - i) as u32))
            .collect()
    }
    /// Calculate total upsampling factor
    pub fn hop_length(&self) -> usize {
        self.ratios.iter().product()
    }
}
