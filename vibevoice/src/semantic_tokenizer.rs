use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use tracing::info;

use crate::config::VibeVoiceConfig;
use crate::vae_encoder::VAEEncoder;

/// Semantic Tokenizer - encoder-only model for semantic speech features
/// No decoder, no VAE sampling (std_dist_type: "none")
/// Input: Audio waveform [batch, channels, samples]
/// Output: Semantic latents [batch, seq_len, 128]
pub struct SemanticTokenizer {
    encoder: VAEEncoder,
    #[allow(dead_code)]
    vae_dim: usize,
    #[allow(dead_code)]
    device: Device,
}

impl SemanticTokenizer {
    pub fn new(vb: VarBuilder, config: &VibeVoiceConfig, device: Device) -> Result<Self> {
        let cfg = &config.semantic_tokenizer_config;

        info!("\n🔧 Initializing Semantic Tokenizer:");
        info!("   VAE dim: {}", cfg.vae_dim);
        info!("   Encoder filters: {}", cfg.encoder_n_filters);
        info!("   Encoder ratios: {:?}", cfg.encoder_ratios);
        info!("   Encoder depths: {}", cfg.encoder_depths);
        info!("   Causal: {}", cfg.causal);
        info!("   Mixer layer: {}", cfg.mixer_layer);

        // Reuse VAEEncoder with semantic config
        let encoder = VAEEncoder::new_from_vibevoice_config(vb, config, "semantic")?;

        Ok(Self {
            encoder,
            vae_dim: cfg.vae_dim,
            device,
        })
    }

    /// Encode audio to semantic latent representations
    /// Input: audio [batch, channels, samples]
    /// Output: latents [batch, seq_len, vae_dim]
    pub fn encode(&self, audio: &Tensor) -> Result<Tensor> {
        // Pass through encoder
        // Output shape: [batch, vae_dim, seq_len]
        let latents = self.encoder.encode(audio)?;

        // Permute to [batch, seq_len, vae_dim]
        // This matches Python line 1175: latents.permute(0, 2, 1)
        let latents = latents.permute((0, 2, 1))?;

        Ok(latents)
    }

    /// Encode audio with streaming cache support
    /// With cache, this outputs only NEW tokens (incremental)
    /// Input: audio [batch, channels, samples] - new audio chunk
    /// Output: latents [batch, seq_len, vae_dim] - only new tokens
    pub fn encode_with_cache(
        &self,
        audio: &Tensor,
        cache: &mut crate::streaming_cache::StreamingCache,
    ) -> Result<Tensor> {
        // Pass through encoder with cache
        // Output shape: [batch, vae_dim, seq_len] where seq_len is only new tokens
        let latents = self.encoder.encode_with_cache(audio, cache)?;

        // Permute to [batch, seq_len, vae_dim]
        let latents = latents.permute((0, 2, 1))?;

        Ok(latents)
    }
}
