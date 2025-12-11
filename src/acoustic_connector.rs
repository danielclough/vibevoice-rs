use anyhow::{Error as AnyErr, Result};
use candle_core::Tensor;
use candle_nn::{Linear, Module, RmsNorm, VarBuilder, linear};
use tracing::info;

use crate::config::VibeVoiceConfig;

/// Acoustic Connector - projects VAE latents to LLM embedding space
/// Matches Python SpeechConnector: fc1 -> norm -> fc2
/// Input: [batch, seq_len, vae_dim] (64)
/// Output: [batch, seq_len, hidden_size] (1536 or 3584)
pub struct AcousticConnector {
    pub fc1: Linear,
    pub norm: RmsNorm,
    pub fc2: Linear,
    input_dim: usize,
    #[allow(dead_code)]
    output_dim: usize,
}

impl AcousticConnector {
    pub fn new(vb: VarBuilder, config: &VibeVoiceConfig) -> Result<Self> {
        let input_dim = config.acoustic_vae_dim; // 64
        let output_dim = config.llm_config.hidden_size; // 1536 or 3584

        info!("\n🔧 Initializing Acoustic Connector:");
        info!("   Input dim (VAE): {}", input_dim);
        info!("   Output dim (LLM): {}", output_dim);

        Ok(Self {
            fc1: linear(input_dim, output_dim, vb.pp("fc1"))?,
            norm: candle_nn::rms_norm(output_dim, 1e-6, vb.pp("norm"))?,
            fc2: linear(output_dim, output_dim, vb.pp("fc2"))?,
            input_dim,
            output_dim,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_batch, _seq_len, input_dim) = x.dims3()?;

        // Validate input dimension
        if input_dim != self.input_dim {
            return Err(AnyErr::msg(format!(
                "Input dimension mismatch: expected {}, got {}",
                self.input_dim, input_dim
            )));
        }

        // Simple linear projection: VAE dim -> LLM hidden dim
        // fc1: [batch, seq_len, 64] -> [batch, seq_len, hidden_size]
        let x = self.fc1.forward(x)?;

        // norm: [batch, seq_len, hidden_size] -> [batch, seq_len, hidden_size]
        let x = self.norm.forward(&x)?;

        // fc2: [batch, seq_len, hidden_size] -> [batch, seq_len, hidden_size]
        let x = self.fc2.forward(&x)?;

        Ok(x)
    }
}
