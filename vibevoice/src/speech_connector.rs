use anyhow::{Error as AnyErr, Result};
use candle_core::Tensor;
use candle_nn::{Linear, Module, RmsNorm, VarBuilder, linear};
use tracing::info;

use crate::config::VibeVoiceConfig;

/// Speech Connector - projects VAE latents to LLM embedding space
/// Matches Python SpeechConnector: fc1 -> norm -> fc2
/// Used for both acoustic and semantic embeddings
/// Input: [batch, seq_len, input_dim]
/// Output: [batch, seq_len, hidden_size]
pub struct SpeechConnector {
    pub fc1: Linear,
    pub norm: RmsNorm,
    pub fc2: Linear,
    input_dim: usize,
    #[allow(dead_code)]
    output_dim: usize,
}

impl SpeechConnector {
    pub fn new(vb: VarBuilder, input_dim: usize, output_dim: usize, name: &str) -> Result<Self> {
        info!("\n🔧 Initializing {} Speech Connector:", name);
        info!("   Input dim: {}", input_dim);
        info!("   Output dim: {}", output_dim);

        Ok(Self {
            fc1: linear(input_dim, output_dim, vb.pp("fc1"))?,
            norm: candle_nn::rms_norm(output_dim, 1e-6, vb.pp("norm"))?,
            fc2: linear(output_dim, output_dim, vb.pp("fc2"))?,
            input_dim,
            output_dim,
        })
    }

    /// Create semantic connector (128 -> hidden_size)
    pub fn new_semantic(vb: VarBuilder, config: &VibeVoiceConfig) -> Result<Self> {
        Self::new(
            vb,
            config.semantic_vae_dim,
            config.llm_config.hidden_size,
            "Semantic",
        )
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

        // fc1: [batch, seq_len, input_dim] -> [batch, seq_len, hidden_size]
        let x = self.fc1.forward(x)?;

        // norm: [batch, seq_len, hidden_size] -> [batch, seq_len, hidden_size]
        let x = self.norm.forward(&x)?;

        // fc2: [batch, seq_len, hidden_size] -> [batch, seq_len, hidden_size]
        let x = self.fc2.forward(&x)?;

        Ok(x)
    }
}
