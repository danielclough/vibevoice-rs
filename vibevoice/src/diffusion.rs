use crate::{config::VibeVoiceConfig, utils::timestep_embedding};
use anyhow::Result;
use candle_core::{D, DType, Tensor};
use candle_nn::{Linear, Module, RmsNorm, VarBuilder, linear_no_bias, rms_norm};
use tracing::{debug, info};

/// SwiGLU activation (used in feed-forward networks)
struct SwiGLU {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}
impl SwiGLU {
    fn new(vb: VarBuilder, hidden_size: usize, intermediate_size: usize) -> Result<Self> {
        Ok(Self {
            w1: linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
            w2: linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            w3: linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?,
        })
    }
}
impl Module for SwiGLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
        let gate = self.w1.forward(x)?;
        let swish = candle_nn::ops::silu(&gate)?;
        let up = self.w2.forward(x)?;
        let gated = (swish * up)?;
        self.w3.forward(&gated)
    }
}
struct DiffusionLayer {
    norm: RmsNorm,
    ada_ln_modulation: Linear,
    ffn: SwiGLU,
}
impl DiffusionLayer {
    fn new(vb: VarBuilder, hidden_size: usize) -> Result<Self> {
        Ok(Self {
            norm: rms_norm(hidden_size, 1e-5, vb.pp("norm"))?,
            ada_ln_modulation: linear_no_bias(
                hidden_size,
                3 * hidden_size,
                vb.pp("adaLN_modulation.1"),
            )?,
            ffn: SwiGLU::new(vb.pp("ffn"), hidden_size, hidden_size * 3)?,
        })
    }
    fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        debug!(
            "DEBUG Layer.forward: x shape = {:?}, c shape = {:?}",
            x.dims(),
            c.dims()
        );
        let c_activated = candle_nn::ops::silu(c)?;
        debug!("DEBUG c_activated shape = {:?}", c_activated.dims());
        let modulation = self.ada_ln_modulation.forward(&c_activated)?;
        debug!("DEBUG modulation shape = {:?}", modulation.dims());
        let chunks: Vec<Tensor> = modulation.chunk(3, D::Minus1)?;
        let shift_ffn = chunks[0].clone(); // [batch, hidden_size]
        let scale_ffn = chunks[1].clone(); // [batch, hidden_size]
        let gate_ffn = chunks[2].clone(); // [batch, hidden_size]
        debug!(
            "DEBUG modulation shapes: shift={:?}, scale={:?}, gate={:?}",
            shift_ffn.dims(),
            scale_ffn.dims(),
            gate_ffn.dims()
        );
        let normed = self.norm.forward(x)?;
        let one = Tensor::ones_like(&scale_ffn)?;
        debug!("DEBUG normed={:?}, one={:?}", normed.dims(), one.dims());
        let modulated = normed.broadcast_mul(&(scale_ffn + one)?)?;
        let modulated = modulated.broadcast_add(&shift_ffn)?;
        let ffn_out = self.ffn.forward(&modulated)?;
        let gated = ffn_out.broadcast_mul(&gate_ffn)?;
        debug!("DEBUG gated shape = {:?}", gated.dims());
        Ok((x + gated)?)
    }
}
/// RMSNorm without learnable weights (elementwise_affine=False in Python)
struct RmsNormNoWeight {
    eps: f64,
}
impl RmsNormNoWeight {
    fn new(eps: f64) -> Self {
        Self { eps }
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let norm = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x_f32.broadcast_div(&(norm + self.eps)?.sqrt()?)?;
        Ok(normed.to_dtype(x.dtype())?)
    }
}
struct FinalLayer {
    norm_final: RmsNormNoWeight,
    ada_ln_modulation: Linear,
    linear: Linear,
}
impl FinalLayer {
    fn new(
        vb: VarBuilder,
        hidden_size: usize,
        output_size: usize,
        cond_size: usize,
    ) -> Result<Self> {
        Ok(Self {
            norm_final: RmsNormNoWeight::new(1e-5),
            ada_ln_modulation: linear_no_bias(
                cond_size,
                2 * hidden_size,
                vb.pp("adaLN_modulation.1"),
            )?,
            linear: linear_no_bias(hidden_size, output_size, vb.pp("linear"))?,
        })
    }
    fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        debug!(
            "DEBUG FinalLayer: x shape = {:?}, c shape = {:?}",
            x.dims(),
            c.dims()
        );
        let c_activated = candle_nn::ops::silu(c)?;
        let modulation = self.ada_ln_modulation.forward(&c_activated)?;
        let chunks: Vec<Tensor> = modulation.chunk(2, D::Minus1)?;
        let shift = chunks[0].clone(); // [batch, hidden_size]
        let scale = chunks[1].clone(); // [batch, hidden_size]
        let x = self.norm_final.forward(x)?;
        let one = Tensor::ones_like(&scale)?;
        let scale_plus_one = (scale + one)?;
        let x = x.broadcast_mul(&scale_plus_one)?;
        let x = x.broadcast_add(&shift)?;
        debug!("DEBUG FinalLayer before linear: x={:?}", x.dims());
        // Debug the linear layer internals
        debug!(
            "DEBUG linear weight shape = {:?}",
            self.linear.weight().dims()
        );
        debug!(
            "DEBUG linear bias = {:?}",
            self.linear.bias().map(|b| b.dims())
        );
        let result = self.linear.forward(&x)?;
        debug!("DEBUG linear output shape = {:?}", result.dims());
        Ok(result)
    }
}

/// Based on the paper and VibeVoice's use of DPMSolverMultistepScheduler
pub struct DPMSolverPP {
    pub num_steps: usize,
    #[allow(dead_code)]
    alpha_min: f32,
    #[allow(dead_code)]
    alpha_max: f32,
}
impl DPMSolverPP {
    pub fn new(num_steps: usize) -> Self {
        Self {
            num_steps,
            alpha_min: 0.0001,
            alpha_max: 0.9999,
        }
    }
}

pub struct DiffusionHead {
    noisy_images_proj: Linear,
    cond_proj: Linear,
    t_embedder_mlp: Vec<Linear>,
    layers: Vec<DiffusionLayer>,
    final_layer: FinalLayer,
    #[allow(dead_code)]
    hidden_size: usize,
    #[allow(dead_code)]
    cond_dim: usize,
}
impl DiffusionHead {
    pub fn new(vb: VarBuilder, config: &VibeVoiceConfig) -> Result<Self> {
        Self::new_with_params(
            vb,
            config.llm_config.hidden_size,
            config.diffusion_head_config.latent_size,
            config.diffusion_head_config.head_layers,
            Some(config.variant_name()),
        )
    }

    /// Create a new DiffusionHead with explicit parameters.
    ///
    /// This is useful for the Realtime model which has a different config structure.
    pub fn new_with_params(
        vb: VarBuilder,
        hidden_size: usize,
        latent_size: usize,
        num_layers: usize,
        variant_name: Option<&str>,
    ) -> Result<Self> {
        let cond_dim = hidden_size;
        info!("\n🔧 Initializing Diffusion Head:");
        if let Some(name) = variant_name {
            info!("   Variant: {}", name);
        }
        info!("   Layers: {}", num_layers);
        info!("   Hidden size: {}", hidden_size);
        info!("   Latent size: {}", latent_size);
        let noisy_images_proj =
            linear_no_bias(latent_size, hidden_size, vb.pp("noisy_images_proj"))?;
        let cond_proj = linear_no_bias(hidden_size, cond_dim, vb.pp("cond_proj"))?;
        let frequency_embedding_size = 256;
        let t_embedder_mlp = vec![
            linear_no_bias(
                frequency_embedding_size,
                hidden_size,
                vb.pp("t_embedder.mlp.0"),
            )?,
            linear_no_bias(hidden_size, hidden_size, vb.pp("t_embedder.mlp.2"))?,
        ];
        let mut layers = Vec::new();
        for i in 0..num_layers {
            layers.push(DiffusionLayer::new(
                vb.pp(format!("layers.{}", i)),
                hidden_size,
            )?);
        }
        let final_layer =
            FinalLayer::new(vb.pp("final_layer"), hidden_size, latent_size, cond_dim)?;
        Ok(Self {
            noisy_images_proj,
            cond_proj,
            t_embedder_mlp,
            layers,
            final_layer,
            hidden_size,
            cond_dim,
        })
    }
    pub fn forward(
        &self,
        noisy_images: &Tensor,
        timesteps: &Tensor,
        condition: &Tensor,
    ) -> Result<Tensor> {
        let mut x = self.noisy_images_proj.forward(noisy_images)?; // [1, 1, 1536]
        let t_freq = timestep_embedding(timesteps, 256)?;
        let mut t_emb = self.t_embedder_mlp[0].forward(&t_freq)?;
        t_emb = candle_nn::ops::silu(&t_emb)?;
        t_emb = self.t_embedder_mlp[1].forward(&t_emb)?; // [1, 1536]
        let condition_proj = self.cond_proj.forward(condition)?; // [1, 1536]
        let c = (condition_proj + t_emb)?; // [1, 1536]
        for layer in &self.layers {
            x = layer.forward(&x, &c)?; // Broadcasting happens in layer
        }
        x = self.final_layer.forward(&x, &c)?;
        Ok(x)
    }
}
