use crate::vae_layers::SConv1d;
use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{Linear, Module, VarBuilder, linear};
use tracing::debug;
pub struct VAEStage {
    pub blocks: Vec<VAEBlock>,
}
impl VAEStage {
    pub fn new(vb: VarBuilder, channels: usize, num_blocks: usize) -> Result<Self> {
        let mut blocks = Vec::new();
        for i in 0..num_blocks {
            blocks.push(VAEBlock::new(vb.pp(i.to_string()), channels)?);
        }
        Ok(Self { blocks })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        Ok(x)
    }

    /// Forward pass with streaming cache support
    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        cache: &mut crate::streaming_cache::StreamingCache,
        stage_id: &str,
    ) -> Result<Tensor> {
        let mut x = x.clone();
        for (i, block) in self.blocks.iter().enumerate() {
            let layer_id = format!("{}.block{}", stage_id, i);
            x = block.forward_with_cache(&x, cache, &layer_id)?;
        }
        Ok(x)
    }
}
pub struct VAEBlock {
    pub norm: CustomRmsNorm,
    pub mixer_conv: SConv1d,
    pub gamma: Tensor,
    pub ffn_norm: CustomRmsNorm,
    pub ffn: FeedForwardNetwork,
    pub ffn_gamma: Tensor,
}

/// Custom RmsNorm that exactly matches PyTorch implementation
#[derive(Clone, Debug)]
pub struct CustomRmsNorm {
    pub weight: Tensor,
    pub eps: f64,
}

impl CustomRmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Result<Self> {
        Ok(Self { weight, eps })
    }
}

impl Module for CustomRmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::error::Error> {
        // Match PyTorch: x * rsqrt(mean(x^2, dim=-1, keepdim=True) + eps) * weight
        let hidden_size = xs.dim(candle_core::D::Minus1)? as f64;

        // x^2
        let x_sq = xs.sqr()?;

        // mean(x^2, dim=-1, keepdim=True)
        let mean_sq = (x_sq.sum_keepdim(candle_core::D::Minus1)? / hidden_size)?;

        // rsqrt(mean + eps) = 1 / sqrt(mean + eps)
        let rms = (mean_sq + self.eps)?.sqrt()?;

        // x / rms
        let normed = xs.broadcast_div(&rms)?;

        // * weight
        normed.broadcast_mul(&self.weight)
    }
}
impl VAEBlock {
    pub fn new(vb: VarBuilder, channels: usize) -> Result<Self> {
        let gamma = vb.get(&[channels], "gamma")?;

        // DEBUG: Print loaded gamma values
        let gamma_vals = gamma.flatten_all()?.to_vec1::<f32>()?;
        debug!(
            "Stage gamma first 10: {:?}",
            &gamma_vals[0..10.min(gamma_vals.len())]
        );

        // Load norm weights directly to use custom implementation
        let norm_weight = vb.get(&[channels], "norm.weight")?;
        let ffn_norm_weight = vb.get(&[channels], "ffn_norm.weight")?;

        Ok(Self {
            norm: CustomRmsNorm::new(norm_weight, 1e-5)?,
            mixer_conv: SConv1d::new(
                vb.pp("mixer.conv"), // SConv1d adds .conv.conv internally
                channels,            // in_channels
                channels,            // out_channels
                7,                   // kernel_size
                1,                   // stride
                1,                   // dilation
                channels,            // groups (depthwise)
                true,                // bias
                true,                // causal
                "constant",          // pad_mode (matches config.pad_mode)
            )?,
            gamma, // Use the loaded gamma
            ffn_norm: CustomRmsNorm::new(ffn_norm_weight, 1e-5)?,
            ffn: FeedForwardNetwork::new(vb.pp("ffn"), channels, channels * 4)?,
            ffn_gamma: vb.get(&[channels], "ffn_gamma")?,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Mixer branch: norm operates on [batch, seq, channels], so we transpose
        let residual = x;
        let x_t = x.transpose(1, 2)?; // [batch, channels, seq] -> [batch, seq, channels]
        let normed = self.norm.forward(&x_t)?;
        let normed = normed.transpose(1, 2)?; // [batch, seq, channels] -> [batch, channels, seq]
        let conv_out = self.mixer_conv.forward(&normed)?;
        let gamma_exp = self.gamma.reshape((1, self.gamma.dims()[0], 1))?; // [1, channels, 1]
        let gated = conv_out.broadcast_mul(&gamma_exp)?;
        let x = (residual + gated)?;

        // FFN branch
        let residual = &x;
        let x_t = x.transpose(1, 2)?; // [batch, channels, seq] -> [batch, seq, channels]
        let normed = self.ffn_norm.forward(&x_t)?;
        let ffn_out = self.ffn.forward(&normed)?; // [batch, seq, channels]
        let ffn_out = ffn_out.transpose(1, 2)?; // [batch, seq, channels] -> [batch, channels, seq]
        let ffn_gamma_exp = self.ffn_gamma.reshape((1, self.ffn_gamma.dims()[0], 1))?; // [1, channels, 1]
        let gated = ffn_out.broadcast_mul(&ffn_gamma_exp)?;
        Ok((residual + gated)?)
    }

    /// Forward pass with streaming cache support for the mixer convolution
    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        cache: &mut crate::streaming_cache::StreamingCache,
        layer_id: &str,
    ) -> Result<Tensor> {
        // Mixer branch with cache (only mixer_conv needs caching)
        let residual = x;
        let x_t = x.transpose(1, 2)?; // [batch, channels, seq] -> [batch, seq, channels]
        let normed = self.norm.forward(&x_t)?;
        let normed = normed.transpose(1, 2)?; // [batch, seq, channels] -> [batch, channels, seq]
        let conv_out = self
            .mixer_conv
            .forward_with_cache(&normed, cache, layer_id)?;
        let gamma_exp = self.gamma.reshape((1, self.gamma.dims()[0], 1))?; // [1, channels, 1]
        let gated = conv_out.broadcast_mul(&gamma_exp)?;
        let x = (residual + gated)?;

        // FFN branch (stateless, no cache needed)
        let residual = &x;
        let x_t = x.transpose(1, 2)?; // [batch, channels, seq] -> [batch, seq, channels]
        let normed = self.ffn_norm.forward(&x_t)?;
        let ffn_out = self.ffn.forward(&normed)?; // [batch, seq, channels]
        let ffn_out = ffn_out.transpose(1, 2)?; // [batch, seq, channels] -> [batch, channels, seq]
        let ffn_gamma_exp = self.ffn_gamma.reshape((1, self.ffn_gamma.dims()[0], 1))?; // [1, channels, 1]
        let gated = ffn_out.broadcast_mul(&ffn_gamma_exp)?;
        Ok((residual + gated)?)
    }
}

/// 1D Causal Convolution with asymmetric padding for streaming
pub struct FeedForwardNetwork {
    pub linear1: Linear,
    pub linear2: Linear,
}
impl FeedForwardNetwork {
    pub fn new(vb: VarBuilder, input_dim: usize, hidden_dim: usize) -> Result<Self> {
        Ok(Self {
            linear1: linear(input_dim, hidden_dim, vb.pp("linear1"))?,
            linear2: linear(hidden_dim, input_dim, vb.pp("linear2"))?,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        // Use gelu_erf (exact ERF-based) to match Python's ACT2FN["gelu"]
        let x = x.gelu_erf()?;
        Ok(self.linear2.forward(&x)?)
    }
}
