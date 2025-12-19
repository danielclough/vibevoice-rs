/// VAE layer implementations matching Python modular_vibevoice_tokenizer.py
use anyhow::Result;
use candle_core::{D, DType, Module, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Linear, VarBuilder, linear,
};

/// Get extra padding for conv1d to ensure same output length
fn get_extra_padding_for_conv1d(
    input_length: usize,
    kernel_size: usize,
    stride: usize,
    padding_total: usize,
) -> usize {
    let n_frames =
        (input_length as f64 - kernel_size as f64 + padding_total as f64) / stride as f64 + 1.0;
    let ideal_length =
        (n_frames.ceil() - 1.0) * stride as f64 + (kernel_size as f64 - padding_total as f64);
    (ideal_length as usize).saturating_sub(input_length)
}

/// Pad 1D tensor with handling for small inputs in reflect mode
pub fn pad1d(x: &Tensor, paddings: (usize, usize), mode: &str, value: f64) -> Result<Tensor> {
    let (padding_left, padding_right) = paddings;
    let length = x.dim(D::Minus1)?;

    if mode == "reflect" {
        // Match Python's approach: handle small inputs by padding with zeros first
        // Python code (lines 142-147):
        // if length <= max_pad:
        //     extra_pad = max_pad - length + 1
        //     x = F.pad(x, (0, extra_pad))
        // padded = F.pad(x, paddings, mode, value)
        // return padded[..., :end]

        let max_pad = padding_left.max(padding_right);
        let mut x = x.clone();
        let mut extra_pad = 0;

        // If input is too small for reflection padding, pre-pad with zeros
        if length <= max_pad {
            extra_pad = max_pad - length + 1;
            x = x.pad_with_zeros(D::Minus1, 0, extra_pad)?;
            use tracing::debug;
            debug!(
                "pad1d: Input length {} <= max_pad {}, adding zero padding of {}",
                length, max_pad, extra_pad
            );
        }

        // Now do the reflection padding on the pre-padded tensor
        let dims = x.dims();
        let last_dim = dims.len() - 1;
        let padded_length = x.dim(D::Minus1)?;
        let mut parts = Vec::new();

        // Left padding (reflect) - contiguous for CUDA compatibility
        if padding_left > 0 {
            let left_slice = x.narrow(D::Minus1, 1, padding_left)?.contiguous()?;
            let indices: Vec<u32> = (0..padding_left).rev().map(|i| i as u32).collect();
            let indices_tensor = Tensor::from_vec(indices, (padding_left,), x.device())?;
            let left_pad = left_slice
                .index_select(&indices_tensor, last_dim)?
                .contiguous()?;
            parts.push(left_pad);
        }

        // Original data
        parts.push(x.clone());

        // Right padding (reflect) - contiguous for CUDA compatibility
        if padding_right > 0 {
            let start = padded_length - padding_right - 1;
            let right_slice = x.narrow(D::Minus1, start, padding_right)?.contiguous()?;
            let indices: Vec<u32> = (0..padding_right).rev().map(|i| i as u32).collect();
            let indices_tensor = Tensor::from_vec(indices, (padding_right,), x.device())?;
            let right_pad = right_slice
                .index_select(&indices_tensor, last_dim)?
                .contiguous()?;
            parts.push(right_pad);
        }

        // Concatenate all parts and ensure contiguous memory layout for CUDA
        let padded = Tensor::cat(&parts, D::Minus1)?.contiguous()?;

        // Remove the extra zero padding we added (contiguous for CUDA)
        if extra_pad > 0 {
            let end = padded.dim(D::Minus1)? - extra_pad;
            Ok(padded.narrow(D::Minus1, 0, end)?.contiguous()?)
        } else {
            Ok(padded)
        }
    } else if mode == "constant" {
        // Use constant padding with value
        if value == 0.0 {
            Ok(x.pad_with_zeros(D::Minus1, padding_left, padding_right)?)
        } else {
            Ok(x.pad_with_same(D::Minus1, padding_left, padding_right)?)
        }
    } else {
        // Default to zero padding
        Ok(x.pad_with_zeros(D::Minus1, padding_left, padding_right)?)
    }
}

/// Remove padding from tensor
pub fn unpad1d(x: &Tensor, paddings: (usize, usize)) -> Result<Tensor> {
    let (padding_left, padding_right) = paddings;
    let length = x.dim(D::Minus1)?;

    if padding_left + padding_right > length {
        anyhow::bail!(
            "Cannot unpad: padding ({} + {}) > length ({})",
            padding_left,
            padding_right,
            length
        );
    }

    let end = length - padding_right;
    Ok(x.narrow(D::Minus1, padding_left, end - padding_left)?)
}

/// ConvRMSNorm - RMSNorm for convolutional layers with channel transposition
/// Python: class ConvRMSNorm(RMSNorm) in modular_vibevoice_tokenizer.py:77
pub struct ConvRMSNorm {
    pub eps: f64,
    pub weight: Option<Tensor>,
}

impl ConvRMSNorm {
    pub fn new(vb: VarBuilder, dim: usize, eps: f64, elementwise_affine: bool) -> Result<Self> {
        let weight = if elementwise_affine {
            Some(vb.get(dim, "weight")?)
        } else {
            None
        };

        Ok(Self { eps, weight })
    }

    /// Forward pass: transpose (b, c, t) -> (b, t, c), normalize, transpose back
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Transpose: b c t -> b t c (swap dims 1 and 2)
        let x_t = x.transpose(1, 2)?;

        // Convert to f32 for normalization
        let x_f32 = x_t.to_dtype(DType::F32)?;

        // RMS normalization over last dimension (channel dimension after transpose)
        let norm = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let mut normed = x_f32.broadcast_div(&(norm + self.eps)?.sqrt()?)?;

        // Apply weight if present
        if let Some(ref weight) = self.weight {
            let weight_f32 = weight.to_dtype(DType::F32)?;
            normed = normed.broadcast_mul(&weight_f32)?;
        }

        // Convert back to original dtype
        let normed = normed.to_dtype(x.dtype())?;

        // Transpose back: b t c -> b c t (contiguous for CUDA compatibility)
        Ok(normed.transpose(1, 2)?.contiguous()?)
    }
}

/// SConv1d - Conv1d with built-in asymmetric/causal padding
/// Python: class SConv1d(nn.Module) in modular_vibevoice_tokenizer.py:258
pub struct SConv1d {
    pub conv: Conv1d,
    pub norm: Option<Box<dyn Module>>, // Normalization applied after conv
    pub causal: bool,
    pub pad_mode: String,
    pub kernel_size: usize,
    pub dilation: usize,
    pub stride: usize,
    pub padding_total: usize,
    pub context_size: usize, // Number of samples to keep as context for streaming
}

impl SConv1d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
        causal: bool,
        pad_mode: &str,
    ) -> Result<Self> {
        // Calculate padding_total matching Python line 285
        let padding_total = (kernel_size - 1) * dilation - (stride - 1);

        let config = Conv1dConfig {
            padding: 0, // We handle padding manually
            stride,
            dilation,
            groups,
            cudnn_fwd_algo: None,
        };

        // Match Python path structure: SConv1d -> NormConv1d -> Conv1d
        // Creates path "layer.conv.conv.weight" to match Python's "layer.conv.conv.weight"
        let conv_vb = vb.pp("conv").pp("conv");
        let conv = if bias {
            candle_nn::conv1d(in_channels, out_channels, kernel_size, config, conv_vb)?
        } else {
            candle_nn::conv1d_no_bias(in_channels, out_channels, kernel_size, config, conv_vb)?
        };

        // No normalization needed - VibeVoice uses conv_norm="none" (default)
        // Python's NormConv1d with norm="none" uses nn.Identity() which is a no-op
        // This None is functionally equivalent and correct for numerical parity
        // Note: If future models use conv_norm="weight_norm"|"layer_norm"|etc., implement here
        let norm = None;

        // Calculate context size for streaming (Python line 283)
        // context_size = (kernel_size - 1) * dilation - (stride - 1)
        let context_size = (kernel_size - 1) * dilation - (stride - 1);

        Ok(Self {
            conv,
            norm,
            causal,
            pad_mode: pad_mode.to_string(),
            kernel_size,
            dilation,
            stride,
            padding_total,
            context_size,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Get input length
        let input_length = x.dim(D::Minus1)?;

        // Calculate extra padding for stride alignment (Python line 393)
        let extra_padding = get_extra_padding_for_conv1d(
            input_length,
            self.kernel_size,
            self.stride,
            self.padding_total,
        );

        // Apply padding (Python lines 398-408)
        let x_padded = if self.causal {
            // Causal: left padding
            pad1d(x, (self.padding_total, extra_padding), &self.pad_mode, 0.0)?
        } else {
            // Non-causal: symmetric padding
            let padding_right = self.padding_total / 2;
            let padding_left = self.padding_total - padding_right;
            pad1d(
                x,
                (padding_left, padding_right + extra_padding),
                &self.pad_mode,
                0.0,
            )?
        };

        // Apply convolution (Python line 413)
        let mut x = self.conv.forward(&x_padded)?;

        // Apply normalization if present (Python NormConv1d.forward)
        if let Some(ref norm) = self.norm {
            x = norm.forward(&x)?;
        }

        Ok(x)
    }

    /// Forward pass with streaming cache support
    /// Matches Python's _forward_streaming method in modular_vibevoice_tokenizer.py:328
    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        cache: &mut crate::streaming_cache::StreamingCache,
        layer_id: &str,
    ) -> Result<Tensor> {
        use tracing::debug;

        // Only works with causal convolution
        if !self.causal {
            return self.forward(x);
        }

        debug!(
            "[CACHE {}] Input shape: {:?}, context_size: {}",
            layer_id,
            x.dims(),
            self.context_size
        );

        // Get cached context from previous iteration
        let cached_states = if let Some(cached) = cache.get(layer_id) {
            debug!(
                "[CACHE {}] Using existing cache: {:?}",
                layer_id,
                cached.dims()
            );
            cached.clone()
        } else if self.context_size > 0 {
            // First chunk - initialize with zeros for context
            let dims = x.dims();
            debug!(
                "[CACHE {}] Initializing cache with zeros: [{}x{}x{}]",
                layer_id, dims[0], dims[1], self.context_size
            );
            let zeros = Tensor::zeros(
                &[dims[0], dims[1], self.context_size],
                x.dtype(),
                x.device(),
            )?;
            x.device().synchronize()?;
            zeros
        } else {
            // No context needed (kernel_size == stride)
            let dims = x.dims();
            debug!("[CACHE {}] No context needed", layer_id);
            let zeros = Tensor::zeros(&[dims[0], dims[1], 0], x.dtype(), x.device())?;
            x.device().synchronize()?;
            zeros
        };

        // Concatenate cached context with new input
        let input_with_context = if cached_states.dim(D::Minus1)? > 0 {
            let combined = Tensor::cat(&[&cached_states, x], D::Minus1)?.contiguous()?;
            debug!(
                "[CACHE {}] Combined: cache {:?} + input {:?} = {:?}",
                layer_id,
                cached_states.dims(),
                x.dims(),
                combined.dims()
            );
            combined
        } else {
            x.clone()
        };

        // Log cache sum for stem layer to verify zeros
        if layer_id == "upsample_0" {
            let cache_sum: f32 = cached_states
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .map(|v| v.abs())
                .sum();
            let input_sum: f32 = x
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .map(|v| v.abs())
                .sum();
            tracing::info!(
                "🔬 [STEM CACHE] cache_abs_sum={:.6}, input_abs_sum={:.6}, cache_shape={:?}",
                cache_sum,
                input_sum,
                cached_states.dims()
            );
        }

        // Apply convolution WITHOUT extra padding (streaming mode doesn't need stride alignment)
        // Just apply the convolution directly - the cached context provides continuity
        let mut output = self.conv.forward(&input_with_context)?;
        debug!("[CACHE {}] Conv output: {:?}", layer_id, output.dims());

        // Apply normalization if present
        if let Some(ref norm) = self.norm {
            output = norm.forward(&output)?;
        }

        // Update cache with last context_size samples from input_with_context
        // IMPORTANT: Call .contiguous() to ensure we store an independent copy, not a view
        if self.context_size > 0 {
            let total_length = input_with_context.dim(D::Minus1)?;
            let new_cache = if total_length >= self.context_size {
                let start = total_length - self.context_size;
                input_with_context
                    .narrow(D::Minus1, start, self.context_size)?
                    .contiguous()?
            } else {
                input_with_context.contiguous()?
            };
            debug!("[CACHE {}] Storing cache: {:?}", layer_id, new_cache.dims());
            cache.set(layer_id.to_string(), new_cache)?;
        }

        Ok(output)
    }
}

/// SConvTranspose1d - ConvTranspose1d with built-in padding removal
/// Python: class SConvTranspose1d(nn.Module) in modular_vibevoice_tokenizer.py:421
pub struct SConvTranspose1d {
    pub convtr: ConvTranspose1d,
    pub causal: bool,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding_total: usize,
    pub trim_right_ratio: f64,
}

impl SConvTranspose1d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        _bias: bool,
        causal: bool,
        trim_right_ratio: f64,
    ) -> Result<Self> {
        // Calculate padding_total (Python line 443)
        let padding_total = kernel_size - stride;

        let config = ConvTranspose1dConfig {
            padding: 0, // We handle padding manually
            output_padding: 0,
            stride,
            dilation: 1,
            groups: 1,
        };

        // Match Python path structure: SConvTranspose1d -> NormConvTranspose1d -> ConvTranspose1d
        // Creates path "layer.convtr.convtr.weight" to match Python's "layer.convtr.convtr.weight"
        let convtr_vb = vb.pp("convtr").pp("convtr");
        let convtr =
            candle_nn::conv_transpose1d(in_channels, out_channels, kernel_size, config, convtr_vb)?;

        Ok(Self {
            convtr,
            causal,
            kernel_size,
            stride,
            padding_total,
            trim_right_ratio,
        })
    }

    /// Forward pass with streaming cache support
    /// Matches Python's _forward_streaming method in modular_vibevoice_tokenizer.py:478
    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        cache: &mut crate::streaming_cache::StreamingCache,
        layer_id: &str,
    ) -> Result<Tensor> {
        use tracing::debug;

        let dims = x.dims();
        let t = dims[2];

        debug!("[CACHE {}] Input shape: {:?}", layer_id, dims);

        // Get cached input from previous iteration
        // CRITICAL: For transposed conv, Python initializes with EMPTY cache (size 0), NOT zeros!
        // This is different from regular SConv1d which uses zeros.
        let cached_input = if let Some(cached) = cache.get(layer_id) {
            debug!(
                "[CACHE {}] Using existing cache: {:?}",
                layer_id,
                cached.dims()
            );
            cached.clone()
        } else {
            // First chunk - initialize with EMPTY cache (matching Python's SConvTranspose1d behavior)
            debug!(
                "[CACHE {}] Initialized empty cache for transposed conv",
                layer_id
            );
            let zeros = Tensor::zeros(&[dims[0], dims[1], 0], x.dtype(), x.device())?;
            x.device().synchronize()?;
            zeros
        };

        // FIXED: Check cache SHAPE, not existence (matching Python line 522)
        // Python: `if cached_input.shape[2] == 0: return full_output`
        // An empty cache (shape [B,C,0]) should be treated as "first chunk"
        let is_first_chunk = cached_input.dim(D::Minus1)? == 0;

        // Concatenate cached input with new input (contiguous for CUDA compatibility)
        let full_input = if cached_input.dim(D::Minus1)? > 0 {
            Tensor::cat(&[&cached_input, x], D::Minus1)?.contiguous()?
        } else {
            x.clone()
        };
        debug!("[CACHE {}] Combined: {:?}", layer_id, full_input.dims());

        // Apply transposed convolution
        let mut full_output = self.convtr.forward(&full_input)?;
        debug!(
            "[CACHE {}] ConvTr output: {:?}",
            layer_id,
            full_output.dims()
        );

        // Calculate padding to remove
        let (padding_left, padding_right) = if self.causal {
            let pr = (self.padding_total as f64 * self.trim_right_ratio).ceil() as usize;
            let pl = self.padding_total - pr;
            (pl, pr)
        } else {
            let pr = self.padding_total / 2;
            let pl = self.padding_total - pr;
            (pl, pr)
        };

        // Remove padding
        if padding_left + padding_right > 0 {
            full_output = unpad1d(&full_output, (padding_left, padding_right))?;
        }
        debug!(
            "[CACHE {}] After unpadding: {:?}",
            layer_id,
            full_output.dims()
        );

        // CRITICAL: Python behavior differs between first chunk and subsequent chunks!
        // - First chunk (empty cache): return ALL output
        // - Subsequent chunks (has cache): return only the last expected_new_output samples
        let output = if is_first_chunk {
            // First chunk - return all output (matching Python lines 522-524)
            debug!("[CACHE {}] First chunk - returning all output", layer_id);
            full_output.clone()
        } else {
            // Subsequent chunks - return only the new output (matching Python lines 525-533)
            let expected_new_output = t * self.stride;
            let output_len = full_output.dim(D::Minus1)?;
            if output_len >= expected_new_output {
                debug!(
                    "[CACHE {}] Taking last {} samples",
                    layer_id, expected_new_output
                );
                full_output.narrow(
                    D::Minus1,
                    output_len - expected_new_output,
                    expected_new_output,
                )?
            } else {
                // This should not happen in normal operation - indicates potential audio loss
                tracing::warn!(
                    "[CACHE {}] Output shorter than expected: {} < {} (may cause audio truncation)",
                    layer_id,
                    output_len,
                    expected_new_output
                );
                full_output.clone()
            }
        };
        debug!(
            "[CACHE {}] Final streaming output shape: {:?}",
            layer_id,
            output.dims()
        );

        // Update cache with last context_size samples
        // IMPORTANT: Call .contiguous() to ensure we store an independent copy, not a view
        let context_size = self.kernel_size - 1;
        let full_input_len = full_input.dim(D::Minus1)?;
        let new_cache = if full_input_len > context_size {
            full_input
                .narrow(D::Minus1, full_input_len - context_size, context_size)?
                .contiguous()?
        } else {
            full_input.contiguous()?
        };
        debug!("[CACHE {}] Storing cache: {:?}", layer_id, new_cache.dims());
        cache.set(layer_id.to_string(), new_cache)?;

        Ok(output)
    }
}

/// FFN - Feed Forward Network
/// Python: class FFN(nn.Module) in modular_vibevoice_tokenizer.py:579
pub struct FFN {
    pub linear1: Linear,
    pub linear2: Linear,
}

impl FFN {
    pub fn new(vb: VarBuilder, embed_dim: usize, ffn_dim: usize) -> Result<Self> {
        Ok(Self {
            linear1: linear(embed_dim, ffn_dim, vb.pp("linear1"))?,
            linear2: linear(ffn_dim, embed_dim, vb.pp("linear2"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        // Use gelu_erf (exact ERF-based) to match Python's ACT2FN["gelu"]
        // Note: gelu() uses tanh approximation which differs by ~2e-4 per value
        let x = x.gelu_erf()?;
        Ok(self.linear2.forward(&x)?)
    }
}

/// Block1D - Transformer-like block with depthwise conv mixer and FFN
/// Python: class Block1D(nn.Module) in modular_vibevoice_tokenizer.py:620
pub struct Block1D {
    pub norm: ConvRMSNorm,
    pub mixer: SConv1d,
    pub gamma: Option<Tensor>,
    pub ffn_norm: ConvRMSNorm,
    pub ffn: FFN,
    pub ffn_gamma: Option<Tensor>,
    pub drop_path: Option<Tensor>, // For stochastic depth (default: identity)
}

impl Block1D {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vb: VarBuilder,
        dim: usize,
        kernel_size: usize,
        causal: bool,
        pad_mode: &str,
        layernorm_eps: f64,
        layer_scale_init_value: f64,
        ffn_expansion: usize,
        mixer_groups: usize, // dim for depthwise, 1 for regular conv
    ) -> Result<Self> {
        // Normalization layers (Python lines 628-630)
        let norm = ConvRMSNorm::new(vb.pp("norm"), dim, layernorm_eps, true)?;
        let ffn_norm = ConvRMSNorm::new(vb.pp("ffn_norm"), dim, layernorm_eps, true)?;

        // Mixer layer (depthwise conv or regular conv) (Python lines 640-647)
        // Model has path: stages.X.Y.mixer.conv.conv.conv.weight (4 levels total)
        // Block1D provides: mixer.conv, SConv1d adds: conv.conv
        let mixer = SConv1d::new(
            vb.pp("mixer").pp("conv"),
            dim,
            dim,
            kernel_size,
            1, // stride
            1, // dilation
            mixer_groups,
            true, // bias
            causal,
            pad_mode,
        )?;

        // FFN (Python line 651)
        let ffn = FFN::new(vb.pp("ffn"), dim, ffn_expansion * dim)?;

        // Layer scale parameters (Python lines 658-663)
        let gamma = if layer_scale_init_value > 0.0 {
            Some(vb.get(dim, "gamma")?)
        } else {
            None
        };

        let ffn_gamma = if layer_scale_init_value > 0.0 {
            Some(vb.get(dim, "ffn_gamma")?)
        } else {
            None
        };

        // Drop path (Python line 656: nn.Identity() if drop_path <= 0. else nn.modules.DropPath(drop_path))
        // For now, implement as identity since default config has drop_path_rate = 0.0
        let drop_path = None; // Identity operation

        Ok(Self {
            norm,
            mixer,
            gamma,
            ffn_norm,
            ffn,
            ffn_gamma,
            drop_path,
        })
    }

    /// Forward pass with streaming cache support for the mixer convolution
    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        cache: &mut crate::streaming_cache::StreamingCache,
        layer_id: &str,
    ) -> Result<Tensor> {
        // Mixer path with cache (only mixer needs caching)
        let residual = x;
        let mut x = self.norm.forward(x)?;
        x = self.mixer.forward_with_cache(&x, cache, layer_id)?;

        // Apply gamma scaling if present
        if let Some(ref gamma) = self.gamma {
            let gamma_expanded = gamma.reshape((1, gamma.dims()[0], 1))?;
            x = x.broadcast_mul(&gamma_expanded)?;
        }

        // First residual connection with drop path
        let mut x = if let Some(ref dp) = self.drop_path {
            (residual + dp.broadcast_mul(&x)?)?
        } else {
            (residual + x)?
        };

        // FFN path (no caching needed for FFN)
        let residual_ffn = x.clone();
        x = self.ffn_norm.forward(&residual_ffn)?;

        // Permute: b c t -> b t c for FFN (contiguous for CUDA compatibility)
        x = x.transpose(1, 2)?.contiguous()?;
        x = self.ffn.forward(&x)?;

        // Permute back: b t c -> b c t (contiguous for CUDA compatibility)
        x = x.transpose(1, 2)?.contiguous()?;

        // Apply ffn_gamma scaling if present
        if let Some(ref ffn_gamma) = self.ffn_gamma {
            let ffn_gamma_expanded = ffn_gamma.reshape((1, ffn_gamma.dims()[0], 1))?;
            x = x.broadcast_mul(&ffn_gamma_expanded)?;
        }

        // Second residual connection with drop path
        if let Some(ref dp) = self.drop_path {
            Ok((residual_ffn + dp.broadcast_mul(&x)?)?)
        } else {
            Ok((&residual_ffn + x)?)
        }
    }
}
