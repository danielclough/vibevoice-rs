/// VAE Decoder implementation matching Python TokenizerDecoder
/// Reference: VibeVoice/vibevoice/modular/modular_vibevoice_tokenizer.py:816
use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarBuilder;
use tracing::{debug, info};

use crate::config::VAEDecoderConfig;
use crate::vae_layers::{Block1D, ConvRMSNorm, SConv1d, SConvTranspose1d};

/// VAE Stage containing multiple Block1D modules
/// Matches Python: self.stages in TokenizerDecoder (line 895)
pub struct VAEStage {
    blocks: Vec<Block1D>,
}

impl VAEStage {
    pub fn new(
        vb: VarBuilder,
        #[allow(unused_variables)]
        stage_idx: usize,
        dim: usize,
        num_blocks: usize,
        kernel_size: usize,
        causal: bool,
        pad_mode: &str,
        layernorm_eps: f64,
        layer_scale_init_value: f64,
        ffn_expansion: usize,
        mixer_groups: usize,
    ) -> Result<Self> {
        let mut blocks = Vec::new();

        for block_idx in 0..num_blocks {
            // Weight path: stages.{stage_idx}.{block_idx}.*
            let block_vb = vb.pp(&block_idx.to_string());
            let block = Block1D::new(
                block_vb,
                dim,
                kernel_size,
                causal,
                pad_mode,
                layernorm_eps,
                layer_scale_init_value,
                ffn_expansion,
                mixer_groups,
            )?;
            blocks.push(block);
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

/// VAE Decoder matching Python TokenizerDecoder architecture
/// Reference: VibeVoice/vibevoice/modular/modular_vibevoice_tokenizer.py:816
pub struct VAEDecoder {
    config: VAEDecoderConfig,
    upsample_layers: Vec<UpsampleLayer>,
    stages: Vec<VAEStage>,
    norm: Option<ConvRMSNorm>,
    head: SConv1d,
}

/// Enum to represent either stem conv or transposed conv in upsample_layers
enum UpsampleLayer {
    Stem(SConv1d),
    Upsample(SConvTranspose1d),
}

impl UpsampleLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            UpsampleLayer::Stem(conv) => conv.forward(x),
            UpsampleLayer::Upsample(convtr) => convtr.forward(x),
        }
    }

    /// Forward pass with streaming cache support
    fn forward_with_cache(
        &self,
        x: &Tensor,
        cache: &mut crate::streaming_cache::StreamingCache,
        layer_id: &str,
    ) -> Result<Tensor> {
        match self {
            UpsampleLayer::Stem(conv) => conv.forward_with_cache(x, cache, layer_id),
            UpsampleLayer::Upsample(convtr) => convtr.forward_with_cache(x, cache, layer_id),
        }
    }
}

impl VAEDecoder {
    pub fn new(vb: VarBuilder, config: VAEDecoderConfig) -> Result<Self> {
        info!("\n🔧 Initializing VAE Decoder (Python-matched architecture)...");
        debug!(
            "  Config: depths={:?}, ratios={:?}",
            config.depths, config.ratios
        );

        let num_stages = config.depths.len();
        let mut upsample_layers = Vec::new();
        let mut stages = Vec::new();

        // Calculate mixer groups based on mixer_layer type
        let _mixer_groups = if config.mixer_layer == "depthwise_conv" {
            0 // Will be set to dim per stage
        } else {
            1
        };

        // Create upsample layers matching Python lines 864-880
        for i in 0..num_stages {
            if i == 0 {
                // Stem: dimension -> n_filters * 2^(num_stages-1)
                let out_channels = config.n_filters * 2_usize.pow((num_stages - 1) as u32);
                debug!("  Stage {}: Stem {} -> {}", i, config.vae_dim, out_channels);

                let stem = SConv1d::new(
                    vb.pp("upsample_layers").pp(&i.to_string()).pp("0"),
                    config.vae_dim,
                    out_channels,
                    config.kernel_size,
                    1, // stride
                    1, // dilation
                    1, // groups
                    config.conv_bias,
                    config.causal,
                    &config.pad_mode,
                )?;
                upsample_layers.push(UpsampleLayer::Stem(stem));
            } else {
                // Upsampling: in_ch -> out_ch
                let in_ch = config.n_filters * 2_usize.pow((num_stages - 1 - (i - 1)) as u32);
                let out_ch = config.n_filters * 2_usize.pow((num_stages - 1 - i) as u32);
                let ratio = config.ratios[i - 1];
                let kernel_size = ratio * 2;

                debug!(
                    "  Stage {}: Upsample {} -> {} (ratio={})",
                    i, in_ch, out_ch, ratio
                );

                let upsample = SConvTranspose1d::new(
                    vb.pp("upsample_layers").pp(&i.to_string()).pp("0"),
                    in_ch,
                    out_ch,
                    kernel_size,
                    ratio,
                    config.conv_bias,
                    config.causal,
                    config.trim_right_ratio,
                )?;
                upsample_layers.push(UpsampleLayer::Upsample(upsample));
            }
        }

        // Create stages matching Python lines 899-906
        // Track last channel count (like Python's in_ch that persists after loop)
        let mut final_channels = config.n_filters;
        for i in 0..num_stages {
            let channels = config.n_filters * 2_usize.pow((num_stages - 1 - i) as u32);
            final_channels = channels;
            let depth = config.depths[i];
            let stage_mixer_groups = if config.mixer_layer == "depthwise_conv" {
                channels // Depthwise: groups = channels
            } else {
                1 // Regular conv
            };

            debug!("  Stage {}: {} blocks, {} channels", i, depth, channels);

            let stage = VAEStage::new(
                vb.pp("stages").pp(&i.to_string()),
                i,
                channels,
                depth,
                config.kernel_size,
                config.causal,
                &config.pad_mode,
                config.layernorm_eps,
                config.layer_scale_init_value,
                config.ffn_expansion,
                stage_mixer_groups,
            )?;
            stages.push(stage);
        }

        // Final normalization (Python lines 908-911)
        // final_channels already holds last stage's channel count from loop above
        let norm = if !config.disable_last_norm {
            Some(ConvRMSNorm::new(
                vb.pp("norm"),
                final_channels,
                config.layernorm_eps,
                config.layernorm_elementwise_affine,
            )?)
        } else {
            None
        };

        // Head convolution (Python line 912)
        // Model has path: head.conv.conv.weight (3 levels total)
        // VAE decoder provides: head, SConv1d adds: conv.conv
        let head = SConv1d::new(
            vb.pp("head"),
            final_channels,
            config.channels,
            config.last_kernel_size,
            1, // stride
            1, // dilation
            1, // groups
            config.conv_bias,
            config.causal,
            &config.pad_mode,
        )?;

        info!("✓ VAE Decoder initialized successfully\n");

        Ok(Self {
            config,
            upsample_layers,
            stages,
            norm,
            head,
        })
    }

    /// Decode latents to audio
    /// Matches Python TokenizerDecoder.forward (lines 948-951)
    pub fn decode(&self, x: &Tensor) -> Result<Tensor> {
        debug!("\n🔍 === VAE DECODE (Python-matched) ===");
        debug!("  Input shape: {:?}", x.dims());

        let mut x = x.clone();

        // Forward features matching Python forward_features (lines 914-946)
        for i in 0..self.config.depths.len() {
            // Apply upsampling layer
            x = self.upsample_layers[i].forward(&x)?;
            debug!("  After upsample_layer[{}]: {:?}", i, x.dims());

            // Apply stage (Block1D modules)
            x = self.stages[i].forward(&x)?;
            debug!("  After stage[{}]: {:?}", i, x.dims());
        }

        // Apply final normalization if present
        if let Some(ref norm) = self.norm {
            x = norm.forward(&x)?;
            debug!("  After norm: {:?}", x.dims());
        }

        // Apply head convolution
        x = self.head.forward(&x)?;
        debug!("  After head: {:?}", x.dims());

        // Apply tanh activation (output in [-1, 1])
        let audio = x.tanh()?;
        debug!("  After tanh: {:?}", audio.dims());
        debug!("🔍 =====================================\n");

        Ok(audio)
    }

    /// Decode latents to audio with streaming cache support
    /// Matches Python TokenizerDecoder._decode_frame for streaming decoding
    /// Reference: VibeVoice/vibevoice/modular/modular_vibevoice_tokenizer.py:953
    pub fn decode_with_cache(
        &self,
        x: &Tensor,
        cache: &mut crate::streaming_cache::StreamingCache,
    ) -> Result<Tensor> {
        debug!("\n🔍 === VAE DECODE WITH CACHE (streaming) ===");
        debug!("  Input shape: {:?}", x.dims());
        debug!(
            "  Token count: {}, in_warmup: {}",
            cache.tokens_processed(),
            cache.in_warmup()
        );

        // Debug: check input values
        let input_flat = x.flatten_all()?;
        let input_vals: Vec<f32> = input_flat.to_vec1()?;
        let input_rms =
            (input_vals.iter().map(|v| v * v).sum::<f32>() / input_vals.len() as f32).sqrt();
        if cache.tokens_processed() < 3 {
            info!(
                "  🔍 DECODER INPUT RMS (token {}): {:.6}",
                cache.tokens_processed(),
                input_rms
            );
        }

        let mut x = x.clone();

        // Forward features with streaming cache
        for i in 0..self.config.depths.len() {
            // Apply upsampling layer with cache
            let upsample_id = format!("upsample_{}", i);
            x = self.upsample_layers[i].forward_with_cache(&x, cache, &upsample_id)?;
            debug!("  After upsample_layer[{}]: {:?}", i, x.dims());

            // Debug: check RMS after first upsample
            if i == 0 && cache.tokens_processed() < 3 {
                let flat = x.flatten_all()?;
                let vals: Vec<f32> = flat.to_vec1()?;
                let rms = (vals.iter().map(|v| v * v).sum::<f32>() / vals.len() as f32).sqrt();
                info!(
                    "  🔍 AFTER STEM RMS (token {}): {:.6}",
                    cache.tokens_processed(),
                    rms
                );
            }

            // Apply stage (Block1D modules) with cache
            let stage_id = format!("stage_{}", i);
            x = self.stages[i].forward_with_cache(&x, cache, &stage_id)?;
            debug!("  After stage[{}]: {:?}", i, x.dims());
        }

        // Apply final normalization if present (no caching needed)
        if let Some(ref norm) = self.norm {
            x = norm.forward(&x)?;
            debug!("  After norm: {:?}", x.dims());
        }

        // Apply head convolution with cache
        x = self.head.forward_with_cache(&x, cache, "head")?;
        debug!("  After head: {:?}", x.dims());

        // Apply tanh activation (output in [-1, 1])
        let audio = x.tanh()?;
        debug!("  After tanh: {:?}", audio.dims());

        // Debug: check output RMS for first few tokens
        if cache.tokens_processed() < 3 {
            let flat = audio.flatten_all()?;
            let vals: Vec<f32> = flat.to_vec1()?;
            let rms = (vals.iter().map(|v| v * v).sum::<f32>() / vals.len() as f32).sqrt();
            info!(
                "  🔍 FINAL AUDIO RMS (token {}): {:.6}",
                cache.tokens_processed(),
                rms
            );
        }

        // Increment token counter for next call
        cache.increment_token_count();

        debug!("🔍 =====================================\n");

        Ok(audio)
    }
}
