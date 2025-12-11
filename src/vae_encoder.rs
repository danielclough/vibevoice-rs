use crate::config::{AcousticTokenizerConfig, VibeVoiceConfig};
use crate::vae_layers::{ConvRMSNorm, SConv1d};
use crate::vae_utils::VAEStage;
use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarBuilder;
use tracing::{debug, info};

// VAE Encoder - mirrors decoder but reverses the process (audio → latents)
// Can be used for both acoustic and semantic tokenizers
pub struct VAEEncoder {
    pub downsample_layers: Vec<SConv1d>,
    pub stages: Vec<VAEStage>,
    pub norm: Option<ConvRMSNorm>, // Final norm before head (Python: self.norm)
    pub head: SConv1d,             // Final output layer
}

impl VAEEncoder {
    /// Create encoder from AcousticTokenizerConfig (legacy)
    pub fn new(vb: VarBuilder, config: &AcousticTokenizerConfig) -> Result<Self> {
        info!("\n🔧 Initializing VAE encoder (acoustic) with config...");

        // Parse encoder depths string (e.g., "3-3-3-3-3-3-8" -> [3,3,3,3,3,3,8])
        let depths: Vec<usize> = config
            .encoder_depths
            .split('-')
            .map(|s| s.parse().unwrap())
            .collect();

        let num_stages = depths.len();
        // Reverse ratios like Python TokenizerEncoder does
        let ratios: Vec<usize> = config.encoder_ratios.iter().rev().cloned().collect();
        info!("  Encoder depths: {:?}", depths);
        info!("  Encoder ratios (original): {:?}", config.encoder_ratios);
        info!("  Encoder ratios (reversed): {:?}", ratios);

        // Create downsample layers dynamically
        let mut downsample_layers = Vec::new();

        // First layer (stem) - no downsampling, just channel increase
        let stem = SConv1d::new(
            vb.pp("downsample_layers.0.0"),
            config.channels,
            config.encoder_n_filters,
            7, // kernel_size
            1, // stride
            1, // dilation
            1, // groups
            config.conv_bias,
            config.causal,
            &config.pad_mode,
        )?;
        downsample_layers.push(stem);

        // Subsequent downsample layers with increasing channels and ratios
        for (i, &ratio) in ratios.iter().enumerate() {
            let in_channels = config.encoder_n_filters * (2_usize.pow(i as u32));
            let out_channels = config.encoder_n_filters * (2_usize.pow((i + 1) as u32));

            // Calculate kernel size and stride based on ratio (Python: kernel_size=self.ratios[i] * 2, stride=self.ratios[i])
            let kernel_size = ratio * 2;
            let stride = ratio;

            let downsample = SConv1d::new(
                vb.pp(format!("downsample_layers.{}.0", i + 1)),
                in_channels,
                out_channels,
                kernel_size,
                stride,
                1, // dilation
                1, // groups
                config.conv_bias,
                config.causal,
                &config.pad_mode,
            )?;
            downsample_layers.push(downsample);
        }

        // Create stages dynamically based on depths
        let mut stages = Vec::new();
        for (stage_idx, &num_blocks) in depths.iter().enumerate() {
            let channels = config.encoder_n_filters * (2_usize.pow(stage_idx as u32));
            let stage =
                VAEStage::new(vb.pp(format!("stages.{}", stage_idx)), channels, num_blocks)?;
            stages.push(stage);
        }

        // Final norm before head (Python: self.norm = norm_type(in_ch, eps=layernorm_eps))
        // Applied after all stages, before head
        let head_channels = config.encoder_n_filters * (2_usize.pow((num_stages - 1) as u32));
        let norm = if !config.disable_last_norm {
            Some(ConvRMSNorm::new(
                vb.pp("norm"),
                head_channels,
                config.layernorm_eps,
                true, // elementwise_affine
            )?)
        } else {
            None
        };

        // Head layer - use SConv1d to match Python structure
        let head = SConv1d::new(
            vb.pp("head"),
            head_channels,
            config.vae_dim,
            7,                // kernel_size (last_kernel_size)
            1,                // stride
            1,                // dilation
            1,                // groups
            config.conv_bias, // bias
            config.causal,
            &config.pad_mode,
        )?;

        Ok(Self {
            downsample_layers,
            stages,
            norm,
            head,
        })
    }

    /// Create encoder from VibeVoiceConfig (supports both acoustic and semantic)
    /// mode: "acoustic" or "semantic"
    pub fn new_from_vibevoice_config(
        vb: VarBuilder,
        config: &VibeVoiceConfig,
        mode: &str,
    ) -> Result<Self> {
        match mode {
            "acoustic" => Self::new(vb, &config.acoustic_tokenizer_config),
            "semantic" => {
                // Semantic tokenizer uses same architecture but different config
                let sem_cfg = &config.semantic_tokenizer_config;
                info!("\n🔧 Initializing VAE encoder (semantic) with config...");

                let depths: Vec<usize> = sem_cfg
                    .encoder_depths
                    .split('-')
                    .map(|s| s.parse().unwrap())
                    .collect();

                let num_stages = depths.len();
                let ratios: Vec<usize> = sem_cfg.encoder_ratios.iter().rev().cloned().collect();
                info!("  Encoder depths: {:?}", depths);
                info!("  Encoder ratios (reversed): {:?}", ratios);

                let mut downsample_layers = Vec::new();

                // First layer (stem)
                let stem = SConv1d::new(
                    vb.pp("downsample_layers.0.0"),
                    sem_cfg.channels,
                    sem_cfg.encoder_n_filters,
                    7,
                    1, // stride
                    1, // dilation
                    1, // groups
                    sem_cfg.conv_bias,
                    sem_cfg.causal,
                    &sem_cfg.pad_mode,
                )?;
                downsample_layers.push(stem);

                // Subsequent downsample layers
                for (i, &ratio) in ratios.iter().enumerate() {
                    let in_channels = sem_cfg.encoder_n_filters * (2_usize.pow(i as u32));
                    let out_channels = sem_cfg.encoder_n_filters * (2_usize.pow((i + 1) as u32));
                    let kernel_size = ratio * 2;
                    let stride = ratio;

                    let downsample = SConv1d::new(
                        vb.pp(format!("downsample_layers.{}.0", i + 1)),
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        1, // dilation
                        1, // groups
                        sem_cfg.conv_bias,
                        sem_cfg.causal,
                        &sem_cfg.pad_mode,
                    )?;
                    downsample_layers.push(downsample);
                }

                // Create stages
                let mut stages = Vec::new();
                for (stage_idx, &depth) in depths.iter().enumerate() {
                    let stage = VAEStage::new(
                        vb.pp(format!("stages.{}", stage_idx)),
                        sem_cfg.encoder_n_filters * (2_usize.pow(stage_idx as u32)),
                        depth,
                    )?;
                    stages.push(stage);
                }

                // Final norm before head
                let final_channels =
                    sem_cfg.encoder_n_filters * (2_usize.pow(num_stages as u32 - 1));
                let norm = if !sem_cfg.disable_last_norm {
                    Some(ConvRMSNorm::new(
                        vb.pp("norm"),
                        final_channels,
                        sem_cfg.layernorm_eps,
                        true,
                    )?)
                } else {
                    None
                };

                // Final projection head
                let head = SConv1d::new(
                    vb.pp("head"),
                    final_channels,
                    sem_cfg.vae_dim,
                    7, // kernel_size - keep for weight compatibility
                    1,
                    1,
                    1,
                    sem_cfg.conv_bias,
                    sem_cfg.causal,
                    &sem_cfg.pad_mode,
                )?;

                Ok(Self {
                    downsample_layers,
                    stages,
                    norm,
                    head,
                })
            }
            _ => Err(anyhow::Error::msg(format!(
                "Unknown mode: {}. Use 'acoustic' or 'semantic'",
                mode
            ))),
        }
    }

    /// Encode audio to latent representation
    /// Input: audio [batch, 1, samples] at 24kHz
    /// Output: latents [batch, 64, vae_tokens]
    /// Where vae_tokens = ceil(samples / 3200)
    pub fn encode(&self, audio: &Tensor) -> Result<Tensor> {
        debug!("\n🔍 === VAE ENCODE ===");
        debug!("  Input audio shape: {:?}", audio.dims());

        let mut x = audio.clone();

        // Apply downsample layers and stages dynamically
        for stage_idx in 0..self.stages.len() {
            // Apply downsample layer (except for the first stage which doesn't have a downsample before it)
            if stage_idx > 0 {
                x = self.downsample_layers[stage_idx].forward(&x)?;
                debug!("  After downsample_{}: {:?}", stage_idx, x.dims());
            } else {
                // First downsample (stem) - applied to input
                x = self.downsample_layers[0].forward(&x)?;
                debug!("  After downsample_0 (stem): {:?}", x.dims());
            }

            // Apply stage blocks
            x = self.stages[stage_idx].forward(&x)?;
            debug!("  After stage_{}: {:?}", stage_idx, x.dims());
        }

        // Apply final norm before head (Python: return self.norm(x) in forward_features)
        if let Some(ref norm) = self.norm {
            x = norm.forward(&x)?;
            debug!("  After final norm: {:?}", x.dims());
        }

        // Apply head to get final latents
        x = self.head.forward(&x)?;
        debug!("  Final latents shape: {:?}", x.dims());
        debug!(
            "  Expected compression: samples / 3200 = {} tokens",
            audio.dims()[2] as f32 / 3200.0
        );

        Ok(x)
    }

    /// Encode audio with streaming cache support
    /// This allows incremental processing, outputting only new tokens
    /// Matches Python's forward_features + forward with cache support
    pub fn encode_with_cache(
        &self,
        audio: &Tensor,
        cache: &mut crate::streaming_cache::StreamingCache,
    ) -> Result<Tensor> {
        let mut x = audio.clone();

        // Apply downsample layers and stages WITH CACHE
        // Each downsample layer gets a unique cache ID
        for stage_idx in 0..self.stages.len() {
            // Apply downsample layer with cache
            let layer_id = format!("downsample_layers.{}.0", stage_idx);
            if stage_idx > 0 {
                x = self.downsample_layers[stage_idx].forward_with_cache(&x, cache, &layer_id)?;
            } else {
                x = self.downsample_layers[0].forward_with_cache(&x, cache, &layer_id)?;
            }

            // Apply stage blocks with cache
            let stage_id = format!("stage_{}", stage_idx);
            x = self.stages[stage_idx].forward_with_cache(&x, cache, &stage_id)?;
        }

        // Apply final norm before head
        if let Some(ref norm) = self.norm {
            x = norm.forward(&x)?;
        }

        // Apply head with cache
        x = self.head.forward_with_cache(&x, cache, "head")?;

        Ok(x)
    }
}
