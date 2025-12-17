//! VibeVoice Realtime (0.5B) streaming model.
//!
//! This module provides the main model interface for real-time streaming TTS.
//!
//! # Architecture
//!
//! The realtime model uses a split LLM architecture:
//! - `language_model`: Lower 4 Qwen2 layers (text processing)
//! - `tts_language_model`: Upper 20 Qwen2 layers (TTS generation)
//! - `tts_eos_classifier`: Binary classifier for EOS detection
//! - `diffusion_head`: 5-step diffusion for speech synthesis
//! - `acoustic_connector`: Projects speech latents to LLM space
//! - `vae_decoder`: Converts latents to audio waveform
//!
//! # Usage
//!
//! ```ignore
//! // Load model
//! let model = VibeVoiceRealtimeModel::from_pretrained(&model_path, &device)?;
//!
//! // Load voice cache
//! let voice_cache = VoiceCache::from_safetensors(&voice_path, &device)?;
//!
//! // Generate with streaming
//! let audio = model.generate(
//!     "Hello, world!",
//!     &voice_cache,
//!     |chunk| {
//!         // Stream audio chunk
//!         Ok(())
//!     },
//! )?;
//! ```

use crate::acoustic_connector::AcousticConnector;
use crate::config::VAEDecoderConfig;
use crate::diffusion::{DPMSolverPP, DiffusionHead};
use crate::realtime::{
    BinaryClassifier, DualSplitLLM, GenerationConfig, RealtimeConfig, TTS_SPEECH_WINDOW_SIZE,
    VoiceCache, WindowedGenerator,
};
use crate::streaming_cache::StreamingCache;
use crate::utils::{create_realtime_remapped_varbuilder, seeded_randn, tensor_stats};
use crate::vae_decoder::VAEDecoder;

use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use std::path::Path;
use tokenizers::Tokenizer;
use tracing::{debug, info};

/// VibeVoice Realtime streaming model.
///
/// This is the main entry point for real-time TTS generation.
pub struct VibeVoiceRealtimeModel {
    /// Device for tensor operations
    pub device: Device,

    /// Model configuration
    config: RealtimeConfig,

    /// Dual Split LLM for CFG (4 lower + 20 upper layers, both pos/neg paths)
    dual_split_llm: DualSplitLLM,

    /// EOS classifier for detecting end of speech
    eos_classifier: BinaryClassifier,

    /// Diffusion head for speech synthesis
    diffusion_head: DiffusionHead,

    /// Acoustic connector (latent → LLM embedding)
    acoustic_connector: AcousticConnector,

    /// VAE decoder (latent → audio)
    vae_decoder: VAEDecoder,

    /// DPM-Solver++ for diffusion sampling
    solver: DPMSolverPP,

    /// Tokenizer for text processing
    tokenizer: Tokenizer,

    /// Speech normalization factors
    speech_scaling_factor: f32,
    speech_bias_factor: f32,

    /// Pre-computed sigma schedule for diffusion
    precomputed_sigmas: Vec<f32>,
}

/// Compute sigma schedule matching Python's betas_for_alpha_bar approach.
fn compute_sigmas_schedule(num_train_timesteps: usize) -> Vec<f32> {
    use std::f64::consts::PI;

    let alpha_bar_fn = |t: f64| -> f64 { ((t + 0.008) / 1.008 * PI / 2.0).cos().powi(2) };

    let max_beta: f64 = 0.999;
    let mut betas = Vec::with_capacity(num_train_timesteps);
    for i in 0..num_train_timesteps {
        let t1 = i as f64 / num_train_timesteps as f64;
        let t2 = (i + 1) as f64 / num_train_timesteps as f64;
        let beta = (1.0 - alpha_bar_fn(t2) / alpha_bar_fn(t1)).min(max_beta);
        betas.push(beta);
    }

    let alphas: Vec<f64> = betas.iter().map(|b| 1.0 - b).collect();

    let mut alphas_cumprod = Vec::with_capacity(num_train_timesteps);
    let mut running_product = 1.0f64;
    for alpha in alphas.iter() {
        running_product *= alpha;
        alphas_cumprod.push(running_product);
    }

    alphas_cumprod
        .iter()
        .map(|&acp| ((1.0 - acp) / acp.max(1e-20)).sqrt() as f32)
        .collect()
}

impl VibeVoiceRealtimeModel {
    /// Load a pretrained model from a directory.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to model directory containing:
    ///   - `config.json` - Model configuration
    ///   - `model.safetensors` - Model weights
    ///   - `tokenizer.json` - Tokenizer
    /// * `device` - Device to load model onto
    pub fn from_pretrained(model_path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let model_path = model_path.as_ref();

        // Load config
        let config_path = model_path.join("config.json");
        let config = RealtimeConfig::from_file(&config_path)?;

        info!(
            "Loading VibeVoice-Realtime ({} lower + {} upper layers)",
            config.lm_num_layers(),
            config.tts_lm_num_layers()
        );

        // Load tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // Load weights with remapping for Qwen2Model compatibility
        let model_dir = model_path.to_path_buf();
        let vb = create_realtime_remapped_varbuilder(&model_dir, device)?;

        // Get image_pad token ID for CFG unconditional baseline
        // This is <|image_pad|> in Qwen2 tokenizer, default 151655
        let image_pad_token_id = tokenizer.token_to_id("<|image_pad|>").unwrap_or(151655);
        debug!("Using image_pad_token_id: {}", image_pad_token_id);

        // Initialize components
        info!("Initializing Dual Split LLM (CFG support)...");
        let dual_split_llm = DualSplitLLM::new(vb.clone(), &config, image_pad_token_id)?;

        info!("Initializing EOS classifier...");
        let eos_classifier =
            BinaryClassifier::new(vb.pp("tts_eos_classifier"), config.hidden_size())?;

        info!("Initializing Diffusion Head...");
        let diffusion_head = DiffusionHead::new_with_params(
            vb.pp("model.prediction_head"),
            config.hidden_size(),
            config.diffusion_head_config.latent_size,
            config.diffusion_head_config.head_layers,
            Some("VibeVoice-Realtime"),
        )?;

        info!("Initializing Acoustic Connector...");
        let acoustic_connector = AcousticConnector::new_with_params(
            vb.pp("model.acoustic_connector"),
            config.acoustic_vae_dim,
            config.hidden_size(),
        )?;

        info!("Initializing VAE Decoder...");
        let vae_decoder = VAEDecoder::new(
            vb.pp("model.acoustic_tokenizer.decoder"),
            VAEDecoderConfig::from_acoustic_config(&config.acoustic_tokenizer_config),
        )?;

        // DPM-Solver with 5 steps for realtime
        let num_diffusion_steps = config.diffusion_head_config.ddpm_num_inference_steps;
        let solver = DPMSolverPP::new(num_diffusion_steps);

        // Load normalization factors
        let speech_scaling_factor = vb
            .get(&[1], "model.speech_scaling_factor")
            .or_else(|_| vb.get(&[], "model.speech_scaling_factor"))?;
        let speech_bias_factor = vb
            .get(&[1], "model.speech_bias_factor")
            .or_else(|_| vb.get(&[], "model.speech_bias_factor"))?;

        let scale_val = if speech_scaling_factor.dims().is_empty() {
            speech_scaling_factor.to_scalar::<f32>()?
        } else {
            speech_scaling_factor.to_vec1::<f32>()?[0]
        };
        let bias_val = if speech_bias_factor.dims().is_empty() {
            speech_bias_factor.to_scalar::<f32>()?
        } else {
            speech_bias_factor.to_vec1::<f32>()?[0]
        };

        debug!(
            "Speech normalization: scale={:.6}, bias={:.6}",
            scale_val, bias_val
        );

        // Pre-compute sigma schedule
        let num_train_timesteps = config.diffusion_head_config.ddpm_num_steps;
        let precomputed_sigmas = compute_sigmas_schedule(num_train_timesteps);

        info!("VibeVoice-Realtime model loaded successfully");

        Ok(Self {
            device: device.clone(),
            config,
            dual_split_llm,
            eos_classifier,
            diffusion_head,
            acoustic_connector,
            vae_decoder,
            solver,
            tokenizer,
            speech_scaling_factor: scale_val,
            speech_bias_factor: bias_val,
            precomputed_sigmas,
        })
    }

    /// Tokenize text for TTS generation.
    ///
    /// Applies Python-compatible preprocessing:
    /// - Strip leading/trailing whitespace
    /// - Append newline (matches `text_input.strip() + "\n"` in Python)
    fn tokenize(&self, text: &str) -> Result<Tensor> {
        // Match Python: text_input.strip() + "\n"
        let processed = format!("{}\n", text.trim());

        let encoding = self
            .tokenizer
            .encode(processed.as_str(), false)
            .map_err(|e| anyhow!("Tokenization error: {}", e))?;

        let ids: Vec<u32> = encoding.get_ids().to_vec();
        Ok(Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?)
    }

    /// Generate audio from text with streaming callback.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to synthesize
    /// * `voice_cache` - Pre-computed voice cache for speaker characteristics
    /// * `audio_callback` - Callback for streaming audio chunks
    ///
    /// # Returns
    ///
    /// Complete audio tensor of shape `[1, 1, samples]`
    pub fn generate<F>(
        &mut self,
        text: &str,
        voice_cache: &VoiceCache,
        mut audio_callback: F,
    ) -> Result<Tensor>
    where
        F: FnMut(&Tensor) -> Result<()>,
    {
        // Tokenize text
        let tts_text_ids = self.tokenize(text)?;
        let text_len = tts_text_ids.dim(1)?;

        info!("Generating speech for {} text tokens", text_len);

        // Create generator
        let gen_config = GenerationConfig {
            cfg_scale: 1.5,
            num_diffusion_steps: self.solver.num_steps,
            speech_scaling_factor: self.speech_scaling_factor,
            speech_bias_factor: self.speech_bias_factor,
        };

        let mut generator = WindowedGenerator::new(gen_config, &self.config, self.device.clone())?;
        generator.initialize_from_cache(voice_cache, &mut self.dual_split_llm)?;

        // Split text into windows
        let windows = crate::realtime::generation::text_windows(&tts_text_ids, 5)?;
        let num_windows = windows.len();
        let mut audio_chunks = Vec::new();
        let mut acoustic_cache = StreamingCache::new(self.device.clone());

        // Maximum generation length (matches Python's max_position_embeddings check)
        let max_speech_tokens = self.config.llm_config.max_position_embeddings;
        let mut total_speech_tokens = 0usize;
        let mut window_idx = 0usize;

        // Main generation loop - continues until EOS or max length
        // This matches Python's while True loop structure
        loop {
            // Process text window only if there's one available
            if window_idx < num_windows {
                let text_window = &windows[window_idx];
                debug!("Processing text window {}/{}", window_idx + 1, num_windows);

                // Forward text through LMs (both positive and negative paths)
                generator.process_text_window(text_window, &mut self.dual_split_llm)?;
                window_idx += 1;
            } else {
                debug!(
                    "Continuing speech generation after text exhausted (speech token {})",
                    total_speech_tokens
                );
            }

            // Generate speech tokens for this iteration (always 6 per iteration)
            for speech_idx in 0..TTS_SPEECH_WINDOW_SIZE {
                // Check max length before generating
                if total_speech_tokens >= max_speech_tokens {
                    info!(
                        "Reached maximum generation length {}, stopping",
                        max_speech_tokens
                    );
                    break;
                }

                // Get conditions for diffusion
                let positive_condition = generator.get_positive_condition()?;
                let negative_condition = generator.get_negative_condition()?;

                debug!(
                    "[generate] window {}/{}, speech {}/{}: pos_cond: {:?}, neg_cond: {:?}",
                    window_idx,
                    num_windows,
                    speech_idx + 1,
                    TTS_SPEECH_WINDOW_SIZE,
                    positive_condition.dims(),
                    negative_condition.dims()
                );

                // Log conditions for TOKEN 0 (matches Python's "[TOKEN 0]" output)
                if total_speech_tokens == 0 {
                    info!(
                        "🔍 [TOKEN 0] pos_condition: {}",
                        tensor_stats(&positive_condition)
                    );
                    info!(
                        "🔍 [TOKEN 0] neg_condition: {}",
                        tensor_stats(&negative_condition)
                    );
                }

                // Sample speech latent via diffusion (logs internally for TOKEN 0)
                let speech_latent = self.sample_diffusion(
                    &positive_condition,
                    &negative_condition,
                    total_speech_tokens == 0, // log_first_token
                )?;

                // Decode to audio
                // Denormalize: x = (latent - bias) / scale
                let scaled_latent = speech_latent.affine(
                    1.0 / self.speech_scaling_factor as f64,
                    -self.speech_bias_factor as f64,
                )?;

                // Log scaled_latent for TOKEN 0
                if total_speech_tokens == 0 {
                    info!(
                        "🔍 [TOKEN 0] scaled_latent (VAE input): {}",
                        tensor_stats(&scaled_latent)
                    );
                }

                // VAE expects [batch, channels, seq_len] = [1, 64, 1]
                let decoder_input = scaled_latent.unsqueeze(2)?;
                let audio_chunk = self
                    .vae_decoder
                    .decode_with_cache(&decoder_input, &mut acoustic_cache)?;

                // Log VAE output for first 6 tokens
                if total_speech_tokens < 6 {
                    debug!(
                        "🔊 [VAE TOKEN {}] output: {}",
                        total_speech_tokens,
                        tensor_stats(&audio_chunk)
                    );
                    // Log first 10 and last 10 audio samples for waveform comparison
                    let samples: Vec<f32> = audio_chunk.flatten_all()?.to_vec1()?;
                    let first_10: Vec<f32> = samples.iter().take(10).cloned().collect();
                    let last_10: Vec<f32> = samples.iter().rev().take(10).rev().cloned().collect();
                    debug!(
                        "🔊 [VAE TOKEN {}] first_10: {:?}",
                        total_speech_tokens, first_10
                    );
                    debug!(
                        "🔊 [VAE TOKEN {}] last_10: {:?}",
                        total_speech_tokens, last_10
                    );
                }

                // Stream and store
                audio_callback(&audio_chunk)?;
                audio_chunks.push(audio_chunk);
                total_speech_tokens += 1;

                // Project for next iteration
                let acoustic_embed = self
                    .acoustic_connector
                    .forward(&speech_latent.unsqueeze(1)?)?;

                // Update generator state (both positive and negative paths)
                generator.update_after_speech_token(&acoustic_embed, &mut self.dual_split_llm)?;

                // Check EOS
                if generator.check_eos(&self.eos_classifier)? {
                    debug!(
                        "EOS detected at window {}, speech token {}",
                        window_idx,
                        speech_idx + 1
                    );
                    break;
                }
            }

            // Check if finished (EOS detected) or max length reached
            if generator.is_finished() || total_speech_tokens >= max_speech_tokens {
                break;
            }
        }

        // Concatenate audio chunks
        if audio_chunks.is_empty() {
            return Ok(Tensor::zeros((1, 1, 0), DType::F32, &self.device)?);
        }

        let audio = Tensor::cat(&audio_chunks, 2)?;
        info!("Generated {} audio samples", audio.dim(2)?);

        Ok(audio)
    }

    /// Sample speech latent using DPM-Solver++ with CFG.
    fn sample_diffusion(
        &self,
        condition: &Tensor,
        neg_condition: &Tensor,
        log_first_token: bool,
    ) -> Result<Tensor> {
        let batch = condition.dim(0)?;
        let latent_size = self.config.acoustic_vae_dim;
        let num_steps = self.solver.num_steps;
        let num_train_timesteps = self.config.diffusion_head_config.ddpm_num_steps;
        let cfg_scale = 1.5f32;

        // Log diffusion config for first token
        if log_first_token {
            info!(
                "🔍 [DIFFUSION] cfg_scale={}, num_steps={}, latent_size={}",
                cfg_scale, num_steps, latent_size
            );
        }

        // Timestep calculation
        let last_timestep = num_train_timesteps;
        let n_points = num_steps + 1;
        let timesteps: Vec<i64> = (0..n_points)
            .map(|i| {
                let t = (i as f64 / (n_points - 1) as f64) * (last_timestep - 1) as f64;
                t.round() as i64
            })
            .rev()
            .take(num_steps)
            .collect();

        // Sigma calculation
        let mut sigmas: Vec<f32> = timesteps
            .iter()
            .map(|&t| {
                let t_idx = t as usize;
                if t_idx >= self.precomputed_sigmas.len() {
                    *self.precomputed_sigmas.last().unwrap()
                } else {
                    self.precomputed_sigmas[t_idx]
                }
            })
            .collect();
        sigmas.push(0.0);

        // Log timesteps and sigmas for first token
        if log_first_token {
            info!("🔍 [DIFFUSION] timesteps: {:?}", timesteps);
            info!("🔍 [DIFFUSION] sigmas: {:?}", sigmas);
        }

        let sigma_to_alpha_sigma = |sigma: f32| -> (f32, f32) {
            let alpha_t = 1.0 / (sigma.powi(2) + 1.0).sqrt();
            let sigma_t = sigma * alpha_t;
            (alpha_t, sigma_t)
        };

        // Initial noise (using PyTorch-compatible Box-Muller RNG)
        let doubled_batch = 2 * batch;
        let mut speech = seeded_randn(0.0, 1.0, &[doubled_batch, latent_size], &self.device)?;

        // Log initial noise for first token (full doubled tensor, like Python)
        if log_first_token {
            info!("🔍 [DIFFUSION] initial_noise: {}", tensor_stats(&speech));
        }

        // Concatenate conditions
        let conditions = Tensor::cat(&[condition, neg_condition], 0)?.contiguous()?;

        // Multistep solver state
        let solver_order = 2usize;
        let mut model_outputs: Vec<Option<Tensor>> = vec![None; solver_order];
        let mut lower_order_nums = 0usize;

        for step_index in 0..num_steps {
            let t = timesteps[step_index];
            let sigma_s = sigmas[step_index];
            let sigma_t = sigmas[step_index + 1];

            let (alpha_s, sigma_s_actual) = sigma_to_alpha_sigma(sigma_s);
            let (alpha_t, sigma_t_actual) = sigma_to_alpha_sigma(sigma_t);

            // Forward pass with CFG
            let half = speech.narrow(0, 0, batch)?.contiguous()?;
            let combined = Tensor::cat(&[&half, &half], 0)?.contiguous()?;

            let t_tensor = Tensor::new(&[t as f32], &self.device)?;
            let t_batch = t_tensor.broadcast_as((doubled_batch,))?;

            let model_output = self
                .diffusion_head
                .forward(&combined, &t_batch, &conditions)?;

            // Apply CFG
            let output_chunks: Vec<Tensor> = model_output.chunk(2, 0)?;
            let cond_output = &output_chunks[0];
            let uncond_output = &output_chunks[1];

            let diff = (cond_output - uncond_output)?;
            let half_output = (uncond_output + diff.affine(cfg_scale as f64, 0.0)?)?;

            // Log first step's intermediate values
            if log_first_token && step_index == 0 {
                info!(
                    "🔍 [DIFFUSION] model_output: {}",
                    tensor_stats(&model_output)
                );
                info!("🔍 [DIFFUSION] cond_output: {}", tensor_stats(cond_output));
                info!(
                    "🔍 [DIFFUSION] uncond_output: {}",
                    tensor_stats(uncond_output)
                );
                info!(
                    "🔍 [DIFFUSION] half_output (CFG): {}",
                    tensor_stats(&half_output)
                );
            }

            // Convert to x0_pred
            let x0_pred = (half.affine(alpha_s as f64, 0.0)?
                - half_output.affine(sigma_s_actual as f64, 0.0)?)?;

            // Determine solver order
            let lower_order_final =
                step_index == num_steps - 1 && (num_steps < 15 || sigma_t == 0.0);

            let use_first_order = solver_order == 1 || lower_order_nums < 1 || lower_order_final;

            // Shift outputs
            for i in 0..(solver_order - 1) {
                model_outputs[i] = model_outputs[i + 1].take();
            }
            model_outputs[solver_order - 1] = Some(x0_pred.clone());

            // Solver step
            let lambda_s = alpha_s.ln() - sigma_s_actual.ln();
            let lambda_t = alpha_t.ln() - sigma_t_actual.ln();
            let h = lambda_t - lambda_s;

            let prev_sample = if use_first_order {
                let coeff1 = sigma_t_actual / sigma_s_actual;
                let coeff2 = alpha_t * ((-h).exp() - 1.0);
                let term1 = speech.affine(coeff1 as f64, 0.0)?;
                let x0_full = Tensor::cat(&[&x0_pred, &x0_pred], 0)?.contiguous()?;
                let term2 = x0_full.affine(coeff2 as f64, 0.0)?;
                (term1 - term2)?
            } else {
                // Second-order with midpoint
                let m0 = model_outputs[solver_order - 1].as_ref().unwrap();
                let m1 = model_outputs[solver_order - 2].as_ref();

                if let Some(m1) = m1 {
                    let sigma_s1 = sigmas[step_index - 1];
                    let (alpha_s1, sigma_s1_actual) = sigma_to_alpha_sigma(sigma_s1);
                    let lambda_s1 = alpha_s1.ln() - sigma_s1_actual.ln();

                    let h_0 = lambda_s - lambda_s1;
                    let r0 = h_0 / h;

                    let d0 = m0;
                    let d1 = ((m0 - m1)? * (1.0 / r0 as f64))?;

                    let coeff1 = sigma_t_actual / sigma_s_actual;
                    let coeff2 = alpha_t * ((-h).exp() - 1.0);
                    let coeff3 = 0.5 * coeff2;

                    let term1 = speech.affine(coeff1 as f64, 0.0)?;
                    let d0_full = Tensor::cat(&[d0, d0], 0)?.contiguous()?;
                    let d1_full = Tensor::cat(&[&d1, &d1], 0)?.contiguous()?;
                    let term2 = d0_full.affine(coeff2 as f64, 0.0)?;
                    let term3 = d1_full.affine(coeff3 as f64, 0.0)?;
                    ((term1 - term2)? - term3)?
                } else {
                    let coeff1 = sigma_t_actual / sigma_s_actual;
                    let coeff2 = alpha_t * ((-h).exp() - 1.0);
                    let term1 = speech.affine(coeff1 as f64, 0.0)?;
                    let x0_full = Tensor::cat(&[&x0_pred, &x0_pred], 0)?.contiguous()?;
                    let term2 = x0_full.affine(coeff2 as f64, 0.0)?;
                    (term1 - term2)?
                }
            };

            speech = prev_sample;

            // Log step progress for first token (full doubled tensor, like Python)
            if log_first_token {
                info!(
                    "🔍 [DIFFUSION] step {}: t={}, speech: {}",
                    step_index,
                    t,
                    tensor_stats(&speech)
                );
            }

            if lower_order_nums < solver_order {
                lower_order_nums += 1;
            }
        }

        // Extract final result
        let final_output = speech.narrow(0, 0, batch)?;

        // Log final output for first token
        if log_first_token {
            info!(
                "🔍 [DIFFUSION] final_output: {}",
                tensor_stats(&final_output)
            );
        }

        Ok(final_output)
    }

    /// Get the model configuration.
    pub fn config(&self) -> &RealtimeConfig {
        &self.config
    }

    /// Get the number of diffusion steps.
    pub fn num_diffusion_steps(&self) -> usize {
        self.solver.num_steps
    }

    /// Set the number of diffusion steps.
    pub fn set_diffusion_steps(&mut self, num_steps: usize) {
        self.solver.num_steps = num_steps;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigma_schedule() {
        let sigmas = compute_sigmas_schedule(1000);
        assert_eq!(sigmas.len(), 1000);
        assert!(sigmas[0] > 0.0);
        assert!(sigmas[999] > sigmas[0]); // Sigmas increase with timestep
    }
}
