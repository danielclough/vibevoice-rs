use anyhow::{Error as AnyErr, Result};
use candle_core::{D, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::qwen2::Model as Qwen2Model;
use std::path::PathBuf;
use tokenizers::Tokenizer;
use tracing::{debug, info, warn};

use crate::{
    acoustic_connector::AcousticConnector,
    config::{VAEDecoderConfig, VibeVoiceConfig},
    diffusion::{DPMSolverPP, DiffusionHead},
    semantic_tokenizer::SemanticTokenizer,
    speech_connector::SpeechConnector,
    utils::seeded_randn,
    vae_decoder::VAEDecoder,
    vae_encoder::VAEEncoder,
};

pub struct VibeVoiceModel {
    pub device: Device,
    config: VibeVoiceConfig,
    llm: Qwen2Model,
    lm_head: Linear,
    embed_tokens: Tensor, // Token embeddings (separate from lm_head for 7B)
    diffusion_head: DiffusionHead,
    // Public for injection testing - allows isolated component validation
    pub acoustic_connector: AcousticConnector,
    pub semantic_connector: SpeechConnector,
    pub semantic_tokenizer: SemanticTokenizer,
    vae_encoder: VAEEncoder,
    pub vae_decoder: VAEDecoder,
    tokenizer: Tokenizer,
    solver: DPMSolverPP,
    pub speech_scaling_factor: Tensor,
    pub speech_bias_factor: Tensor,
    cfg_scale: f32,
    // Special token IDs loaded from tokenizer
    speech_start_id: u32,
    speech_end_id: u32,
    speech_diffusion_id: u32,
    bos_token_id: Option<u32>,
    eos_token_id: u32,
    // Pre-computed sigma schedule matching Python's betas_for_alpha_bar approach
    // This is critical for DPM-Solver++ numerical parity
    precomputed_sigmas: Vec<f32>,
    // Debug: pre-loaded diffusion noise for parity testing
    debug_diffusion_noise: Option<Tensor>,
}
/// Compute sigma schedule matching Python's betas_for_alpha_bar approach.
///
/// This is critical for DPM-Solver++ numerical parity with Python.
/// The key difference from direct alpha_bar evaluation:
/// - Python: beta[i] = 1 - alpha_bar(t₂)/alpha_bar(t₁), then alphas_cumprod = cumprod(1 - betas)
/// - Direct: alpha_bar(t) directly evaluated at each timestep
///
/// The cumulative product leads to much smaller values at high timesteps,
/// resulting in much larger sigmas (e.g., sigma[999] ≈ 20291 vs 641).
fn compute_sigmas_schedule(num_train_timesteps: usize) -> Vec<f32> {
    use std::f64::consts::PI;

    // Step 1: Define alpha_bar function (cosine schedule)
    let alpha_bar_fn = |t: f64| -> f64 { ((t + 0.008) / 1.008 * PI / 2.0).cos().powi(2) };

    // Step 2: Compute betas from consecutive alpha_bar ratios
    // Python: betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    let max_beta: f64 = 0.999;
    let mut betas = Vec::with_capacity(num_train_timesteps);
    for i in 0..num_train_timesteps {
        let t1 = i as f64 / num_train_timesteps as f64;
        let t2 = (i + 1) as f64 / num_train_timesteps as f64;
        let beta = (1.0 - alpha_bar_fn(t2) / alpha_bar_fn(t1)).min(max_beta);
        betas.push(beta);
    }

    // Step 3: Compute alphas = 1 - betas
    let alphas: Vec<f64> = betas.iter().map(|b| 1.0 - b).collect();

    // Step 4: Compute alphas_cumprod = cumulative product of alphas
    let mut alphas_cumprod = Vec::with_capacity(num_train_timesteps);
    let mut running_product = 1.0f64;
    for alpha in alphas.iter() {
        running_product *= alpha;
        alphas_cumprod.push(running_product);
    }

    // Step 5: Compute sigmas = sqrt((1 - alphas_cumprod) / alphas_cumprod)
    let sigmas: Vec<f32> = alphas_cumprod
        .iter()
        .map(|&acp| {
            let sigma = ((1.0 - acp) / acp.max(1e-20)).sqrt();
            sigma as f32
        })
        .collect();

    sigmas
}

impl VibeVoiceModel {
    pub fn new(
        vb: VarBuilder,
        device: Device,
        config_path: &PathBuf,
        tokenizer_path: &PathBuf,
    ) -> Result<Self> {
        // Load config - it will deserialize the nested structure
        let config = VibeVoiceConfig::from_file(config_path)?;
        info!(
            "📋 Model Configuration: {} ({} layers, hidden={})",
            config.variant_name(),
            config.llm_config.num_hidden_layers,
            config.llm_config.hidden_size
        );
        debug!(
            "   rms_norm_eps: {}, rope_theta: {}, intermediate_size: {}",
            config.llm_config.rms_norm_eps,
            config.llm_config.rope_theta,
            config.llm_config.intermediate_size
        );

        // Initialize tokenizer
        // After loading tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Verify special tokens exist with correct IDs
        let vision_start = tokenizer.token_to_id("<|vision_start|>");
        let vision_end = tokenizer.token_to_id("<|vision_end|>");
        let vision_pad = tokenizer.token_to_id("<|vision_pad|>");

        debug!(
            "🔍 Runtime token verification: start={:?}, end={:?}, pad={:?}",
            vision_start, vision_end, vision_pad
        );

        // CRITICAL: Check if they match expected IDs
        if vision_start != Some(151652) {
            return Err(anyhow::anyhow!(
                "Token <|vision_start|> has wrong ID: {:?}, expected 151652",
                vision_start
            ));
        }
        if vision_end != Some(151653) {
            return Err(anyhow::anyhow!(
                "Token <|vision_end|> has wrong ID: {:?}, expected 151653",
                vision_end
            ));
        }
        if vision_pad != Some(151654) {
            return Err(anyhow::anyhow!(
                "Token <|vision_pad|> has wrong ID: {:?}, expected 151654",
                vision_pad
            ));
        }

        debug!("   ✅ All tokens verified with correct IDs!");

        // Initialize dual LLM instances for CFG dual-pass architecture
        // Both models share weights (via vb.clone()) but maintain independent KV caches
        let qwen2_config = config.to_qwen2_config();
        info!("  🔄 Initializing dual-pass LLM architecture:");
        info!("     - LLM model (for CFG unconditional context)");
        let llm = Qwen2Model::new(&qwen2_config, vb.clone())?;
        let lm_head = match config.variant_name() {
            "VibeVoice-1.5B" => {
                // 1.5B uses tied embeddings - lm_head shares weights with embed_tokens
                info!("  ✓ Using tied embeddings (model.embed_tokens) for lm_head");
                let embed_weight = vb.pp("model.embed_tokens").get(
                    &[config.llm_config.vocab_size, config.llm_config.hidden_size],
                    "weight",
                )?;
                candle_nn::Linear::new(embed_weight, None)
            }
            "VibeVoice-7B" => {
                info!("  ✓ Loading separate lm_head for 7B model");
                candle_nn::linear_no_bias(
                    config.llm_config.hidden_size,
                    config.llm_config.vocab_size,
                    vb.pp("lm_head"),
                )?
            }
            _ => return Err(AnyErr::msg("Unsupported model variant")),
        };

        // Load embed_tokens separately (always from model.embed_tokens, works for both variants)
        // This is the correct weight for token embeddings - lm_head is only for output projection
        let embed_tokens = vb.pp("model.embed_tokens").get(
            &[config.llm_config.vocab_size, config.llm_config.hidden_size],
            "weight",
        )?;
        info!("  ✓ Loaded embed_tokens: {:?}", embed_tokens.dims());

        // Pass full config to components
        let diffusion_head = DiffusionHead::new(vb.pp("model.prediction_head"), &config)?;
        let acoustic_connector = AcousticConnector::new(vb.pp("acoustic_connector"), &config)?;
        let semantic_connector =
            SpeechConnector::new_semantic(vb.pp("semantic_connector"), &config)?;
        let semantic_tokenizer =
            SemanticTokenizer::new(vb.pp("semantic_tokenizer.encoder"), &config, device.clone())?;
        let vae_encoder = VAEEncoder::new(
            vb.pp("acoustic_tokenizer.encoder"),
            &config.acoustic_tokenizer_config,
        )?;
        let vae_decoder = VAEDecoder::new(
            vb.pp("acoustic_tokenizer.decoder"),
            VAEDecoderConfig::from_acoustic_config(&config.acoustic_tokenizer_config),
        )?;
        // Use inference steps from config
        let solver = DPMSolverPP::new(config.diffusion_head_config.ddpm_num_inference_steps);

        // Pre-compute sigma schedule matching Python's betas_for_alpha_bar approach
        // This is critical for DPM-Solver++ numerical parity
        let num_train_timesteps = config.diffusion_head_config.ddpm_num_steps;
        let precomputed_sigmas = compute_sigmas_schedule(num_train_timesteps);
        debug!(
            "📊 Pre-computed sigmas schedule ({} values)",
            precomputed_sigmas.len()
        );
        debug!(
            "   sigma[0]={:.6}, sigma[999]={:.2}, sigma[100]={:.6}",
            precomputed_sigmas[0], precomputed_sigmas[999], precomputed_sigmas[100]
        );

        // Load speech normalization factors - they may be stored as [1] or [] shapes
        let speech_scaling_tensor = vb
            .get(&[1], "model.speech_scaling_factor")
            .or_else(|_| vb.get(&[], "model.speech_scaling_factor"))?;
        let speech_bias_tensor = vb
            .get(&[1], "model.speech_bias_factor")
            .or_else(|_| vb.get(&[], "model.speech_bias_factor"))?;

        // Extract scalar values - handle both [1] and [] shapes
        let scale_val = if speech_scaling_tensor.dims().is_empty() {
            speech_scaling_tensor.to_scalar::<f32>()?
        } else {
            speech_scaling_tensor.to_vec1::<f32>()?[0]
        };
        let bias_val = if speech_bias_tensor.dims().is_empty() {
            speech_bias_tensor.to_scalar::<f32>()?
        } else {
            speech_bias_tensor.to_vec1::<f32>()?[0]
        };

        // Create scalar tensors (rank 0) for later use
        let speech_scaling_factor = Tensor::from_vec(vec![scale_val], &[], &device)?;
        let speech_bias_factor = Tensor::from_vec(vec![bias_val], &[], &device)?;

        debug!(
            "📊 Speech normalization: scaling={:.6}, bias={:.6}",
            scale_val, bias_val
        );

        // Default CFG scale - Python implementation uses 1.3
        let cfg_scale = 1.3;
        debug!("📊 CFG scale: {:.1}", cfg_scale);

        // Load special token IDs from tokenizer (dynamic lookup like Python)
        let speech_start_id = tokenizer
            .token_to_id("<|vision_start|>")
            .ok_or_else(|| AnyErr::msg("Token <|vision_start|> not found in tokenizer"))?;
        let speech_end_id = tokenizer
            .token_to_id("<|vision_end|>")
            .ok_or_else(|| AnyErr::msg("Token <|vision_end|> not found in tokenizer"))?;
        let speech_diffusion_id = tokenizer
            .token_to_id("<|vision_pad|>")
            .ok_or_else(|| AnyErr::msg("Token <|vision_pad|> not found in tokenizer"))?;
        let eos_token_id = tokenizer
            .token_to_id("<|endoftext|>")
            .ok_or_else(|| AnyErr::msg("Token <|endoftext|> not found in tokenizer"))?;
        let bos_token_id = tokenizer.token_to_id("<|endoftext|>"); // Use eos as fallback

        debug!(
            "🔑 Special token IDs: start={}, end={}, diffusion={}, eos={}",
            speech_start_id, speech_end_id, speech_diffusion_id, eos_token_id
        );

        // Verify token IDs are valid
        if speech_start_id == 0
            || speech_end_id == 0
            || speech_diffusion_id == 0
            || eos_token_id == 0
        {
            warn!("⚠️  Warning: Some special token IDs are 0, which may indicate missing tokens");
        }

        Ok(Self {
            device,
            config,
            llm,
            lm_head,
            embed_tokens,
            diffusion_head,
            acoustic_connector,
            semantic_connector,
            semantic_tokenizer,
            vae_encoder,
            vae_decoder,
            tokenizer,
            solver,
            speech_scaling_factor,
            speech_bias_factor,
            cfg_scale,
            speech_start_id,
            speech_end_id,
            speech_diffusion_id,
            eos_token_id,
            bos_token_id,
            precomputed_sigmas,
            debug_diffusion_noise: None,
        })
    }

    /// Load debug diffusion noise from a Python-exported .npz file
    /// This allows testing the DPM solver with identical inputs as Python
    pub fn load_debug_diffusion_noise(&mut self, npz_path: &std::path::Path) -> Result<()> {
        use crate::test_helpers::{Checkpoint, ToTensor};

        info!("🔧 Loading debug diffusion noise from {:?}", npz_path);
        let mut checkpoint = Checkpoint::open(npz_path)?;

        // Load 2D noise array [2, 64] and convert to tensor
        let noise_array = checkpoint.load_array2("initial_noise")?;
        let noise = noise_array.to_tensor(&self.device)?;

        info!("   Noise shape: {:?}", noise.dims());
        let flat: Vec<f32> = noise.flatten_all()?.to_vec1()?;
        info!("   First 10 values: {:?}", &flat[..10.min(flat.len())]);
        self.debug_diffusion_noise = Some(noise);
        info!("✓ Debug diffusion noise loaded");

        Ok(())
    }

    /// Set the classifier-free guidance scale
    pub fn set_cfg_scale(&mut self, cfg_scale: f32) {
        self.cfg_scale = cfg_scale;
    }

    pub fn set_ddpm_inference_steps(&mut self, num_steps: usize) {
        self.solver.num_steps = num_steps;
    }

    /// Test-only: Run diffusion sampler with provided conditions
    /// This allows direct testing of the diffusion process in isolation
    /// condition: [batch, hidden_size] - positive LLM hidden state
    /// neg_condition: [batch, hidden_size] - negative/CFG hidden state
    /// Returns: diffusion_latent [batch, 64]
    pub fn sample_diffusion(&self, condition: &Tensor, neg_condition: &Tensor) -> Result<Tensor> {
        self.sample_dpm_solver(
            condition,
            neg_condition,
            self.solver.num_steps,
            self.cfg_scale,
        )
    }

    /// Encode audio to latent representation using VAE encoder
    /// Input: audio [batch, 1, samples] at 24kHz
    /// Output: latents [batch, 64, vae_tokens]
    pub fn encode_voice(&self, audio: &Tensor) -> Result<Tensor> {
        self.vae_encoder.encode(audio)
    }

    /// Test-only: Get reference to diffusion head for step-by-step testing
    pub fn get_diffusion_head(&self) -> &crate::diffusion::DiffusionHead {
        &self.diffusion_head
    }

    /// Get token embeddings from input_ids
    /// This accesses the dedicated embed_tokens weight matrix
    fn get_token_embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Use dedicated embed_tokens (correct for both 1.5B and 7B)
        // Note: lm_head is for output projection, embed_tokens is for input embeddings
        let embed_weight = &self.embed_tokens;

        // Manual embedding lookup: index into weight matrix
        // embed_weight: [vocab_size, hidden_size]
        // input_ids: [batch, seq_len]
        // Result: [batch, seq_len, hidden_size]

        let input_ids_vec = input_ids.to_vec2::<u32>()?;
        let (vocab_size, hidden_size) = embed_weight.dims2()?;
        let batch_size = input_ids_vec.len();
        let seq_len = input_ids_vec[0].len();

        // Convert weight to vec for indexing
        let weight_vec = embed_weight.to_vec2::<f32>()?;

        // Build embeddings by indexing
        let mut embeds = Vec::with_capacity(batch_size * seq_len * hidden_size);
        for batch in &input_ids_vec {
            for &token_id in batch {
                let idx = token_id as usize;
                if idx >= vocab_size {
                    return Err(AnyErr::msg(format!(
                        "Token ID {} out of range (vocab size: {})",
                        idx, vocab_size
                    )));
                }
                embeds.extend_from_slice(&weight_vec[idx]);
            }
        }

        Ok(Tensor::from_vec(
            embeds,
            (batch_size, seq_len, hidden_size),
            embed_weight.device(),
        )?)
    }

    /// Convert voice audio to embeddings ready for injection into LLM
    ///
    /// Input:
    ///   - audio: [num_speakers, 1, samples] at 24kHz (padded to max_samples)
    ///   - speech_masks: Optional per-speaker validity mask [num_speakers][max_vae_tokens]
    ///                   True = valid token, False = padding
    ///
    /// Output:
    ///   - If speech_masks provided: [total_valid_tokens, hidden_size] - flattened valid embeddings
    ///   - If no speech_masks: [num_tokens, hidden_size] - squeezed single speaker embeddings
    ///
    /// This matches Python's behavior where speech_masks is used to filter padded tokens
    /// and flatten all valid tokens across speakers into a 1D tensor for injection.
    pub fn process_voice_for_injection(
        &self,
        audio: &Tensor,
        speech_masks: Option<&Vec<Vec<bool>>>,
    ) -> Result<Tensor> {
        debug!(
            "🎤 Processing voice for embedding injection, input shape: {:?}",
            audio.dims()
        );

        // Step 1: Encode audio to VAE latents [num_speakers, 64, vae_tokens]
        let latents = self.vae_encoder.encode(audio)?;
        debug!("  VAE latents shape: {:?}", latents.dims());

        // Step 2: Permute to [num_speakers, vae_tokens, 64] (matching Python encoder output)
        // This must happen BEFORE sampling and bias/scale to match Python's flow
        let latents_permuted = latents.transpose(1, 2)?;
        debug!("  Permuted shape: {:?}", latents_permuted.dims());

        // Step 3: Gaussian sampling (matching Python's std_dist_type='gaussian')
        // Python: value = fix_std / 0.8, std_per_batch = randn(batch) * value
        // sampled = mean + std_per_batch.unsqueeze(-1).unsqueeze(-1) * randn_like(mean)
        // This adds per-speaker variation which is critical for multi-speaker voice quality
        let batch_size = latents_permuted.dim(0)?;
        let fix_std = 0.5_f64; // From config.acoustic_tokenizer_config.fix_std
        let value = fix_std / 0.8; // = 0.625

        // Generate different random scale per speaker: randn(batch_size) * value
        // CRITICAL: Must match Python exactly: torch.randn(batch_size) * value
        // NOT seeded_randn(0, value, ...) which generates N(0, value) directly!
        // randn() * value produces different values than N(0, value) even with same seed.
        let std_per_batch_base = seeded_randn(0.0, 1.0, &[batch_size], audio.device())?;
        let std_per_batch = (std_per_batch_base * value)?;
        debug!(
            "  Gaussian sampling: std_per_batch shape {:?}",
            std_per_batch.dims()
        );

        // Expand to [batch, 1, 1] for broadcasting with [batch, time, dim]
        let std_expanded = std_per_batch.unsqueeze(1)?.unsqueeze(2)?;

        // Generate noise same shape as latents
        let noise = seeded_randn(0.0, 1.0, latents_permuted.dims(), audio.device())?;

        // sampled = mean + std_expanded * noise
        let scaled_noise = std_expanded.broadcast_mul(&noise)?;
        let latents_sampled = latents_permuted.broadcast_add(&scaled_noise)?;
        debug!("  Sampled latents shape: {:?}", latents_sampled.dims());

        // Step 4: Apply normalization (same as Python: (latents + bias) * scale)
        let scale_val = self.speech_scaling_factor.to_scalar::<f32>()?;
        let bias_val = self.speech_bias_factor.to_scalar::<f32>()?;
        let normalized = ((latents_sampled + bias_val as f64)? * scale_val as f64)?;
        debug!("  Normalized latents shape: {:?}", normalized.dims());

        // Step 5: Project through acoustic connector to LLM hidden space
        // voice_embeds: [num_speakers, max_vae_tokens, hidden_size]
        let voice_embeds = self.acoustic_connector.forward(&normalized)?;
        debug!("  Voice embeddings shape: {:?}", voice_embeds.dims());

        // Step 6: Apply speech_masks to filter padded tokens and flatten
        // This matches Python's: acoustic_connected[speech_masks.cpu()]
        if let Some(masks) = speech_masks {
            let voice_vec = voice_embeds.to_vec3::<f32>()?;
            let hidden_size = voice_embeds.dim(2)?;

            // Flatten and filter: collect only valid tokens from all speakers
            let mut flat_embeds: Vec<f32> = Vec::new();
            let mut total_valid_tokens = 0;

            for (speaker_idx, mask) in masks.iter().enumerate() {
                for (tok_idx, &is_valid) in mask.iter().enumerate() {
                    if is_valid && tok_idx < voice_vec[speaker_idx].len() {
                        flat_embeds.extend_from_slice(&voice_vec[speaker_idx][tok_idx]);
                        total_valid_tokens += 1;
                    }
                }
            }

            debug!(
                "  Filtered with speech_masks: {} valid tokens from {} speakers -> [{}, {}]",
                total_valid_tokens,
                masks.len(),
                total_valid_tokens,
                hidden_size
            );

            // Return flattened 2D tensor [total_valid_tokens, hidden_size]
            Ok(Tensor::from_vec(
                flat_embeds,
                (total_valid_tokens, hidden_size),
                audio.device(),
            )?)
        } else {
            // Single speaker case: squeeze batch dim to get [num_tokens, hidden_size]
            let squeezed = voice_embeds.squeeze(0)?;
            debug!("  Single speaker, squeezed to: {:?}", squeezed.dims());
            Ok(squeezed)
        }
    }

    /// Inject voice embeddings into token embeddings at specified positions
    ///
    /// Input formats:
    ///   - base_embeds: [batch, seq_len, hidden_size] - token embeddings (batch is typically 1)
    ///   - voice_embeds: [total_valid_tokens, hidden_size] - flattened voice embeddings (from multi-speaker)
    ///                   OR [num_tokens, hidden_size] - single speaker embeddings
    ///   - injection_mask: [batch, seq_len] or [seq_len] - boolean mask marking positions to inject
    ///
    /// Output: [batch, seq_len, hidden_size] - embeddings with voice injected
    ///
    /// This matches Python's behavior: inputs_embeds[speech_input_mask] = speech_embeds
    /// where speech_embeds is a flat 1D tensor of all valid voice embeddings
    pub fn inject_voice_embeddings(
        &self,
        base_embeds: &Tensor,
        voice_embeds: &Tensor,
        injection_mask: &Tensor,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = base_embeds.dims3()?;

        debug!(
            "💉 Injecting voice embeddings: base={:?}, voice={:?}, mask={:?}",
            base_embeds.dims(),
            voice_embeds.dims(),
            injection_mask.dims()
        );

        // Squeeze the mask if it has batch dimension: [1, seq_len] -> [seq_len]
        let mask_1d = if injection_mask.dims().len() == 2 {
            injection_mask.squeeze(0)?
        } else {
            injection_mask.clone()
        };

        // Convert to CPU for easier manipulation
        // base_embeds: [batch, seq_len, hidden_size]
        let base_vec = base_embeds.to_vec3::<f32>()?;
        let mask_vec = mask_1d.to_vec1::<u8>()?;

        // Handle both 2D [total_tokens, hidden] and 3D [batch, tokens, hidden] voice_embeds
        // After our fix, voice_embeds should always be 2D [total_valid_tokens, hidden_size]
        let voice_vec_2d: Vec<Vec<f32>> = if voice_embeds.dims().len() == 2 {
            voice_embeds.to_vec2::<f32>()?
        } else {
            // Legacy 3D format: flatten to 2D by taking first batch and all tokens
            // This handles backward compatibility with old code paths
            let vec_3d = voice_embeds.to_vec3::<f32>()?;
            vec_3d[0].clone()
        };

        // Create new embeddings array (clone base)
        let mut result = base_vec.clone();

        // Inject voice embeddings at marked positions
        // Simple 1-to-1 assignment: for each marked position, take next embedding from voice_vec_2d
        let mut voice_idx = 0;
        let total_voice_tokens = voice_vec_2d.len();
        let marked_positions: usize = mask_vec.iter().map(|&v| v as usize).sum();

        for pos in 0..seq_len {
            if mask_vec[pos] == 1 && voice_idx < total_voice_tokens {
                // Inject the same voice embedding for all batch items (typically batch=1)
                for b in 0..batch_size {
                    result[b][pos] = voice_vec_2d[voice_idx].clone();
                }
                voice_idx += 1;
            }
        }

        debug!(
            "  Injected {} voice tokens into {} marked positions",
            voice_idx, marked_positions
        );

        if voice_idx != marked_positions {
            tracing::warn!(
                "⚠️ Voice token count mismatch: {} voice tokens vs {} marked positions",
                total_voice_tokens,
                marked_positions
            );
        }

        // Flatten and convert back to tensor
        let flat: Vec<f32> = result
            .into_iter()
            .flat_map(|batch| batch.into_iter().flat_map(|seq| seq))
            .collect();

        let injected = Tensor::from_vec(
            flat,
            (batch_size, seq_len, hidden_size),
            base_embeds.device(),
        )?;

        Ok(injected)
    }

    pub fn tokenize(&self, text: &str) -> Result<Tensor> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| AnyErr::msg(format!("Tokenization error: {}", e)))?;
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        Ok(Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?)
    }
    /// DPM-Solver++ sampling matching Python's DPMSolverMultistepScheduler
    /// Reference: VibeVoice/vibevoice/schedule/dpm_solver.py
    /// Key matching points:
    /// 1. Timesteps: np.linspace(0, last_timestep-1, num_steps+1).round()[::-1][:-1]
    /// 2. Solver order: 2nd-order for steps 2+ (multistep with midpoint)
    /// 3. Noise shape: [2*batch, latent] to match Python (only first half used)
    fn sample_dpm_solver(
        &self,
        condition: &Tensor,
        neg_condition: &Tensor,
        num_steps: usize,
        cfg_scale: f32,
    ) -> Result<Tensor> {
        let batch = condition.dim(0)?;
        let latent_size = self.config.acoustic_vae_dim;
        let num_train_timesteps = self.config.diffusion_head_config.ddpm_num_steps;

        debug!("\n🔍 ========== DPM-Solver++ Sampling (Python-matched) ==========");
        debug!("🔍 Batch: {}, Latent size: {}", batch, latent_size);
        debug!(
            "🔍 Num steps: {}, Train timesteps: {}",
            num_steps, num_train_timesteps
        );

        // === TIMESTEP CALCULATION (matching Python exactly) ===
        // Python: np.linspace(0, last_timestep - 1, num_inference_steps + 1).round()[::-1][:-1]
        // The [:-1] drops the LAST element after reverse (which is 0), not the first (999)
        let last_timestep = num_train_timesteps;
        let n_points = num_steps + 1;
        let timesteps: Vec<i64> = (0..n_points)
            .map(|i| {
                let t = (i as f64 / (n_points - 1) as f64) * (last_timestep - 1) as f64;
                t.round() as i64
            })
            .rev() // reverse: [999, 899, ..., 100, 0]
            .take(num_steps) // take first num_steps elements: [999, 899, ..., 100] (drops 0)
            .collect();
        debug!("🔍 Timesteps (Python-matched): {:?}", timesteps);

        // === SIGMA CALCULATION (using pre-computed schedule matching Python) ===
        // Python uses: sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        // where sigmas is pre-computed from betas_for_alpha_bar with cosine schedule
        let mut sigmas: Vec<f32> = timesteps
            .iter()
            .map(|&t| {
                // Interpolate from pre-computed sigmas array (matching Python's np.interp)
                let t_idx = t as usize;
                if t_idx >= self.precomputed_sigmas.len() {
                    *self.precomputed_sigmas.last().unwrap()
                } else {
                    // For integer timesteps, direct lookup (no interpolation needed)
                    self.precomputed_sigmas[t_idx]
                }
            })
            .collect();
        sigmas.push(0.0); // Final sigma is 0 (matching Python's final_sigmas_type="zero")
        debug!(
            "🔍 Sigmas (pre-computed): {:?}",
            &sigmas[..sigmas.len().min(6)]
        );

        // Helper: convert sigma to (alpha_t, sigma_t) matching Python's _sigma_to_alpha_sigma_t
        let sigma_to_alpha_sigma = |sigma: f32| -> (f32, f32) {
            let alpha_t = 1.0 / (sigma.powi(2) + 1.0).sqrt();
            let sigma_t = sigma * alpha_t;
            (alpha_t, sigma_t)
        };

        // === INITIAL NOISE (matching Python's [2*batch, latent] shape) ===
        // Python: speech = torch.randn(condition.shape[0], self.config.acoustic_vae_dim)
        // where condition.shape[0] = 2*batch (because [cond, neg_cond] are concatenated)
        // Only the first half is actually used, but we generate the same amount for RNG consistency
        let doubled_batch = 2 * batch;
        let mut speech = if let Some(ref debug_noise) = self.debug_diffusion_noise {
            // Use pre-loaded debug noise for parity testing
            info!("🔧 Using debug diffusion noise (Python-exported)");
            debug_noise.clone()
        } else {
            seeded_randn(0.0, 1.0, &[doubled_batch, latent_size], &self.device)?
        };

        let init_vec = speech.flatten_all()?.to_vec1::<f32>()?;
        let initial_rms = (init_vec.iter().map(|v| v.powi(2)).sum::<f32>() / init_vec.len() as f32).sqrt();
        let cond_vec = condition.flatten_all()?.to_vec1::<f32>()?;
        let cond_rms = (cond_vec.iter().map(|v| v.powi(2)).sum::<f32>() / cond_vec.len() as f32).sqrt();
        debug!(
            "🔍 Initial noise (doubled): shape=[{}, {}], mean={:.3}, std={:.3}",
            doubled_batch,
            latent_size,
            init_vec.iter().sum::<f32>() / init_vec.len() as f32,
            initial_rms
        );

        // Diffusion debugging: log initial state (matching Python format)
        info!("[DIFF] cfg_scale={:.1}, steps={}", cfg_scale, num_steps);
        info!("[DIFF] Initial noise RMS: {:.6}, condition RMS: {:.6}", initial_rms, cond_rms);
        // DIAGNOSTIC: Log first 10 values for exact comparison with Python
        info!("[DIFF] First 10 noise: {:?}", &init_vec[..10.min(init_vec.len())]);
        info!("[DIFF] First 10 cond: {:?}", &cond_vec[..10.min(cond_vec.len())]);

        // === MULTISTEP SOLVER STATE ===
        // Python tracks model_outputs for 2nd-order solver
        let solver_order = 2usize;
        let mut model_outputs: Vec<Option<Tensor>> = vec![None; solver_order];
        let mut lower_order_nums = 0usize;

        // Concatenate conditions once (used for all steps)
        let conditions = Tensor::cat(&[condition, neg_condition], 0)?;

        debug!("\n🔍 Starting denoising loop (2nd-order DPM-Solver++)...");

        for step_index in 0..num_steps {
            let t = timesteps[step_index];
            let sigma_s = sigmas[step_index];
            let sigma_t = sigmas[step_index + 1];

            let (alpha_s, sigma_s_actual) = sigma_to_alpha_sigma(sigma_s);
            let (alpha_t, sigma_t_actual) = sigma_to_alpha_sigma(sigma_t);

            debug!("\n  📍 Step {}/{}: t={}", step_index + 1, num_steps, t);
            debug!("     sigma_s={:.6}, sigma_t={:.6}", sigma_s, sigma_t);

            // === FORWARD PASS (matching Python's half-duplication pattern) ===
            // Python: half = speech[: len(speech) // 2]
            //         combined = torch.cat([half, half], dim=0)
            let half = speech.narrow(0, 0, batch)?;
            let combined = Tensor::cat(&[&half, &half], 0)?;

            let t_tensor = Tensor::new(&[t as f32], &self.device)?;
            let t_batch = t_tensor.broadcast_as((doubled_batch,))?;

            // Forward through diffusion head
            let model_output = self
                .diffusion_head
                .forward(&combined, &t_batch, &conditions)?;

            // Split and apply CFG
            // Python: cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            //         half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            let output_chunks: Vec<Tensor> = model_output.chunk(2, 0)?;
            let cond_output = &output_chunks[0];
            let uncond_output = &output_chunks[1];

            let diff = (cond_output - uncond_output)?;
            let half_output = (uncond_output + diff.affine(cfg_scale as f64, 0.0)?)?;

            // === CONVERT MODEL OUTPUT TO x0_pred ===
            // For v_prediction: x0_pred = alpha_t * sample - sigma_t * model_output
            // But we need to use the current sigma (sigma_s), not next sigma
            let x0_pred = (half.affine(alpha_s as f64, 0.0)?
                - half_output.affine(sigma_s_actual as f64, 0.0)?)?;

            // === DETERMINE SOLVER ORDER FOR THIS STEP ===
            let lower_order_final =
                step_index == num_steps - 1 && (num_steps < 15 || sigma_t == 0.0);
            let lower_order_second = step_index == num_steps - 2 && num_steps < 15;

            let use_first_order = solver_order == 1 || lower_order_nums < 1 || lower_order_final;
            let use_second_order = !use_first_order
                && (solver_order == 2 || lower_order_nums < 2 || lower_order_second);

            // Shift model outputs for multistep
            for i in 0..(solver_order - 1) {
                model_outputs[i] = model_outputs[i + 1].take();
            }
            model_outputs[solver_order - 1] = Some(x0_pred.clone());

            // === SOLVER STEP ===
            let lambda_s = alpha_s.ln() - sigma_s_actual.ln();
            let lambda_t = alpha_t.ln() - sigma_t_actual.ln();
            let h = lambda_t - lambda_s;

            let prev_sample = if use_first_order {
                debug!("     Using 1st-order solver");
                // First-order DPM-Solver++ update
                // x_t = (sigma_t / sigma_s) * sample - alpha_t * (exp(-h) - 1) * x0_pred
                let coeff1 = sigma_t_actual / sigma_s_actual;
                let coeff2 = alpha_t * ((-h).exp() - 1.0);

                // Apply to BOTH halves of speech (matching Python's scheduler.step on full tensor)
                let term1 = speech.affine(coeff1 as f64, 0.0)?;
                // For the x0_pred term, we need to apply it to both halves
                let x0_full = Tensor::cat(&[&x0_pred, &x0_pred], 0)?;
                let term2 = x0_full.affine(coeff2 as f64, 0.0)?;
                (term1 - term2)?
            } else if use_second_order {
                debug!("     Using 2nd-order solver (midpoint)");
                // Second-order DPM-Solver++ with midpoint
                // Requires model_outputs[-1] and model_outputs[-2]
                let m0 = model_outputs[solver_order - 1].as_ref().unwrap();
                let m1 = model_outputs[solver_order - 2].as_ref();

                if let Some(m1) = m1 {
                    // Get previous sigmas for lambda calculation
                    let sigma_s1 = sigmas[step_index - 1];
                    let (alpha_s1, sigma_s1_actual) = sigma_to_alpha_sigma(sigma_s1);
                    let lambda_s1 = alpha_s1.ln() - sigma_s1_actual.ln();

                    let h_0 = lambda_s - lambda_s1;
                    let r0 = h_0 / h;

                    // D0 = m0, D1 = (1/r0) * (m0 - m1)
                    let d0 = m0;
                    let d1 = ((m0 - m1)? * (1.0 / r0 as f64))?;

                    // Midpoint formula:
                    // x_t = (sigma_t/sigma_s) * sample
                    //     - alpha_t * (exp(-h) - 1) * D0
                    //     - 0.5 * alpha_t * (exp(-h) - 1) * D1
                    let coeff1 = sigma_t_actual / sigma_s_actual;
                    let coeff2 = alpha_t * ((-h).exp() - 1.0);
                    let coeff3 = 0.5 * coeff2;

                    let term1 = speech.affine(coeff1 as f64, 0.0)?;
                    let d0_full = Tensor::cat(&[d0, d0], 0)?;
                    let d1_full = Tensor::cat(&[&d1, &d1], 0)?;
                    let term2 = d0_full.affine(coeff2 as f64, 0.0)?;
                    let term3 = d1_full.affine(coeff3 as f64, 0.0)?;
                    ((term1 - term2)? - term3)?
                } else {
                    // Fallback to first-order if m1 not available
                    debug!("     Falling back to 1st-order (no m1)");
                    let coeff1 = sigma_t_actual / sigma_s_actual;
                    let coeff2 = alpha_t * ((-h).exp() - 1.0);
                    let term1 = speech.affine(coeff1 as f64, 0.0)?;
                    let x0_full = Tensor::cat(&[&x0_pred, &x0_pred], 0)?;
                    let term2 = x0_full.affine(coeff2 as f64, 0.0)?;
                    (term1 - term2)?
                }
            } else {
                // Fallback (shouldn't reach here with solver_order=2)
                debug!("     Using fallback 1st-order");
                let coeff1 = sigma_t_actual / sigma_s_actual;
                let coeff2 = alpha_t * ((-h).exp() - 1.0);
                let term1 = speech.affine(coeff1 as f64, 0.0)?;
                let x0_full = Tensor::cat(&[&x0_pred, &x0_pred], 0)?;
                let term2 = x0_full.affine(coeff2 as f64, 0.0)?;
                (term1 - term2)?
            };

            speech = prev_sample;

            if lower_order_nums < solver_order {
                lower_order_nums += 1;
            }

            let sample_vec = speech
                .narrow(0, 0, batch)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let sample_std = (sample_vec.iter().map(|v| v.powi(2)).sum::<f32>()
                / sample_vec.len() as f32)
                .sqrt();
            debug!("     Updated sample (first half) std={:.3}", sample_std);

            // Diffusion debugging: log each step (matching Python format)
            let eps_vec = model_output.flatten_all()?.to_vec1::<f32>()?;
            let eps_rms = (eps_vec.iter().map(|v| v.powi(2)).sum::<f32>() / eps_vec.len() as f32).sqrt();
            let half_output_vec = half_output.flatten_all()?.to_vec1::<f32>()?;
            let half_eps_rms = (half_output_vec.iter().map(|v| v.powi(2)).sum::<f32>() / half_output_vec.len() as f32).sqrt();
            let speech_rms = sample_std; // Already computed as sample_std
            info!("[DIFF Step {}] t={}, eps_rms={:.6}, half_eps_rms={:.6}, speech_rms={:.6}",
                step_index, t, eps_rms, half_eps_rms, speech_rms);
        }

        // Return only the first half (matching Python's return speech[: len(speech) // 2])
        let result = speech.narrow(0, 0, batch)?;

        let final_vec = result.flatten_all()?.to_vec1::<f32>()?;
        let final_mean = final_vec.iter().sum::<f32>() / final_vec.len() as f32;
        let final_std = (final_vec
            .iter()
            .map(|v| (v - final_mean).powi(2))
            .sum::<f32>()
            / final_vec.len() as f32)
            .sqrt();

        debug!("\n🔍 ========== Sampling Complete ==========");
        debug!(
            "🔍 Final output: mean={:.3}, std={:.3}",
            final_mean, final_std
        );
        debug!("🔍 ==========================================\n");

        // Diffusion debugging: log final output (matching Python format)
        let final_rms = (final_vec.iter().map(|v| v.powi(2)).sum::<f32>() / final_vec.len() as f32).sqrt();
        info!("[DIFF] Final output RMS: {:.6}", final_rms);

        Ok(result)
    }
    /// Get logits from hidden states, optionally only computing for last N tokens.
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor of shape [batch, seq_len, hidden_size]
    /// * `logits_to_keep` - Number of tokens to keep (from the end). 0 = keep all.
    ///
    /// # Memory Optimization
    /// During autoregressive generation with KV cache, we only need logits for the
    /// last token (the one being sampled). Computing logits for all tokens wastes
    /// memory: for 12,866 tokens × 151,936 vocab × 4 bytes = 7.3 GB!
    /// By setting logits_to_keep=1, we reduce this to just 0.6 MB.
    fn get_logits(
        &self,
        hidden_states: &Tensor,
        logits_to_keep: usize,
    ) -> Result<Tensor, candle_core::Error> {
        let hidden_to_project = if logits_to_keep == 0 {
            // Keep all tokens (backward compatibility)
            hidden_states.clone()
        } else {
            // Slice last N tokens: hidden_states[:, -N:, :]
            let seq_len = hidden_states.dim(1)?;
            if logits_to_keep >= seq_len {
                hidden_states.clone()
            } else {
                hidden_states.narrow(1, seq_len - logits_to_keep, logits_to_keep)?
            }
        };
        self.lm_head.forward(&hidden_to_project)
    }

    /// Apply token constraints to logits, masking invalid tokens
    /// Only allows valid speech tokens: SPEECH_START, SPEECH_END, SPEECH_DIFFUSION, EOS
    ///
    /// Token IDs are loaded dynamically from the tokenizer during initialization
    fn apply_token_constraints(&self, logits: &Tensor) -> Result<Tensor> {
        // logits shape: [batch, vocab_size]
        let vocab_size = logits.dim(D::Minus1)?;
        let batch_size = logits.dim(0)?;

        // For each item in batch, apply constraints
        let mut constrained_logits = Vec::new();

        for b in 0..batch_size {
            let batch_logits = logits.i((b, ..))?;
            let mut logits_vec = batch_logits.to_vec1::<f32>()?;

            // Mask all tokens except valid speech tokens
            for i in 0..vocab_size {
                let mut is_valid = i == self.speech_start_id as usize
                    || i == self.speech_end_id as usize
                    || i == self.speech_diffusion_id as usize
                    || i == self.eos_token_id as usize;

                // Include bos_token_id if it exists and is different from eos
                if let Some(bos_id) = self.bos_token_id {
                    is_valid = is_valid || i == bos_id as usize;
                }

                if !is_valid {
                    logits_vec[i] = f32::NEG_INFINITY;
                }
            }

            constrained_logits.push(logits_vec);
        }

        // Flatten and create new tensor
        let flat_logits: Vec<f32> = constrained_logits.into_iter().flatten().collect();
        let result = Tensor::from_vec(flat_logits, &[batch_size, vocab_size], logits.device())?;

        let valid_count = 4 + if self.bos_token_id.is_some() { 1 } else { 0 };
        debug!(
            "  🔒 Token constraints applied: masked {} invalid tokens, allowing {} valid speech tokens",
            vocab_size * batch_size - valid_count * batch_size,
            valid_count
        );

        Ok(result)
    }

    /// Sample next token from logits using various sampling strategies
    fn sample_next_token(&self, logits: &Tensor) -> Result<u32> {
        // Apply token constraints to mask invalid tokens
        let constrained_logits = self.apply_token_constraints(logits)?;

        // In sample_next_token, after apply_token_constraints:
        let constrained_vec = &constrained_logits.to_vec2::<f32>()?[0];
        let mut indexed: Vec<(usize, f32)> = constrained_vec
            .iter()
            .enumerate()
            .filter(|(_, v)| v > &&f32::NEG_INFINITY) // Only show valid tokens
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

        debug!("🔍 Top logits AFTER constraints (valid tokens only):");
        for (i, &(token_id, logit_val)) in indexed.iter().enumerate() {
            debug!("  {}. token {} = {:.4}", i + 1, token_id, logit_val);
        }

        // Apply sampling strategy
        let next_token = self.sample_greedy(&constrained_logits)?;

        Ok(next_token)
    }

    /// Greedy sampling (argmax)
    fn sample_greedy(&self, logits: &Tensor) -> Result<u32> {
        let argmax_result = logits.argmax(D::Minus1)?;
        let next_token = if argmax_result.dims().is_empty() {
            argmax_result.to_scalar::<u32>()?
        } else {
            argmax_result.to_vec1::<u32>()?[0]
        };
        Ok(next_token)
    }

    /// Generate speech with voice cloning support (accepts processed inputs)
    /// Supports both streaming and non-streaming modes based on callback presence
    ///
    /// # Arguments
    /// * `input_ids` - Processed input token IDs [batch, seq_len]
    /// * `voice_audio` - Optional voice sample [num_speakers, 1, samples] at 24kHz
    /// * `speech_input_mask` - Optional mask [batch, seq_len] marking positions for voice injection
    /// * `attention_mask` - Optional attention mask [batch, seq_len]
    /// * `speech_masks` - Optional per-speaker VAE token validity masks [num_speakers][max_vae_tokens]
    /// * `max_new_tokens` - Maximum tokens to generate
    /// * `is_prefill` - If true, use voice embeddings for cloning
    /// * `chunk_callback` - Optional callback for streaming audio chunks
    ///
    /// # Returns
    /// Vec<Tensor> containing all generated audio chunks
    pub fn generate_processed<F>(
        &mut self,
        input_ids: &Tensor,
        voice_audio: Option<&Tensor>,
        speech_input_mask: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        speech_masks: Option<&Vec<Vec<bool>>>,
        max_new_tokens: Option<usize>,
        is_prefill: bool,
        chunk_callback: Option<F>,
    ) -> Result<Vec<Tensor>>
    where
        F: FnMut(&Tensor) -> Result<()>,
    {
        // Process voice if provided and is_prefill is true
        let voice_embeds = if is_prefill && voice_audio.is_some() {
            Some(self.process_voice_for_injection(voice_audio.unwrap(), speech_masks)?)
        } else {
            None
        };

        // Generate tokens and audio chunks
        self.generate_autoregressive_streaming(
            input_ids,
            voice_embeds.as_ref(),
            speech_input_mask,
            attention_mask,
            max_new_tokens,
            chunk_callback,
        )
    }

    /// Generate speech with pre-computed voice embeddings (bypasses RNG)
    ///
    /// This method is useful for debugging/testing where you want to inject
    /// voice embeddings computed by Python to isolate whether differences
    /// are in embedding generation or downstream processing.
    ///
    /// # Arguments
    /// * `input_ids` - Processed input token IDs [batch, seq_len]
    /// * `voice_embeds` - Pre-computed voice embeddings [total_tokens, hidden_size]
    /// * `speech_input_mask` - Mask [batch, seq_len] marking positions for voice injection
    /// * `attention_mask` - Optional attention mask [batch, seq_len]
    /// * `max_new_tokens` - Maximum tokens to generate
    /// * `chunk_callback` - Optional callback for streaming audio chunks
    ///
    /// # Returns
    /// Vec<Tensor> containing all generated audio chunks
    pub fn generate_with_precomputed_embeds<F>(
        &mut self,
        input_ids: &Tensor,
        voice_embeds: &Tensor,
        speech_input_mask: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        max_new_tokens: Option<usize>,
        chunk_callback: Option<F>,
    ) -> Result<Vec<Tensor>>
    where
        F: FnMut(&Tensor) -> Result<()>,
    {
        debug!("🔧 Using pre-computed voice embeddings (bypassing RNG)");
        debug!("   voice_embeds shape: {:?}", voice_embeds.dims());

        self.generate_autoregressive_streaming(
            input_ids,
            Some(voice_embeds),
            speech_input_mask,
            attention_mask,
            max_new_tokens,
            chunk_callback,
        )
    }

    /// Generate speech with voice cloning support
    /// Supports both streaming and non-streaming modes based on callback presence
    ///
    /// NOTE: This is a simplified API that doesn't support multi-speaker with speech_masks.
    /// For multi-speaker synthesis, use generate_processed with ProcessorOutput.
    ///
    /// # Arguments
    /// * `text` - Input text to synthesize
    /// * `voice_audio` - Optional voice sample [batch, 1, samples] at 24kHz (single speaker)
    /// * `speech_input_mask` - Optional mask [batch, seq_len] marking positions for voice injection
    /// * `max_new_tokens` - Maximum tokens to generate
    /// * `is_prefill` - If true, use voice embeddings for cloning
    /// * `chunk_callback` - Optional callback for streaming audio chunks
    ///
    /// # Returns
    /// Vec<Tensor> containing all generated audio chunks
    pub fn generate<F>(
        &mut self,
        text: &str,
        voice_audio: Option<&Tensor>,
        speech_input_mask: Option<&Tensor>,
        max_new_tokens: Option<usize>,
        is_prefill: bool,
        chunk_callback: Option<F>,
    ) -> Result<Vec<Tensor>>
    where
        F: FnMut(&Tensor) -> Result<()>,
    {
        let is_streaming = chunk_callback.is_some();
        debug!(
            "🎵 {} Audio Generation",
            if is_streaming { "Streaming" } else { "Batch" }
        );

        // Process voice if provided and is_prefill is true
        // Note: speech_masks is None here - this API is for single-speaker only
        let voice_embeds = if is_prefill && voice_audio.is_some() {
            Some(self.process_voice_for_injection(voice_audio.unwrap(), None)?)
        } else {
            None
        };

        // Tokenize the input text
        let input_ids = self.tokenize(text)?;
        let attention_mask = Tensor::ones(input_ids.dims(), candle_core::DType::U32, &self.device)?;

        // Generate tokens and audio chunks
        let audio_chunks = self.generate_autoregressive_streaming(
            &input_ids,
            voice_embeds.as_ref(),
            speech_input_mask,
            Some(&attention_mask),
            max_new_tokens,
            chunk_callback,
        )?;

        Ok(audio_chunks)
    }

    /// Internal streaming generation method
    fn generate_autoregressive_streaming<F>(
        &mut self,
        input_ids: &Tensor,
        voice_embeds: Option<&Tensor>,
        speech_input_mask: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        max_new_tokens: Option<usize>,
        mut chunk_callback: Option<F>,
    ) -> Result<Vec<Tensor>>
    where
        F: FnMut(&Tensor) -> Result<()>,
    {
        debug!(
            "📝 Processing with pre-tokenized inputs, shape: {:?}",
            input_ids.dims()
        );

        let mut input_ids = input_ids.clone();
        let mut audio_chunks = Vec::new();

        // Initialize attention mask (all 1s if not provided)
        let mut attn_mask = if let Some(mask) = attention_mask {
            mask.clone()
        } else {
            let seq_len = input_ids.dim(1)?;
            Tensor::ones((1, seq_len), candle_core::DType::U32, &self.device)?
        };

        // Dual-pass architecture: prepare initial negative input
        let mut neg_input_ids = Tensor::new(&[[self.speech_start_id]], &self.device)?;
        let mut neg_attn_mask = Tensor::ones((1, 1), candle_core::DType::U32, &self.device)?;

        // Track if we've used voice embeddings on first pass
        let mut used_voice_embeds = false;

        // Determine max iterations (matching Python logic from line 421)
        // Python: max_steps = min(generation_config.max_length - initial_length, max_length_times * initial_length)
        // where max_length = initial_length + max_new_tokens
        let initial_length = input_ids.dims()[1] as usize;
        let max_length_times = 2;

        // If max_new_tokens provided, use it; otherwise use config max
        let effective_max_new_tokens = max_new_tokens
            .unwrap_or(self.config.llm_config.max_position_embeddings - initial_length);
        let max_steps = std::cmp::min(effective_max_new_tokens, max_length_times * initial_length);

        debug!(
            "📊 Input length: {}, max_new_tokens: {:?}, max_steps: {}",
            initial_length, max_new_tokens, max_steps
        );

        // Clear KV cache before starting generation
        self.llm.clear_kv_cache();

        // Track custom embeddings for SPEECH_DIFFUSION tokens
        let mut custom_embeds: Option<Tensor> = None;
        let mut neg_custom_embeds: Option<Tensor> = None;

        // Create streaming caches for semantic tokenizer and VAE decoder
        let mut semantic_cache = crate::streaming_cache::StreamingCache::new(self.device.clone());
        let mut acoustic_cache = crate::streaming_cache::StreamingCache::new(self.device.clone());

        // Start with EMPTY cache (matching Python's DynamicCache behavior)
        // Python uses DynamicCache() which starts empty - NOT StaticCache with zeros!
        // When DynamicCache.update is first called, it just appends (no concatenation).
        let mut neg_cache_opt: Option<Vec<Option<(Tensor, Tensor)>>> = None;

        // Will be set after the first SPEECH_DIFFUSION to store clean SPEECH_START K/V
        // This is used to reset the negative cache for each new speaker
        let mut initial_speech_start_kv: Option<Vec<Option<(Tensor, Tensor)>>> = None;

        // Track the absolute position in the negative sequence for cache_position-based masking
        let mut neg_cache_position: usize = 0;

        let mut step = 0;
        loop {
            // === STOPPING CONDITION ===
            if step >= max_steps {
                info!("⚠️ Reached max generation limit ({} steps)", max_steps);
                break;
            }

            let (current_input, seqlen_offset) = if step == 0 {
                (input_ids.clone(), 0)
            } else {
                let token_offset = input_ids.dim(1)? - 1;
                // ✅ FIX: For step N, the new token is at position (initial_length + N - 1)
                // Step 1: position 114, Step 2: position 115, etc.
                let rope_offset = initial_length + step - 1;
                (input_ids.narrow(1, token_offset, 1)?, rope_offset)
            };

            if step < 5 {
                debug!(
                    "📍 Step {}: seqlen_offset = {} (initial_length={})",
                    step, seqlen_offset, initial_length
                );
            }

            // Don't pass attention mask - let Candle create proper causal mask automatically
            // (We have no padding, so we only need causality)
            let current_attn_mask = None;

            // Debug output (step 0 only)
            if step == 0 {
                debug!(
                    "🔍 Step 0: input_ids shape={:?}, voice_embeds={}, used_voice_embeds={}",
                    input_ids.dims(),
                    voice_embeds.is_some(),
                    used_voice_embeds
                );
            }

            // === DUAL FORWARD PASS ===
            // Positive pass (for token generation - main context)
            let pos_hidden_states = if let Some(embeds) = custom_embeds.take() {
                // Use custom acoustic embedding from previous SPEECH_DIFFUSION token
                debug!(
                    "🎵 Using custom acoustic embedding (shape: {:?})",
                    embeds.dims()
                );
                self.llm
                    .forward_from_embeds(&embeds, seqlen_offset, current_attn_mask)?
            } else if step == 0
                && voice_embeds.is_some()
                && speech_input_mask.is_some()
                && !used_voice_embeds
            {
                debug!("🎤 Voice cloning active - injecting voice embeddings");

                // Get base token embeddings and inject voice at marked positions
                let base_embeds = self.get_token_embeddings(&current_input)?;

                // DIAGNOSTIC: Log base embeddings RMS
                let base_rms = base_embeds.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
                info!("[DIAG Step0] Base embeddings RMS: {:.6}", base_rms);

                // DIAGNOSTIC: Log voice embeddings RMS
                let voice_rms = voice_embeds.unwrap().sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
                info!("[DIAG Step0] Voice embeddings RMS: {:.6}", voice_rms);

                let injected_embeds = self.inject_voice_embeddings(
                    &base_embeds,
                    voice_embeds.unwrap(),
                    speech_input_mask.unwrap(),
                )?;

                // DIAGNOSTIC: Log injected embeddings RMS
                let injected_rms = injected_embeds.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
                info!("[DIAG Step0] Injected embeddings RMS: {:.6}", injected_rms);

                used_voice_embeds = true;

                // Forward pass WITH voice-injected embeddings
                self.llm
                    .forward_from_embeds(&injected_embeds, seqlen_offset, current_attn_mask)?
            } else {
                // Normal forward pass without voice injection
                if step == 0 {
                    debug!("📝 Normal forward pass (no voice injection)");
                }
                self.llm
                    .forward(&current_input, seqlen_offset, current_attn_mask)?
            };

            if step == 0 {
                debug!(
                    "🔍 After LLM Forward: hidden_states shape={:?}",
                    pos_hidden_states.dims()
                );
                // DIAGNOSTIC: Log hidden states RMS at last position (becomes condition for 1st diffusion)
                let hs_seq_len = pos_hidden_states.dim(1)?;
                let last_hidden = pos_hidden_states.i((.., hs_seq_len - 1, ..))?;
                let last_hidden_rms = last_hidden.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
                let all_hidden_rms = pos_hidden_states.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
                info!("[DIAG Step0] LLM hidden states: all_rms={:.6}, last_pos_rms={:.6}", all_hidden_rms, last_hidden_rms);
            }

            // ✅ Save positive cache AFTER positive forward
            let pos_cache = self.llm.extract_kv_cache();

            // Note: Negative forward pass ONLY runs inside SPEECH_DIFFUSION block
            // This matches Python's refresh_negative=True behavior (line 631-645)
            // Python only runs negative forward when generating SPEECH_DIFFUSION tokens

            // === TOKEN SAMPLING ===
            // Only compute logits for the last token (the one we're sampling).
            // This is a critical memory optimization matching Python's logits_to_keep=1.
            // For 12,866 tokens: reduces memory from 7.3GB to 0.6MB.
            let logits = self.get_logits(&pos_hidden_states, 1)?;

            // Get last token logits
            let seq_len = logits.dim(1)?;
            let logits_last = logits.i((.., seq_len - 1, ..))?;

            // Debug: Show top logits BEFORE constraints
            let logits_vec = &logits_last.to_vec2::<f32>()?[0];
            let mut indexed: Vec<(usize, f32)> = logits_vec
                .iter()
                .enumerate()
                .map(|(i, &v)| (i, v))
                .collect();
            indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

            debug!("🔍 Top 5 logits BEFORE constraints:");
            for (i, &(token_id, logit_val)) in indexed.iter().take(5).enumerate() {
                debug!("  {}. token {} = {:.4}", i + 1, token_id, logit_val);
            }

            // Sample next token
            let next_token = self.sample_next_token(&logits_last)?;

            // === STOPPING CONDITIONS ===
            let token_name = if next_token == self.speech_diffusion_id {
                "SPEECH_DIFFUSION"
            } else if next_token == self.speech_start_id {
                "SPEECH_START"
            } else if next_token == self.speech_end_id {
                "SPEECH_END"
            } else if next_token == self.eos_token_id {
                "EOS"
            } else {
                "OTHER"
            };
            debug!(
                "🎯 Step {}: token {} ({})",
                step + 1,
                next_token,
                token_name
            );

            // SPEECH_END zeros caches but continues until EOS
            // Matches Python's set_to_zero() which zeros values but keeps cache structure
            // This preserves context dimensions [B, C, context_size] with zero values
            if next_token == self.speech_end_id {
                tracing::info!(
                    "🔄 SPEECH_END at step {}: acoustic_cache entries={}, semantic_cache entries={}, audio_chunks={}",
                    step + 1,
                    acoustic_cache.cache_entry_count(),
                    semantic_cache.cache_entry_count(),
                    audio_chunks.len()
                );
                acoustic_cache.reset_to_zero();
                semantic_cache.reset_to_zero();
                tracing::info!("🔄 After reset: caches zeroed (structure preserved)");
            }

            if next_token == self.eos_token_id {
                debug!("✅ EOS: stopping generation after {} steps", step + 1);
                break;
            }

            // === AUDIO GENERATION ===
            if next_token == self.speech_diffusion_id {
                // Run negative forward ONLY for SPEECH_DIFFUSION tokens
                debug!("🔄 Running negative forward for CFG");

                // ✅ FIXED: neg_seqlen_offset = actual cache size = neg_cache_position
                // This is used for RoPE positional encoding, so it must match the actual cache state.
                // When cache is empty (first negative forward), this should be 0 (not neg_input_ids.dim - 1)!
                let neg_seqlen_offset = neg_cache_position;

                // Restore negative cache (or clear if first time)
                if let Some(ref neg_cache) = neg_cache_opt {
                    self.llm.restore_kv_cache(neg_cache.clone());
                } else {
                    self.llm.clear_kv_cache();
                }

                // Get the last token from neg_input_ids for forward
                let neg_current_input = neg_input_ids.narrow(1, neg_input_ids.dim(1)? - 1, 1)?;

                // ✅ FIXED: Use forward_with_cache_position to properly handle attention masks
                // Create cache_position tensor for the current token's absolute position
                let cache_pos_tensor = Tensor::new(&[neg_cache_position as i64], &self.device)?;

                // Pass both the 2D attention mask AND cache_position to enable proper 4D mask creation
                // This allows selective attention (e.g., after SPEECH_START, only attend to reset position)
                // DIAGNOSTIC: Log negative state before forward (for first token)
                let token_num = audio_chunks.len() + 1;
                if token_num <= 3 {
                    info!("[DIAG NegFwd Token{}] Before: neg_seqlen_offset={}, neg_cache_pos={}, neg_input_len={}, mask_len={}",
                        token_num, neg_seqlen_offset, neg_cache_position, neg_input_ids.dim(1)?, neg_attn_mask.dim(1)?);
                    let mask_vec: Vec<u32> = neg_attn_mask.flatten_all()?.to_vec1()?;
                    let ones_count = mask_vec.iter().filter(|&&x| x == 1).count();
                    info!("[DIAG NegFwd Token{}] Mask ones: {}/{}", token_num, ones_count, mask_vec.len());
                }

                let neg_hidden_states = if let Some(neg_embeds) = neg_custom_embeds.take() {
                    self.llm.forward_from_embeds_with_cache_position(
                        &neg_embeds,
                        neg_seqlen_offset,
                        Some(&neg_attn_mask),
                        Some(&cache_pos_tensor),
                    )?
                } else {
                    self.llm.forward_with_cache_position(
                        &neg_current_input,
                        neg_seqlen_offset,
                        Some(&neg_attn_mask),
                        Some(&cache_pos_tensor),
                    )?
                };

                // DIAGNOSTIC: Log negative hidden states after forward (for first 3 tokens)
                if token_num <= 3 {
                    let neg_hs_seq_len = neg_hidden_states.dim(1)?;
                    let neg_last_hidden = neg_hidden_states.i((.., neg_hs_seq_len - 1, ..))?;
                    let neg_last_rms = neg_last_hidden.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
                    info!("[DIAG NegFwd Token{}] After: neg_hidden_states shape={:?}, last_pos_rms={:.6}",
                        token_num, neg_hidden_states.dims(), neg_last_rms);
                }

                // === MATCH PYTHON's _update_model_kwargs_for_generation ===
                // Python updates all three immediately after forward:
                // 1. past_key_values (KV cache)
                // 2. attention_mask (append 1)
                // 3. cache_position (increment)

                // 1. Save negative cache (past_key_values)
                neg_cache_opt = Some(self.llm.extract_kv_cache());

                // Store initial K/V after first SPEECH_DIFFUSION for later speaker resets
                // This captures the clean SPEECH_START context before any contamination
                if initial_speech_start_kv.is_none() {
                    initial_speech_start_kv = neg_cache_opt.clone();
                    debug!("📌 Stored initial SPEECH_START K/V for speaker resets");
                }

                // 2. Update attention_mask - append 1 (Python line 903-905)
                let neg_mask_tensor = Tensor::ones((1, 1), candle_core::DType::U32, &self.device)?;
                neg_attn_mask = Tensor::cat(&[&neg_attn_mask, &neg_mask_tensor], 1)?;

                // 3. Increment cache_position (Python line 916)
                neg_cache_position += 1;

                // Also append to neg_input_ids (Python line 645)
                let neg_token_tensor = Tensor::new(&[[next_token]], &self.device)?;
                neg_input_ids = Tensor::cat(&[&neg_input_ids, &neg_token_tensor], 1)?;

                debug!(
                    "📝 Updated negative state: mask_len={}, cache_pos={}, input_len={}",
                    neg_attn_mask.dim(1)?,
                    neg_cache_position,
                    neg_input_ids.dim(1)?
                );

                // Restore positive cache
                self.llm.restore_kv_cache(pos_cache.clone());

                // Now extract conditions for diffusion
                // BUG FIX: Use pos_hidden_states.dim(1), NOT seq_len (which is from logits = 1)
                let pos_seq_len = pos_hidden_states.dim(1)?;
                let condition = pos_hidden_states.i((.., pos_seq_len - 1, ..))?;
                let neg_seq_len = neg_hidden_states.dim(1)?;
                let neg_condition = neg_hidden_states.i((.., neg_seq_len - 1, ..))?;

                // Diagnostic logging: Track CFG condition quality per token
                // Consistent RMS values across tokens indicate proper negative condition handling
                let pos_cond_rms = condition.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
                let neg_cond_rms = neg_condition.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
                tracing::info!(
                    "[CFG] Token {}: pos_condition RMS={:.6}, neg_condition RMS={:.6}",
                    audio_chunks.len() + 1,
                    pos_cond_rms,
                    neg_cond_rms
                );

                // Step 1: Sample acoustic latent from diffusion
                // BUGFIX: Use solver.num_steps which is set by set_ddpm_inference_steps(),
                // not the config value which is never updated
                let num_steps = self.solver.num_steps;
                let acoustic_latent =
                    self.sample_dpm_solver(&condition, &neg_condition, num_steps, self.cfg_scale)?;
                debug!("  Generated acoustic latent: {:?}", acoustic_latent.dims());

                // Step 2: Decode to audio waveform with streaming cache
                let scale_val = self.speech_scaling_factor.to_scalar::<f32>()?;
                let bias_val = self.speech_bias_factor.to_scalar::<f32>()?;
                let denormalized =
                    acoustic_latent.affine(1.0 / scale_val as f64, -bias_val as f64)?;
                let decoder_input = denormalized.unsqueeze(2)?;
                let audio = self
                    .vae_decoder
                    .decode_with_cache(&decoder_input, &mut acoustic_cache)?;

                // Step 3: Create embeddings for next iteration
                let acoustic_embed = self
                    .acoustic_connector
                    .forward(&acoustic_latent.unsqueeze(1)?)?;
                let semantic_latent = self
                    .semantic_tokenizer
                    .encode_with_cache(&audio, &mut semantic_cache)?;
                let semantic_embed = self.semantic_connector.forward(&semantic_latent)?;
                let combined_embed = (&acoustic_embed + &semantic_embed)?;

                // === DIAGNOSTIC: Track quality metrics per token ===
                let audio_rms = audio.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
                let acoustic_rms = acoustic_latent
                    .sqr()?
                    .mean_all()?
                    .sqrt()?
                    .to_scalar::<f32>()?;
                let semantic_rms = semantic_latent
                    .sqr()?
                    .mean_all()?
                    .sqrt()?
                    .to_scalar::<f32>()?;
                let combined_rms = combined_embed
                    .sqr()?
                    .mean_all()?
                    .sqrt()?
                    .to_scalar::<f32>()?;

                // Store generated audio (before logging to get correct count)
                audio_chunks.push(audio.clone());

                tracing::info!(
                    "📊 Token {}: audio_rms={:.6}, acoustic_rms={:.6}, semantic_rms={:.6}, combined_rms={:.6}",
                    audio_chunks.len(),
                    audio_rms,
                    acoustic_rms,
                    semantic_rms,
                    combined_rms
                );

                // Store for next iteration
                custom_embeds = Some(combined_embed.clone());
                neg_custom_embeds = Some(combined_embed);

                if let Some(ref mut callback) = chunk_callback {
                    callback(&audio)?;
                }
            } else {
                // For non-SPEECH_DIFFUSION tokens, use regular token embeddings
                custom_embeds = None;
                neg_custom_embeds = None;
            }

            // === UPDATE SEQUENCES ===
            // Update positive input_ids and attention mask (every step)
            let next_token_tensor = Tensor::new(&[[next_token]], &self.device)?;
            input_ids = Tensor::cat(&[&input_ids, &next_token_tensor], 1)?;

            let new_mask = Tensor::ones((1, 1), candle_core::DType::U32, &self.device)?;
            attn_mask = Tensor::cat(&[&attn_mask, &new_mask], 1)?;

            // Handle SPEECH_START special case (negative state updates for SPEECH_DIFFUSION
            // are now done immediately after forward, matching Python's _update_model_kwargs_for_generation)
            if next_token == self.speech_start_id {
                tracing::info!(
                    "🎬 SPEECH_START at step {}: starting new speaker segment (neg_mask_len={}, neg_cache_pos={})",
                    step + 1,
                    neg_attn_mask.dim(1).unwrap_or(0),
                    neg_cache_position
                );
                // ✅ PROPER IMPLEMENTATION: Match Python's SPEECH_START handling (lines 607-621)
                //
                // Python's approach:
                //   1. Reset attention mask to [0, 0, ..., 0, 1] (only last position attended)
                //   2. Shift KV cache: copy position 0 to last position
                //   3. Overwrite neg_input_ids[-1] with SPEECH_START
                //   4. Next SPEECH_DIFFUSION: attention mask [0,0,...,0,1] + cache_position = proper masking
                //
                // The new prepare_4d_causal_attention_mask_with_cache_position handles this correctly:
                // - The 2D mask [0,0,...,0,1] combined with cache_position creates a 4D mask
                // - Only the last position (where we shifted SPEECH_START K/V) is attended to

                // Overwrite last position of neg_input_ids with SPEECH_START (Python line 621)
                let seq_len = neg_input_ids.dim(1)?;
                if seq_len > 0 {
                    if seq_len > 1 {
                        let prefix = neg_input_ids.narrow(1, 0, seq_len - 1)?;
                        let new_last = Tensor::new(&[[self.speech_start_id]], &self.device)?;
                        neg_input_ids = Tensor::cat(&[&prefix, &new_last], 1)?;
                    } else {
                        neg_input_ids = Tensor::new(&[[self.speech_start_id]], &self.device)?;
                    }
                }

                // Reset attention mask: all zeros except last position = 1 (Python lines 607-610)
                // This makes the next SPEECH_DIFFUSION only attend to the last position
                let mask_len = neg_attn_mask.dim(1)?;
                let mut mask_data = vec![0u32; mask_len];
                mask_data[mask_len - 1] = 1;
                neg_attn_mask = Tensor::new(mask_data.as_slice(), &self.device)?.unsqueeze(0)?;

                // FIX: Copy position 0 to last position in CURRENT neg cache
                // This matches Python's approach (lines 612-618 in modeling_vibevoice_inference.py):
                //   k_cache[sample_idx, :, -1, :] = k_cache[sample_idx, :, 0, :].clone()
                //   v_cache[sample_idx, :, -1, :] = v_cache[sample_idx, :, 0, :].clone()
                //
                // IMPORTANT: Python operates on the CURRENT cache (keeping full length),
                // NOT restoring from an initial cache. The previous implementation restored
                // from initial_speech_start_kv which had only 1-2 positions, causing a mismatch
                // with the 17-position attention mask.
                //
                // The key insight: Position 0 in the current neg cache still contains the
                // original SPEECH_START context because we never modified it - we only
                // appended new positions. So copying pos 0 to pos -1 achieves the same
                // result as Python without needing to save/restore the initial cache.
                if let Some(ref cache) = neg_cache_opt {
                    let temp_pos_cache = self.llm.extract_kv_cache();
                    self.llm.restore_kv_cache(cache.clone());
                    self.llm.shift_kv_cache_first_to_last()?;
                    neg_cache_opt = Some(self.llm.extract_kv_cache());
                    self.llm.restore_kv_cache(temp_pos_cache);
                }

                // Note: neg_cache_position stays the same (it's the position of the next token to generate)
                // The attention mask handles hiding old positions, not the cache_position

                debug!(
                    "📝 SPEECH_START: Reset attention mask and shifted cache (len={})",
                    neg_input_ids.dim(1)?
                );
            }
            // For other tokens: no change to neg_input_ids (Python doesn't append)

            step += 1;
        }

        // Generation summary
        let total_chunks = audio_chunks.len();
        let total_samples: usize = audio_chunks
            .iter()
            .map(|c| c.dims().iter().product::<usize>())
            .sum();
        let duration_sec = total_samples as f32 / 24000.0;
        debug!(
            "📊 Generation complete: {} chunks, {} samples ({:.2}s), {} steps",
            total_chunks,
            total_samples,
            duration_sec,
            step + 1
        );
        Ok(audio_chunks)
    }
}
