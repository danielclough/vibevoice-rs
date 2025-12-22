//! Windowed generation for VibeVoice Realtime streaming model.
//!
//! This module implements the main generation loop for streaming TTS:
//! - Process text in windows of 5 tokens
//! - Generate 6 speech tokens per text window
//! - Use CFG (Classifier-Free Guidance) with parallel positive/negative paths
//! - Diffusion sampling with 5 steps
//!
//! # Generation Flow
//!
//! ```text
//! for each text window (5 tokens):
//!     forward_lm(text) → hidden states (both paths)
//!     forward_tts_lm(text, hidden) → TTS conditions (both paths)
//!
//!     for each speech token (6 per window):
//!         sample_diffusion(positive, negative) → latent
//!         vae_decode(latent) → audio chunk
//!         stream_audio(chunk)
//!         acoustic_connector(latent) → embedding
//!         forward_tts_lm(speech_embed) → next conditions (both paths)
//!         check_eos() → break if done
//! ```
//!
//! # Python Reference
//!
//! From `modeling_vibevoice_streaming_inference.py:525-704`

use crate::{realtime::{binary_classifier::BinaryClassifier, config::{RealtimeConfig}, split_llm::DualSplitLLM, voice_cache::SafetensorCache}, streaming_cache::StreamingCache};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use tracing::debug;

/// Configuration for the generation process.
pub struct GenerationConfig {
    /// CFG scale for classifier-free guidance (default: 1.5)
    pub cfg_scale: f32,
    /// Number of diffusion steps (default: 5 for realtime)
    pub num_diffusion_steps: usize,
    /// Speech scaling factor for VAE normalization
    pub speech_scaling_factor: f32,
    /// Speech bias factor for VAE normalization
    pub speech_bias_factor: f32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            cfg_scale: 1.5,
            num_diffusion_steps: 5,
            speech_scaling_factor: 1.0,
            speech_bias_factor: 0.0,
        }
    }
}

/// State for the windowed generation process.
///
/// Tracks KV caches, hidden states, and positions for both positive and negative paths.
/// Note: lm and tts_lm have different cache positions because:
/// - lm only processes text tokens
/// - tts_lm processes text + speech tokens
pub struct GenerationState {
    // Positive path (conditioned on text)
    /// Last hidden state from lower LM (positive path)
    pub lm_hidden: Tensor,
    /// Last hidden state from upper TTS LM (positive path)
    pub tts_lm_hidden: Tensor,

    // Negative path (for CFG)
    /// Last hidden state from lower LM (negative path)
    pub neg_lm_hidden: Tensor,
    /// Last hidden state from upper TTS LM (negative path)
    pub neg_tts_lm_hidden: Tensor,

    /// Current cache position for lower LM (positive path)
    pub lm_cache_position: usize,
    /// Current cache position for upper TTS LM (positive path)
    pub tts_lm_cache_position: usize,
    /// Current cache position for lower LM (negative path)
    pub neg_lm_cache_position: usize,
    /// Current cache position for upper TTS LM (negative path)
    pub neg_tts_lm_cache_position: usize,

    /// Streaming cache for VAE decoder
    pub acoustic_cache: StreamingCache,

    /// Whether generation has finished (EOS detected)
    pub finished: bool,
}

impl GenerationState {
    /// Initialize generation state from a voice cache.
    pub fn from_voice_cache(voice_cache: &SafetensorCache, device: &Device) -> Result<Self> {
        let (lm_len, tts_lm_len, neg_lm_len, neg_tts_lm_len) = voice_cache.cache_positions()?;

        debug!(
            "📍 [POSITION] Initialized from voice cache: lm={}, tts_lm={}, neg_lm={}, neg_tts_lm={}",
            lm_len, tts_lm_len, neg_lm_len, neg_tts_lm_len
        );

        Ok(Self {
            lm_hidden: voice_cache.lm.last_hidden_state.clone(),
            tts_lm_hidden: voice_cache.tts_lm.last_hidden_state.clone(),
            neg_lm_hidden: voice_cache.neg_lm.last_hidden_state.clone(),
            neg_tts_lm_hidden: voice_cache.neg_tts_lm.last_hidden_state.clone(),
            lm_cache_position: lm_len,
            tts_lm_cache_position: tts_lm_len,
            neg_lm_cache_position: neg_lm_len,
            neg_tts_lm_cache_position: neg_tts_lm_len,
            acoustic_cache: StreamingCache::new(device.clone()),
            finished: false,
        })
    }
}

/// Windowed generator for streaming TTS.
///
/// This is a stateful generator that processes text in windows and
/// produces audio chunks incrementally.
///
/// # Usage
///
/// ```ignore
/// let mut generator = WindowedGenerator::new(config, device)?;
/// generator.initialize_from_cache(&voice_cache, &mut dual_split_llm)?;
///
/// for window in text_windows(tts_text_ids, 5) {
///     // Process text window (updates both positive and negative paths)
///     generator.process_text_window(&window, &mut dual_split_llm)?;
///
///     // Generate speech tokens
///     for _ in 0..6 {
///         let positive = generator.get_positive_condition()?;
///         let negative = generator.get_negative_condition()?;
///
///         // Diffusion sampling with CFG...
///
///         generator.update_after_speech_token(&acoustic_embed, &mut dual_split_llm)?;
///
///         if generator.state.finished {
///             break;
///         }
///     }
/// }
/// ```
pub struct WindowedGenerator {
    /// Generation configuration
    pub config: GenerationConfig,
    /// Generation state (caches, positions, hidden states)
    pub state: GenerationState,
    /// Device for tensor operations
    device: Device,
    /// Hidden size for creating tensors
    #[allow(dead_code)]
    hidden_size: usize,
    /// Speech token counter (for logging)
    #[allow(dead_code)]
    speech_token_count: usize,
}

impl WindowedGenerator {
    /// Create a new generator (uninitialized - call `initialize_from_cache` next).
    pub fn new(
        gen_config: GenerationConfig,
        model_config: &RealtimeConfig,
        device: Device,
    ) -> Result<Self> {
        let hidden_size = model_config.hidden_size();

        // Create placeholder state (will be replaced by initialize_from_cache)
        let placeholder = Tensor::zeros((1, 1, hidden_size), DType::F32, &device)?;

        let state = GenerationState {
            lm_hidden: placeholder.clone(),
            tts_lm_hidden: placeholder.clone(),
            neg_lm_hidden: placeholder.clone(),
            neg_tts_lm_hidden: placeholder,
            lm_cache_position: 0,
            tts_lm_cache_position: 0,
            neg_lm_cache_position: 0,
            neg_tts_lm_cache_position: 0,
            acoustic_cache: StreamingCache::new(device.clone()),
            finished: false,
        };

        Ok(Self {
            config: gen_config,
            state,
            device,
            hidden_size,
            speech_token_count: 0,
        })
    }

    /// Initialize the generator from a voice cache.
    ///
    /// This loads the pre-computed KV states and hidden states from the voice cache
    /// into both the generator state and the DualSplitLLM (all 4 caches).
    pub fn initialize_from_cache(
        &mut self,
        voice_cache: &SafetensorCache,
        dual_split_llm: &mut DualSplitLLM,
    ) -> Result<()> {
        // Update state from voice cache
        self.state = GenerationState::from_voice_cache(voice_cache, &self.device)?;

        // Restore ALL KV caches to DualSplitLLM (positive and negative paths)
        dual_split_llm.restore_from_voice_cache(voice_cache);

        // Log KV cache statistics for debugging
        dual_split_llm.log_kv_cache_stats();

        Ok(())
    }

    /// Process a text window through both LLM paths (positive and negative).
    ///
    /// This forwards the text tokens through both paths and updates all hidden states.
    ///
    /// # Arguments
    ///
    /// * `text_window` - Text token IDs, shape `[1, window_len]` (up to 5 tokens)
    /// * `dual_split_llm` - The dual split LLM for forward passes
    pub fn process_text_window(
        &mut self,
        text_window: &Tensor,
        dual_split_llm: &mut DualSplitLLM,
    ) -> Result<()> {
        let window_len = text_window.dim(1)?;
        if window_len == 0 {
            return Ok(());
        }

        debug!(
            "📍 [POSITION] Before text window: lm={}, tts_lm={}, neg_lm={}, neg_tts_lm={}",
            self.state.lm_cache_position,
            self.state.tts_lm_cache_position,
            self.state.neg_lm_cache_position,
            self.state.neg_tts_lm_cache_position
        );

        // Forward through both lower LMs (4 layers, no norm)
        let lm_output = dual_split_llm.forward_lm(
            text_window,
            self.state.lm_cache_position,
            self.state.neg_lm_cache_position,
        )?;
        self.state.lm_hidden = lm_output.pos_hidden;
        self.state.neg_lm_hidden = lm_output.neg_hidden;

        // Create text masks (all 1s for text tokens)
        let tts_text_masks = Tensor::ones((1, window_len), DType::U32, &self.device)?;

        // Forward through both upper TTS LMs (20 layers)
        let tts_output = dual_split_llm.forward_tts_lm(
            text_window,
            &self.state.lm_hidden,
            &self.state.neg_lm_hidden,
            &tts_text_masks,
            self.state.tts_lm_cache_position,
            self.state.neg_tts_lm_cache_position,
        )?;
        self.state.tts_lm_hidden = tts_output.pos_hidden;
        self.state.neg_tts_lm_hidden = tts_output.neg_hidden;

        // Update ALL cache positions
        self.state.lm_cache_position += window_len;
        self.state.tts_lm_cache_position += window_len;
        self.state.neg_lm_cache_position += window_len;
        self.state.neg_tts_lm_cache_position += window_len;

        debug!(
            "[Generator::process_text_window] AFTER: lm_pos: {}, tts_pos: {}, neg_lm_pos: {}, neg_tts_pos: {}",
            self.state.lm_cache_position,
            self.state.tts_lm_cache_position,
            self.state.neg_lm_cache_position,
            self.state.neg_tts_lm_cache_position
        );

        Ok(())
    }

    /// Extract the positive condition for diffusion (last hidden state).
    pub fn get_positive_condition(&self) -> Result<Tensor> {
        let seq_len = self.state.tts_lm_hidden.dim(1)?;
        debug!(
            "[Generator::get_positive_condition] tts_lm_hidden: {:?}, seq_len: {}, extracting position {}",
            self.state.tts_lm_hidden.dims(),
            seq_len,
            seq_len - 1
        );
        // Get last position: [batch, hidden_size]
        Ok(self
            .state
            .tts_lm_hidden
            .narrow(1, seq_len - 1, 1)?
            .squeeze(1)?)
    }

    /// Extract the negative condition for CFG (last hidden state).
    pub fn get_negative_condition(&self) -> Result<Tensor> {
        let seq_len = self.state.neg_tts_lm_hidden.dim(1)?;
        debug!(
            "[Generator::get_negative_condition] neg_tts_lm_hidden: {:?}, seq_len: {}, extracting position {}",
            self.state.neg_tts_lm_hidden.dims(),
            seq_len,
            seq_len - 1
        );
        Ok(self
            .state
            .neg_tts_lm_hidden
            .narrow(1, seq_len - 1, 1)?
            .squeeze(1)?)
    }

    /// Update state after generating a speech token (both paths).
    ///
    /// Called after diffusion sampling and VAE decode to prepare for the next token.
    /// Updates both positive and negative TTS LM hidden states.
    ///
    /// # Arguments
    ///
    /// * `acoustic_embed` - The acoustic embedding from acoustic_connector, shape `[1, 1, hidden_size]`
    /// * `dual_split_llm` - The dual split LLM for forward pass
    pub fn update_after_speech_token(
        &mut self,
        acoustic_embed: &Tensor,
        dual_split_llm: &mut DualSplitLLM,
    ) -> Result<()> {
        // Create speech mask (0 for speech tokens)
        let speech_mask = Tensor::zeros((1, 1), DType::U32, &self.device)?;

        // Forward through both TTS LMs with acoustic embedding
        let tts_output = dual_split_llm.forward_tts_lm_with_acoustic(
            acoustic_embed,
            &speech_mask,
            self.state.tts_lm_cache_position,
            self.state.neg_tts_lm_cache_position,
        )?;

        self.state.tts_lm_hidden = tts_output.pos_hidden;
        self.state.neg_tts_lm_hidden = tts_output.neg_hidden;

        // Update TTS LM positions for both paths (lower LM doesn't process speech)
        self.state.tts_lm_cache_position += 1;
        self.state.neg_tts_lm_cache_position += 1;

        debug!(
            "[Generator::update_after_speech_token] AFTER: tts_pos: {}, neg_tts_pos: {}",
            self.state.tts_lm_cache_position, self.state.neg_tts_lm_cache_position
        );

        Ok(())
    }

    /// Check EOS using the binary classifier.
    pub fn check_eos(&mut self, classifier: &BinaryClassifier) -> Result<bool> {
        let condition = self.get_positive_condition()?;
        let should_stop = classifier.should_stop(&condition)?;

        if should_stop {
            self.state.finished = true;
        }

        Ok(should_stop)
    }

    /// Check if generation has finished.
    pub fn is_finished(&self) -> bool {
        self.state.finished
    }
}

/// Split text token IDs into windows of the specified size.
///
/// # Arguments
///
/// * `text_ids` - Text token IDs, shape `[1, total_len]`
/// * `window_size` - Maximum window size (default: 5)
///
/// # Returns
///
/// Iterator over windows, each with shape `[1, window_len]`
pub fn text_windows(text_ids: &Tensor, window_size: usize) -> Result<Vec<Tensor>> {
    let total_len = text_ids.dim(1)?;
    let mut windows = Vec::new();

    let mut start = 0;
    while start < total_len {
        let end = (start + window_size).min(total_len);
        let window = text_ids.narrow(1, start, end - start)?;
        windows.push(window);
        start = end;
    }

    Ok(windows)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_windows_exact() {
        // Test with exact multiple of window size
        let device = Device::Cpu;
        let text_ids = Tensor::zeros((1, 10), DType::U32, &device).unwrap();

        let windows = text_windows(&text_ids, 5).unwrap();
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].dim(1).unwrap(), 5);
        assert_eq!(windows[1].dim(1).unwrap(), 5);
    }

    #[test]
    fn test_text_windows_partial() {
        // Test with partial final window
        let device = Device::Cpu;
        let text_ids = Tensor::zeros((1, 12), DType::U32, &device).unwrap();

        let windows = text_windows(&text_ids, 5).unwrap();
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0].dim(1).unwrap(), 5);
        assert_eq!(windows[1].dim(1).unwrap(), 5);
        assert_eq!(windows[2].dim(1).unwrap(), 2); // Partial window
    }

    #[test]
    fn test_text_windows_short() {
        // Test with text shorter than window size
        let device = Device::Cpu;
        let text_ids = Tensor::zeros((1, 3), DType::U32, &device).unwrap();

        let windows = text_windows(&text_ids, 5).unwrap();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].dim(1).unwrap(), 3);
    }

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.cfg_scale, 1.5);
        assert_eq!(config.num_diffusion_steps, 5);
    }
}
