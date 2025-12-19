//! Dual Split LLM architecture for VibeVoice Realtime streaming model with CFG support.
//!
//! The streaming model splits a 24-layer Qwen2 into:
//! - `language_model`: Lower 4 layers for text processing (no final norm)
//! - `tts_language_model`: Upper 20 layers for TTS generation
//!
//! For CFG (Classifier-Free Guidance), we maintain two parallel paths:
//! - **Positive path**: Conditioned on text and voice
//! - **Negative path**: Unconditional baseline
//!
//! # Architecture
//!
//! ```text
//! Text tokens
//!      ↓
//! embed_tokens (shared)
//!      ↓
//! ┌─────────────────────────────────────────────────────┐
//! │         language_model (4 Qwen2 layers)             │
//! │  ┌──────────────┐       ┌──────────────┐            │
//! │  │ pos_lm (KV1) │       │ neg_lm (KV2) │            │
//! │  └──────────────┘       └──────────────┘            │
//! │  forward_from_embeds_no_norm()                      │
//! └─────────────────────────────────────────────────────┘
//!      ↓ hidden states (spliced into tts_language_model input)
//! ┌─────────────────────────────────────────────────────┐
//! │       tts_language_model (20 Qwen2 layers)          │
//! │  ┌──────────────┐       ┌──────────────┐            │
//! │  │ pos_tts (KV3)│       │ neg_tts (KV4)│            │
//! │  └──────────────┘       └──────────────┘            │
//! │  + tts_input_types embedding                        │
//! └─────────────────────────────────────────────────────┘
//!      ↓
//! pos_condition, neg_condition → CFG diffusion
//! ```
//!
//! # CFG Formula
//!
//! ```text
//! output = negative + cfg_scale * (positive - negative)
//! ```
//!
//! # Python Reference
//!
//! From `modeling_vibevoice_streaming.py:106-122`:
//! ```python
//! # Split LLM creation
//! lm_backbone_num_hidden_layers = num_hidden_layers - tts_backbone_num_hidden_layers  # 24 - 20 = 4
//! self.language_model = AutoModel.from_config(lm_config)
//! self.language_model.norm = nn.Identity()  # No final norm!
//! self.tts_language_model = AutoModel.from_config(tts_lm_config)
//! self.tts_input_types = nn.Embedding(num_embeddings=2, embedding_dim=hidden_size)
//! ```

use crate::realtime::config::RealtimeConfig;
use crate::realtime::voice_cache::VoiceCache;
use crate::utils::tensor_stats;
use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Embedding, VarBuilder};
use candle_transformers::models::qwen2::Model as Qwen2Model;
use tracing::{debug, info};

/// Output from dual lower language_model forward pass.
#[derive(Debug)]
pub struct DualLmOutput {
    /// Positive path hidden states (conditioned).
    /// Shape: `[batch, seq_len, hidden_size]`
    pub pos_hidden: Tensor,

    /// Negative path hidden states (unconditional).
    /// Shape: `[batch, seq_len, hidden_size]`
    pub neg_hidden: Tensor,
}

/// Output from dual upper tts_language_model forward pass.
#[derive(Debug)]
pub struct DualTtsLmOutput {
    /// Positive path hidden states.
    /// Shape: `[batch, seq_len, hidden_size]`
    pub pos_hidden: Tensor,

    /// Negative path hidden states.
    /// Shape: `[batch, seq_len, hidden_size]`
    pub neg_hidden: Tensor,
}

/// KV cache storage for all 4 CFG paths.
#[derive(Clone)]
pub struct DualKvCaches {
    /// Positive lower LM cache (4 layers)
    pub pos_lm: Vec<Option<(Tensor, Tensor)>>,

    /// Positive upper TTS LM cache (20 layers)
    pub pos_tts_lm: Vec<Option<(Tensor, Tensor)>>,

    /// Negative lower LM cache (4 layers)
    pub neg_lm: Vec<Option<(Tensor, Tensor)>>,

    /// Negative upper TTS LM cache (20 layers)
    pub neg_tts_lm: Vec<Option<(Tensor, Tensor)>>,
}

/// Dual Split LLM for VibeVoice Realtime streaming with CFG support.
///
/// Contains two parallel paths (positive and negative), each with:
/// - `language_model`: 4 lower layers for text encoding (no final RMS norm)
/// - `tts_language_model`: 20 upper layers for TTS generation
///
/// Model weights are shared via VarBuilder's tensor registry, but each path
/// maintains independent KV caches for proper CFG computation.
///
/// # Weight Paths
///
/// ```text
/// model.language_model.embed_tokens.weight       [vocab_size, hidden_size]
/// model.language_model.layers.{0-3}.*            Lower 4 layers
/// model.tts_language_model.layers.{0-19}.*       Upper 20 layers
/// model.tts_input_types.weight                   [2, hidden_size]
/// ```
pub struct DualSplitLLM {
    /// Shared token embeddings (from language_model.embed_tokens).
    /// Shape: `[vocab_size, hidden_size]`
    embed_tokens: Tensor,

    /// Positive path: Lower 4 layers for text processing.
    /// Uses `forward_from_embeds_no_norm()` to skip final RMS norm.
    pos_language_model: Qwen2Model,

    /// Negative path: Lower 4 layers (same weights, separate KV cache).
    neg_language_model: Qwen2Model,

    /// Positive path: Upper 20 layers for TTS generation.
    pos_tts_language_model: Qwen2Model,

    /// Negative path: Upper 20 layers (same weights, separate KV cache).
    neg_tts_language_model: Qwen2Model,

    /// Type embedding: index 0 = speech, index 1 = text.
    /// Used to mark tokens as speech or text for the TTS LM.
    tts_input_types: Embedding,

    /// Hidden dimension for embedding lookups.
    hidden_size: usize,

    /// Device for tensor operations.
    device: Device,

    /// Padding token ID for unconditional CFG baseline (<|image_pad|> in Qwen2).
    image_pad_token_id: u32,

    /// Call counter for LM forward (for first-call logging)
    lm_forward_count: usize,

    /// Call counter for TTS LM forward (for first-call logging)
    tts_lm_forward_count: usize,
}

impl DualSplitLLM {
    /// Create a new DualSplitLLM from pretrained weights.
    ///
    /// Creates two parallel paths (positive and negative) that share model weights
    /// but maintain independent KV caches for CFG.
    ///
    /// # Arguments
    ///
    /// * `vb` - VarBuilder pointing to model weights
    /// * `config` - RealtimeConfig with layer split information
    ///
    /// # Weight Sharing
    ///
    /// When creating multiple Qwen2Model instances with the same VarBuilder,
    /// Candle's internal tensor registry ensures the actual weight tensors are
    /// shared in memory (reference counted), only KV cache state is duplicated.
    pub fn new(vb: VarBuilder, config: &RealtimeConfig, image_pad_token_id: u32) -> Result<Self> {
        let hidden_size = config.hidden_size();
        let device = vb.device().clone();

        // Load shared embeddings from language_model
        // After remapping: model.language_model.model.embed_tokens.weight
        let embed_tokens = vb
            .pp("model.language_model.model.embed_tokens")
            .get(&[config.llm_config.vocab_size, hidden_size], "weight")?;

        // Load lower 4-layer language_model configs
        let lm_config = config.to_lm_qwen2_config();
        let tts_lm_config = config.to_tts_lm_qwen2_config();

        // Create positive path models
        let pos_language_model = Qwen2Model::new(&lm_config, vb.pp("model.language_model"))?;
        let pos_tts_language_model =
            Qwen2Model::new(&tts_lm_config, vb.pp("model.tts_language_model"))?;

        // Create negative path models (same weights via VarBuilder, separate KV caches)
        let neg_language_model = Qwen2Model::new(&lm_config, vb.pp("model.language_model"))?;
        let neg_tts_language_model =
            Qwen2Model::new(&tts_lm_config, vb.pp("model.tts_language_model"))?;

        // Load type embeddings (2 embeddings: 0=speech, 1=text)
        let tts_input_types = candle_nn::embedding(2, hidden_size, vb.pp("model.tts_input_types"))?;

        Ok(Self {
            embed_tokens,
            pos_language_model,
            neg_language_model,
            pos_tts_language_model,
            neg_tts_language_model,
            tts_input_types,
            hidden_size,
            device,
            image_pad_token_id,
            lm_forward_count: 0,
            tts_lm_forward_count: 0,
        })
    }

    /// Get token embeddings for the given input IDs.
    ///
    /// Performs GPU-side embedding lookup.
    fn get_embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;
        let flat_ids = input_ids.flatten_all()?.to_dtype(DType::U32)?;
        let flat_embeds = self.embed_tokens.index_select(&flat_ids, 0)?;
        Ok(flat_embeds.reshape((batch_size, seq_len, self.hidden_size))?)
    }

    /// Forward through both lower language_models (4 layers, no final norm).
    ///
    /// Processes text tokens through both positive and negative paths.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Text token IDs, shape `[batch, seq_len]`
    /// * `pos_seqlen_offset` - Position offset for positive path KV cache
    /// * `neg_seqlen_offset` - Position offset for negative path KV cache
    ///
    /// # Returns
    ///
    /// `DualLmOutput` with hidden states from both paths (no final norm applied).
    pub fn forward_lm(
        &mut self,
        input_ids: &Tensor,
        pos_seqlen_offset: usize,
        neg_seqlen_offset: usize,
    ) -> Result<DualLmOutput> {
        // Positive path: actual text embeddings
        let inputs_embeds = self.get_embeddings(input_ids)?;
        let pos_hidden = self.pos_language_model.forward_from_embeds_no_norm(
            &inputs_embeds,
            pos_seqlen_offset,
            None,
        )?;

        // Negative path: padding token embeddings (unconditional baseline)
        // This matches Python's behavior where neg_text_input_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        let neg_input_ids = Tensor::full(self.image_pad_token_id, input_ids.shape(), &self.device)?;
        let neg_embeds = self.get_embeddings(&neg_input_ids)?;
        let neg_hidden = self.neg_language_model.forward_from_embeds_no_norm(
            &neg_embeds,
            neg_seqlen_offset,
            None,
        )?;

        debug!(
            "[DualSplitLLM::forward_lm] input_ids: {:?}, pos_offset: {}, neg_offset: {}, pos_hidden: {:?}, neg_hidden: {:?}",
            input_ids.dims(),
            pos_seqlen_offset,
            neg_seqlen_offset,
            pos_hidden.dims(),
            neg_hidden.dims()
        );

        // Log stats on first call (matches Python's "FIRST LM FORWARD")
        if self.lm_forward_count == 0 {
            info!(
                "🔍 [FIRST LM FORWARD] pos_hidden: {}",
                tensor_stats(&pos_hidden)
            );
            info!(
                "🔍 [FIRST LM FORWARD] neg_hidden: {}",
                tensor_stats(&neg_hidden)
            );
        }
        self.lm_forward_count += 1;

        Ok(DualLmOutput {
            pos_hidden,
            neg_hidden,
        })
    }

    /// Forward through both upper tts_language_models (20 layers).
    ///
    /// Processes combined text/speech sequence through both paths.
    /// Splices in hidden states from `forward_lm()` and adds type embeddings.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Token IDs for embedding lookup, shape `[batch, seq_len]`
    /// * `pos_lm_hidden` - Positive path hidden states from `forward_lm()`
    /// * `neg_lm_hidden` - Negative path hidden states from `forward_lm()`
    /// * `tts_text_masks` - Mask indicating text (1) vs speech (0) tokens
    /// * `pos_seqlen_offset` - Position offset for positive path KV cache
    /// * `neg_seqlen_offset` - Position offset for negative path KV cache
    ///
    /// # Returns
    ///
    /// `DualTtsLmOutput` with hidden states for diffusion conditioning.
    pub fn forward_tts_lm(
        &mut self,
        input_ids: &Tensor,
        pos_lm_hidden: &Tensor,
        neg_lm_hidden: &Tensor,
        tts_text_masks: &Tensor,
        pos_seqlen_offset: usize,
        neg_seqlen_offset: usize,
    ) -> Result<DualTtsLmOutput> {
        let embed_len = input_ids.dim(1)?;

        // Compute type embeddings (shared between both paths)
        let type_indices = tts_text_masks.to_dtype(DType::U32)?;
        let type_embeds = self.tts_input_types.forward(&type_indices)?;

        debug!(
            "[DualSplitLLM::forward_tts_lm] input_ids: {:?}, embed_len: {}, pos_lm_len: {}, neg_lm_len: {}, tts_masks: {:?}",
            input_ids.dims(),
            embed_len,
            pos_lm_hidden.dim(1)?,
            neg_lm_hidden.dim(1)?,
            tts_text_masks.dims()
        );

        // === Positive path: actual text embeddings ===
        let base_embeds = self.get_embeddings(input_ids)?;
        let mut pos_embeds = base_embeds;
        let pos_lm_len = pos_lm_hidden.dim(1)?;
        if pos_lm_len > 0 {
            let start_idx = embed_len.saturating_sub(pos_lm_len);
            pos_embeds = self.splice_hidden_states(&pos_embeds, pos_lm_hidden, start_idx)?;
        }
        let pos_embeds = pos_embeds.broadcast_add(&type_embeds)?;
        let pos_hidden = self.pos_tts_language_model.forward_from_embeds(
            &pos_embeds,
            pos_seqlen_offset,
            None,
        )?;

        // === Negative path: padding token embeddings (unconditional baseline) ===
        // This matches Python's behavior where neg_text_input_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        let neg_input_ids = Tensor::full(self.image_pad_token_id, input_ids.shape(), &self.device)?;
        let neg_base_embeds = self.get_embeddings(&neg_input_ids)?;
        let mut neg_embeds = neg_base_embeds;
        let neg_lm_len = neg_lm_hidden.dim(1)?;
        if neg_lm_len > 0 {
            let start_idx = embed_len.saturating_sub(neg_lm_len);
            neg_embeds = self.splice_hidden_states(&neg_embeds, neg_lm_hidden, start_idx)?;
        }
        let neg_embeds = neg_embeds.broadcast_add(&type_embeds)?;
        let neg_hidden = self.neg_tts_language_model.forward_from_embeds(
            &neg_embeds,
            neg_seqlen_offset,
            None,
        )?;

        debug!(
            "[DualSplitLLM::forward_tts_lm] DONE pos_hidden: {:?}, neg_hidden: {:?}",
            pos_hidden.dims(),
            neg_hidden.dims()
        );

        // Log stats on first call (matches Python's "FIRST TTS_LM FORWARD")
        if self.tts_lm_forward_count == 0 {
            info!(
                "🔍 [FIRST TTS_LM FORWARD] pos_hidden: {}",
                tensor_stats(&pos_hidden)
            );
            info!(
                "🔍 [FIRST TTS_LM FORWARD] neg_hidden: {}",
                tensor_stats(&neg_hidden)
            );
            // Also log last position (what Python calls "last_position")
            let seq_len = pos_hidden.dim(1)?;
            if seq_len > 0 {
                let pos_last = pos_hidden.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
                let neg_last = neg_hidden.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
                info!(
                    "🔍 [FIRST TTS_LM FORWARD] pos_last_position: {}",
                    tensor_stats(&pos_last)
                );
                info!(
                    "🔍 [FIRST TTS_LM FORWARD] neg_last_position: {}",
                    tensor_stats(&neg_last)
                );
            }
        }
        self.tts_lm_forward_count += 1;

        Ok(DualTtsLmOutput {
            pos_hidden,
            neg_hidden,
        })
    }

    /// Forward through both TTS LMs with acoustic embeddings (for speech token generation).
    ///
    /// Used during the speech generation loop when we have acoustic embeddings
    /// from the previous diffusion step instead of token embeddings.
    ///
    /// # Arguments
    ///
    /// * `acoustic_embed` - Acoustic embeddings from acoustic_connector, shape `[batch, 1, hidden_size]`
    /// * `tts_text_masks` - Should be zeros (indicating speech tokens)
    /// * `pos_seqlen_offset` - Position offset for positive path KV cache
    /// * `neg_seqlen_offset` - Position offset for negative path KV cache
    pub fn forward_tts_lm_with_acoustic(
        &mut self,
        acoustic_embed: &Tensor,
        tts_text_masks: &Tensor,
        pos_seqlen_offset: usize,
        neg_seqlen_offset: usize,
    ) -> Result<DualTtsLmOutput> {
        // Add type embeddings (tts_text_masks should be 0 for speech)
        let type_indices = tts_text_masks.to_dtype(DType::U32)?;
        let type_embeds = self.tts_input_types.forward(&type_indices)?;
        let inputs_embeds = acoustic_embed.broadcast_add(&type_embeds)?;

        // Forward through both TTS LMs
        let pos_hidden = self.pos_tts_language_model.forward_from_embeds(
            &inputs_embeds,
            pos_seqlen_offset,
            None,
        )?;
        let neg_hidden = self.neg_tts_language_model.forward_from_embeds(
            &inputs_embeds,
            neg_seqlen_offset,
            None,
        )?;

        debug!(
            "[DualSplitLLM::forward_tts_lm_with_acoustic] acoustic: {:?}, pos_offset: {}, neg_offset: {}, pos_hidden: {:?}, neg_hidden: {:?}",
            acoustic_embed.dims(),
            pos_seqlen_offset,
            neg_seqlen_offset,
            pos_hidden.dims(),
            neg_hidden.dims()
        );

        Ok(DualTtsLmOutput {
            pos_hidden,
            neg_hidden,
        })
    }

    /// Splice hidden states from LM into embedding tensor.
    ///
    /// Replaces `inputs_embeds[:, start_idx:start_idx+lm_len, :]` with `lm_hidden_states`.
    fn splice_hidden_states(
        &self,
        inputs_embeds: &Tensor,
        lm_hidden_states: &Tensor,
        start_idx: usize,
    ) -> Result<Tensor> {
        let (_batch, seq_len, _hidden) = inputs_embeds.dims3()?;
        let lm_len = lm_hidden_states.dim(1)?;

        debug!(
            "[splice] inputs_embeds: {:?}, lm_hidden: {:?}, start_idx: {}",
            inputs_embeds.dims(),
            lm_hidden_states.dims(),
            start_idx
        );

        if start_idx + lm_len > seq_len {
            return Err(anyhow!(
                "Splice out of bounds: start_idx={}, lm_len={}, seq_len={}",
                start_idx,
                lm_len,
                seq_len
            ));
        }

        // Replace entire sequence case
        if start_idx == 0 && lm_len == seq_len {
            return Ok(lm_hidden_states.clone());
        }

        // Split into [prefix, replaced, suffix]
        let prefix = if start_idx > 0 {
            Some(inputs_embeds.narrow(1, 0, start_idx)?)
        } else {
            None
        };

        let suffix_start = start_idx + lm_len;
        let suffix_len = seq_len - suffix_start;
        let suffix = if suffix_len > 0 {
            Some(inputs_embeds.narrow(1, suffix_start, suffix_len)?)
        } else {
            None
        };

        // Concatenate parts
        match (prefix, suffix) {
            (Some(p), Some(s)) => Tensor::cat(&[&p, lm_hidden_states, &s], 1).map_err(Into::into),
            (Some(p), None) => Tensor::cat(&[&p, lm_hidden_states], 1).map_err(Into::into),
            (None, Some(s)) => Tensor::cat(&[lm_hidden_states, &s], 1).map_err(Into::into),
            (None, None) => Ok(lm_hidden_states.clone()),
        }
    }

    // ==================== KV Cache Management ====================

    /// Extract KV caches from all 4 model components.
    pub fn extract_kv_caches(&self) -> DualKvCaches {
        DualKvCaches {
            pos_lm: self.pos_language_model.extract_kv_cache(),
            pos_tts_lm: self.pos_tts_language_model.extract_kv_cache(),
            neg_lm: self.neg_language_model.extract_kv_cache(),
            neg_tts_lm: self.neg_tts_language_model.extract_kv_cache(),
        }
    }

    /// Restore KV caches from a VoiceCache.
    ///
    /// Loads all 4 caches (positive and negative paths) from the voice cache.
    pub fn restore_from_voice_cache(&mut self, voice_cache: &VoiceCache) {
        let (pos_lm, pos_tts) = voice_cache.positive_caches();
        let (neg_lm, neg_tts) = voice_cache.negative_caches();

        self.pos_language_model.restore_kv_cache(pos_lm);
        self.pos_tts_language_model.restore_kv_cache(pos_tts);
        self.neg_language_model.restore_kv_cache(neg_lm);
        self.neg_tts_language_model.restore_kv_cache(neg_tts);
    }

    /// Log KV cache statistics for debugging.
    ///
    /// Logs tensor stats for all 4 caches (pos_lm, pos_tts_lm, neg_lm, neg_tts_lm).
    pub fn log_kv_cache_stats(&self) {
        let caches = self.extract_kv_caches();
        for (name, cache) in [
            ("pos_lm", &caches.pos_lm),
            ("pos_tts_lm", &caches.pos_tts_lm),
            ("neg_lm", &caches.neg_lm),
            ("neg_tts_lm", &caches.neg_tts_lm),
        ] {
            for (layer_idx, kv) in cache.iter().enumerate() {
                if let Some((k, v)) = kv {
                    debug!(
                        "[KV_CACHE] {}/layer_{}: key={}, value={}",
                        name,
                        layer_idx,
                        tensor_stats(k),
                        tensor_stats(v)
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_splice_logic() {
        // Test splice index calculation
        let embed_len = 10;
        let lm_len = 3;
        let start_idx = embed_len - lm_len; // 7

        assert_eq!(start_idx, 7);
        assert_eq!(start_idx + lm_len, embed_len);
    }

    #[test]
    fn test_type_mask_values() {
        // Type masks: 1 = text, 0 = speech
        let text_mask = 1u32;
        let speech_mask = 0u32;

        assert_eq!(text_mask, 1);
        assert_eq!(speech_mask, 0);
    }
}
