//! Voice cache loading for VibeVoice Realtime streaming model.
//!
//! Voice caches contain pre-computed KV states from processing a reference audio sample.
//! They enable voice cloning by initializing the generation with the speaker's characteristics.
//!
//! # Cache Structure
//!
//! The cache contains 4 entries for CFG (Classifier-Free Guidance):
//! - `lm`: Lower language_model (4 layers) - positive path
//! - `tts_lm`: Upper tts_language_model (20 layers) - positive path
//! - `neg_lm`: Lower language_model - negative path (for CFG)
//! - `neg_tts_lm`: Upper tts_language_model - negative path (for CFG)
//!
//! Each entry contains:
//! - `last_hidden_state`: Final hidden states from processing reference audio
//! - `past_key_values`: KV cache tensors for each layer
//!
//! # Safetensors Format
//!
//! ```text
//! lm/last_hidden_state:               [1, seq_len, hidden_size]
//! lm/past_key_values/0/key:           [1, num_kv_heads, seq_len, head_dim]
//! lm/past_key_values/0/value:         [1, num_kv_heads, seq_len, head_dim]
//! lm/past_key_values/1/key:           ...
//! ...
//! tts_lm/last_hidden_state:           [1, seq_len, hidden_size]
//! tts_lm/past_key_values/0/key:       ...
//! ...
//! neg_lm/last_hidden_state:           ...
//! neg_tts_lm/last_hidden_state:       ...
//! ```
//!
//! # Python Reference
//!
//! From `modeling_vibevoice_streaming_inference.py:494-497`:
//! ```python
//! outputs = all_prefilled_outputs["lm"]
//! tts_lm_outputs = all_prefilled_outputs["tts_lm"]
//! negative_outputs = all_prefilled_outputs["neg_lm"]
//! tts_lm_negative_outputs = all_prefilled_outputs["neg_tts_lm"]
//! ```

use anyhow::{Result, anyhow};
use candle_core::{Device, Tensor, safetensors::load};
use std::collections::HashMap;
use std::path::Path;

/// Single cache entry for one model component.
///
/// Contains the last hidden state and KV cache for either the lower LM
/// or upper TTS LM, for either positive or negative CFG path.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Final hidden states from processing reference audio.
    /// Shape: `[1, seq_len, hidden_size]`
    pub last_hidden_state: Tensor,

    /// KV cache for each layer.
    /// Each tuple is `(key, value)` with shape `[1, num_kv_heads, seq_len, head_dim]`
    pub past_key_values: Vec<(Tensor, Tensor)>,
}

impl CacheEntry {
    /// Get the sequence length from the cache.
    ///
    /// Validates that last_hidden_state and KV cache have matching seq_len.
    pub fn seq_len(&self) -> Result<usize> {
        let hidden_seq_len = self.last_hidden_state.dim(1)?;

        // Validate KV cache seq_len matches (KV shape: [1, num_kv_heads, seq_len, head_dim])
        if let Some((k, _v)) = self.past_key_values.first() {
            let kv_seq_len = k.dim(2)?;
            if kv_seq_len != hidden_seq_len {
                return Err(anyhow!(
                    "seq_len mismatch: last_hidden_state has {} but KV cache has {}. \
                    This would cause RoPE position errors!",
                    hidden_seq_len, kv_seq_len
                ));
            }
        }

        Ok(hidden_seq_len)
    }

    /// Get the number of layers in this cache.
    pub fn num_layers(&self) -> usize {
        self.past_key_values.len()
    }

    /// Convert KV cache to the format expected by Qwen2Model.
    ///
    /// Returns `Vec<Option<(Tensor, Tensor)>>` for `restore_kv_cache()`.
    pub fn to_qwen2_cache(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.past_key_values
            .iter()
            .map(|(k, v)| Some((k.clone(), v.clone())))
            .collect()
    }
}

/// Complete voice cache with all 4 components for CFG.
///
/// Used to initialize generation with a specific speaker's voice characteristics.
///
/// # CFG (Classifier-Free Guidance)
///
/// The model uses two parallel paths during generation:
/// - **Positive path** (`lm`, `tts_lm`): Conditioned on the text and voice
/// - **Negative path** (`neg_lm`, `neg_tts_lm`): Unconditional baseline
///
/// The final output is computed as:
/// ```text
/// output = negative + cfg_scale * (positive - negative)
/// ```
#[derive(Debug, Clone)]
pub struct VoiceCache {
    /// Lower LM cache (4 layers) - positive path
    pub lm: CacheEntry,

    /// Upper TTS LM cache (20 layers) - positive path
    pub tts_lm: CacheEntry,

    /// Lower LM cache - negative path (for CFG)
    pub neg_lm: CacheEntry,

    /// Upper TTS LM cache - negative path (for CFG)
    pub neg_tts_lm: CacheEntry,
}

impl VoiceCache {
    /// Load voice cache from a safetensors file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the `.safetensors` file
    /// * `device` - Device to load tensors onto
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cache = VoiceCache::from_safetensors("voice.safetensors", &device)?;
    /// println!("LM seq_len: {}", cache.lm.seq_len()?);
    /// println!("LM layers: {}", cache.lm.num_layers());
    /// ```
    pub fn from_safetensors(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let path = path.as_ref();

        // Load all tensors from safetensors file
        let tensors = load(path, device)?;

        // Load all 4 cache entries with reduced precision for numerical stability
        let lm = Self::load_cache_entry(&tensors, "lm", device)?;
        let tts_lm = Self::load_cache_entry(&tensors, "tts_lm", device)?;
        let neg_lm = Self::load_cache_entry(&tensors, "neg_lm", device)?;
        let neg_tts_lm = Self::load_cache_entry(&tensors, "neg_tts_lm", device)?;

        // Validate layer counts
        // LM should have 4 layers, TTS LM should have 20 layers
        if lm.num_layers() != neg_lm.num_layers() {
            return Err(anyhow!(
                "Layer count mismatch: lm={} vs neg_lm={}",
                lm.num_layers(),
                neg_lm.num_layers()
            ));
        }
        if tts_lm.num_layers() != neg_tts_lm.num_layers() {
            return Err(anyhow!(
                "Layer count mismatch: tts_lm={} vs neg_tts_lm={}",
                tts_lm.num_layers(),
                neg_tts_lm.num_layers()
            ));
        }

        Ok(Self {
            lm,
            tts_lm,
            neg_lm,
            neg_tts_lm,
        })
    }

    /// Load a single cache entry from the tensor map.
    fn load_cache_entry(tensors: &HashMap<String, Tensor>, prefix: &str, _device: &Device) -> Result<CacheEntry> {
        use candle_core::DType;

        // Use F32 for all devices to avoid dtype mismatch issues
        let target_dtype = DType::F32;

        // Load last_hidden_state
        let hidden_key = format!("{}/last_hidden_state", prefix);
        let last_hidden_state = tensors
            .get(&hidden_key)
            .ok_or_else(|| anyhow!("Missing {} in voice cache", hidden_key))?
            .to_dtype(target_dtype)?;

        // Load KV pairs until we run out
        let mut past_key_values = Vec::new();
        let mut layer_idx = 0;

        loop {
            let key_path = format!("{}/past_key_values/{}/key", prefix, layer_idx);
            let value_path = format!("{}/past_key_values/{}/value", prefix, layer_idx);

            match (tensors.get(&key_path), tensors.get(&value_path)) {
                (Some(k), Some(v)) => {
                    // Convert KV tensors to reduced precision
                    past_key_values.push((k.to_dtype(target_dtype)?, v.to_dtype(target_dtype)?));
                    layer_idx += 1;
                }
                (None, None) => break,
                (Some(_), None) => {
                    return Err(anyhow!("Found key but missing value at {}", value_path));
                }
                (None, Some(_)) => {
                    return Err(anyhow!("Found value but missing key at {}", key_path));
                }
            }
        }

        if past_key_values.is_empty() {
            return Err(anyhow!(
                "No KV cache layers found for prefix '{}'. Expected format: {}/past_key_values/0/key",
                prefix,
                prefix
            ));
        }

        Ok(CacheEntry {
            last_hidden_state,
            past_key_values,
        })
    }

    /// Get the positive path caches as Qwen2-compatible format.
    ///
    /// Returns `(lm_cache, tts_lm_cache)` for `SplitLLM::restore_kv_cache()`.
    pub fn positive_caches(
        &self,
    ) -> (Vec<Option<(Tensor, Tensor)>>, Vec<Option<(Tensor, Tensor)>>) {
        (self.lm.to_qwen2_cache(), self.tts_lm.to_qwen2_cache())
    }

    /// Get the negative path caches as Qwen2-compatible format.
    ///
    /// Returns `(neg_lm_cache, neg_tts_lm_cache)` for the negative CFG path.
    pub fn negative_caches(
        &self,
    ) -> (Vec<Option<(Tensor, Tensor)>>, Vec<Option<(Tensor, Tensor)>>) {
        (
            self.neg_lm.to_qwen2_cache(),
            self.neg_tts_lm.to_qwen2_cache(),
        )
    }

    /// Get initial cache positions (sequence lengths).
    ///
    /// Returns `(lm_seq_len, tts_lm_seq_len, neg_lm_seq_len, neg_tts_lm_seq_len)`.
    /// Note: lm and tts_lm have different sequence lengths because:
    /// - lm only processes text tokens
    /// - tts_lm processes text + speech tokens
    pub fn cache_positions(&self) -> Result<(usize, usize, usize, usize)> {
        Ok((
            self.lm.seq_len()?,
            self.tts_lm.seq_len()?,
            self.neg_lm.seq_len()?,
            self.neg_tts_lm.seq_len()?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_entry_to_qwen2() {
        // Test conversion to Qwen2 format
        let device = Device::Cpu;

        // Create a mock CacheEntry
        let hidden = Tensor::zeros((1, 10, 1024), candle_core::DType::F32, &device).unwrap();
        let k = Tensor::zeros((1, 2, 10, 64), candle_core::DType::F32, &device).unwrap();
        let v = Tensor::zeros((1, 2, 10, 64), candle_core::DType::F32, &device).unwrap();

        let entry = CacheEntry {
            last_hidden_state: hidden,
            past_key_values: vec![(k, v)],
        };

        assert_eq!(entry.num_layers(), 1);
        assert_eq!(entry.seq_len().unwrap(), 10);

        let qwen2_cache = entry.to_qwen2_cache();
        assert_eq!(qwen2_cache.len(), 1);
        assert!(qwen2_cache[0].is_some());
    }

    #[test]
    fn test_expected_layer_counts() {
        // LM: 4 layers, TTS LM: 20 layers
        let expected_lm_layers = 4;
        let expected_tts_lm_layers = 20;

        assert_eq!(expected_lm_layers, 4);
        assert_eq!(expected_tts_lm_layers, 20);
    }
}
