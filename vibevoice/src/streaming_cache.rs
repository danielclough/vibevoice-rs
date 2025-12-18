use anyhow::Result;
use candle_core::{Device, Tensor};
use std::collections::HashMap;

/// Cache for streaming convolution, similar to KV cache in attention
/// Stores intermediate states for each conv layer to enable incremental processing
#[derive(Clone, Debug)]
pub struct StreamingCache {
    /// Map from layer_id to cached tensor
    /// Format: layer_id → cached_state [batch, channels, time]
    cache: HashMap<String, Tensor>,
    device: Device,
    /// Number of tokens processed (for debug logging)
    tokens_processed: usize,
    /// Number of warmup tokens (used for debug logging only)
    /// Note: The cache's zero-initialization provides natural silence ramp-up
    warmup_tokens: usize,
}

impl StreamingCache {
    pub fn new(device: Device) -> Self {
        Self {
            cache: HashMap::new(),
            device,
            tokens_processed: 0,
            warmup_tokens: 6, // Used for debug logging only
        }
    }

    /// Check if we're still in the early tokens (for debug logging)
    /// Note: This doesn't affect audio output - the cache's zero-initialization
    /// provides natural silence ramp-up (same as Python)
    pub fn in_warmup(&self) -> bool {
        self.tokens_processed < self.warmup_tokens
    }

    /// Increment token counter (call once per decode)
    pub fn increment_token_count(&mut self) {
        self.tokens_processed += 1;
    }

    /// Get current token count
    pub fn tokens_processed(&self) -> usize {
        self.tokens_processed
    }

    /// Get cached state for a given layer
    pub fn get(&self, layer_id: &str) -> Option<&Tensor> {
        self.cache.get(layer_id)
    }

    /// Set cached state for a given layer
    pub fn set(&mut self, layer_id: String, state: Tensor) -> Result<()> {
        self.cache.insert(layer_id, state);
        Ok(())
    }

    /// Clear all cached states
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Reset all cached states to zero (matching Python's set_to_zero)
    /// Keeps cache structure but zeros tensor values
    /// This is used when SPEECH_END is encountered to prepare for potential continuation
    pub fn reset_to_zero(&mut self) {
        for (_key, tensor) in self.cache.iter_mut() {
            if let Ok(zeros) = Tensor::zeros(tensor.dims(), tensor.dtype(), tensor.device()) {
                let _ = tensor.device().synchronize();
                *tensor = zeros;
            }
        }
        // Note: Do NOT reset tokens_processed - Python doesn't reset its counter
    }

    /// Clear cached state for a specific layer
    pub fn clear_layer(&mut self, layer_id: &str) {
        self.cache.remove(layer_id);
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the number of cache entries (for diagnostics)
    pub fn cache_entry_count(&self) -> usize {
        self.cache.len()
    }
}
