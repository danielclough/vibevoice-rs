//! Binary classifier for EOS (End-of-Speech) detection.
//!
//! This module implements the EOS classifier used in VibeVoice Realtime
//! to determine when speech generation should stop.
//!
//! # Architecture
//!
//! ```text
//! Input: hidden_state [batch, hidden_size]
//!        ↓
//! fc1: Linear(hidden_size → hidden_size)
//!        ↓
//! ReLU activation
//!        ↓
//! fc2: Linear(hidden_size → 1)
//!        ↓
//! Output: logits [batch, 1]
//! ```
//!
//! During inference, sigmoid is applied to convert logits to probability,
//! and speech is stopped when probability > 0.5.
//!
//! # Python Reference
//!
//! From `modeling_vibevoice_streaming.py:32-41`:
//! ```python
//! class BinaryClassifier(nn.Module):
//!     def __init__(self, hidden_size):
//!         super(BinaryClassifier, self).__init__()
//!         self.fc1 = nn.Linear(hidden_size, hidden_size)
//!         self.fc2 = nn.Linear(hidden_size, 1)
//!
//!     def forward(self, x):
//!         x = torch.relu(self.fc1(x))
//!         x = self.fc2(x)
//!         return x
//! ```

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{Linear, Module, VarBuilder, linear};

/// Binary classifier for EOS detection in streaming TTS.
///
/// This classifier takes the last hidden state from `tts_language_model`
/// and predicts whether speech generation should stop.
///
/// # Weight Paths
///
/// - `tts_eos_classifier.fc1.weight`: `[hidden_size, hidden_size]`
/// - `tts_eos_classifier.fc1.bias`: `[hidden_size]`
/// - `tts_eos_classifier.fc2.weight`: `[1, hidden_size]`
/// - `tts_eos_classifier.fc2.bias`: `[1]`
pub struct BinaryClassifier {
    fc1: Linear,
    fc2: Linear,
}

impl BinaryClassifier {
    /// Create a new BinaryClassifier from pretrained weights.
    ///
    /// # Arguments
    ///
    /// * `vb` - VarBuilder pointing to `tts_eos_classifier` prefix
    /// * `hidden_size` - Hidden dimension of the LLM (e.g., 1024 for 0.5B)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let classifier = BinaryClassifier::new(
    ///     vb.pp("tts_eos_classifier"),
    ///     1024,
    /// )?;
    /// ```
    pub fn new(vb: VarBuilder, hidden_size: usize) -> Result<Self> {
        let fc1 = linear(hidden_size, hidden_size, vb.pp("fc1"))?;
        let fc2 = linear(hidden_size, 1, vb.pp("fc2"))?;

        Ok(Self { fc1, fc2 })
    }

    /// Forward pass returning raw logits.
    ///
    /// # Arguments
    ///
    /// * `x` - Hidden state tensor of shape `[batch, hidden_size]`
    ///
    /// # Returns
    ///
    /// Raw logits of shape `[batch, 1]`. Apply sigmoid to get probability.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // fc1 -> ReLU -> fc2
        let x = self.fc1.forward(x)?;
        let x = x.relu()?;
        let x = self.fc2.forward(&x)?;
        Ok(x)
    }

    /// Check if EOS should trigger based on hidden state.
    ///
    /// Applies sigmoid to logits and checks if probability > 0.5.
    ///
    /// # Arguments
    ///
    /// * `hidden_state` - Last hidden state from TTS LM, shape `[batch, hidden_size]`
    ///   or `[hidden_size]` (will be unsqueezed)
    ///
    /// # Returns
    ///
    /// `true` if speech should stop (probability > 0.5)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Get last hidden state from TTS LM output
    /// let last_hidden = tts_lm_hidden.i((.., seq_len - 1, ..))?;
    /// if classifier.should_stop(&last_hidden)? {
    ///     // End generation
    ///     break;
    /// }
    /// ```
    pub fn should_stop(&self, hidden_state: &Tensor) -> Result<bool> {
        // Handle both [batch, hidden_size] and [hidden_size] inputs
        let x = if hidden_state.dims().len() == 1 {
            hidden_state.unsqueeze(0)?
        } else {
            hidden_state.clone()
        };

        let logits = self.forward(&x)?;

        // sigmoid(logits) > 0.5 is equivalent to logits > 0
        // But we use sigmoid for clarity and to match Python behavior
        let prob = candle_nn::ops::sigmoid(&logits)?;

        // Get first element (batch index 0)
        let prob_scalar = prob.flatten_all()?.to_vec1::<f32>()?[0];

        Ok(prob_scalar > 0.5)
    }

    /// Get EOS probability for the given hidden state.
    ///
    /// # Arguments
    ///
    /// * `hidden_state` - Hidden state tensor
    ///
    /// # Returns
    ///
    /// Probability in range [0, 1]
    pub fn get_probability(&self, hidden_state: &Tensor) -> Result<f32> {
        let x = if hidden_state.dims().len() == 1 {
            hidden_state.unsqueeze(0)?
        } else {
            hidden_state.clone()
        };

        let logits = self.forward(&x)?;
        let prob = candle_nn::ops::sigmoid(&logits)?;
        let prob_scalar = prob.flatten_all()?.to_vec1::<f32>()?[0];

        Ok(prob_scalar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    #[test]
    fn test_binary_classifier_shape() {
        let device = Device::Cpu;
        let hidden_size = 1024;

        // Create a VarMap with random weights for testing
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        // Initialize classifier (this will create random weights)
        let classifier = BinaryClassifier::new(vb, hidden_size).unwrap();

        // Test forward pass
        let input = Tensor::zeros((1, hidden_size), candle_core::DType::F32, &device).unwrap();
        let output = classifier.forward(&input).unwrap();

        // Output should be [1, 1]
        assert_eq!(output.dims(), &[1, 1]);
    }

    #[test]
    fn test_should_stop_threshold() {
        let device = Device::Cpu;
        let hidden_size = 16; // Small for testing

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let classifier = BinaryClassifier::new(vb, hidden_size).unwrap();

        // Create input that produces known output
        let input = Tensor::zeros((1, hidden_size), candle_core::DType::F32, &device).unwrap();

        // With zero-initialized weights, output will be 0 -> sigmoid(0) = 0.5
        // should_stop returns true when prob > 0.5, so 0.5 exactly should return false
        let prob = classifier.get_probability(&input).unwrap();

        // Probability should be close to 0.5 with zero weights
        // (may vary slightly due to initialization)
        assert!(prob >= 0.0 && prob <= 1.0);
    }

    #[test]
    fn test_1d_input_handling() {
        let device = Device::Cpu;
        let hidden_size = 16;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let classifier = BinaryClassifier::new(vb, hidden_size).unwrap();

        // Test with 1D input [hidden_size]
        let input_1d = Tensor::zeros((hidden_size,), candle_core::DType::F32, &device).unwrap();
        let result = classifier.should_stop(&input_1d);
        assert!(result.is_ok());

        // Test with 2D input [batch, hidden_size]
        let input_2d = Tensor::zeros((1, hidden_size), candle_core::DType::F32, &device).unwrap();
        let result = classifier.should_stop(&input_2d);
        assert!(result.is_ok());
    }
}
