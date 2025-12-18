//! Audio data container with convenience methods.

use crate::{Result, VibeVoiceError};
use candle_core::{DType, Tensor};
use std::path::Path;

/// Audio data container with convenience methods.
#[derive(Clone)]
pub struct AudioData {
    /// Raw audio samples (normalized -1.0 to 1.0)
    samples: Vec<f32>,
    /// Sample rate in Hz (typically 24000)
    sample_rate: u32,
}

impl AudioData {
    /// Create AudioData from a tensor.
    pub fn from_tensor(tensor: &Tensor, sample_rate: u32) -> Result<Self> {
        let samples = tensor
            .to_dtype(DType::F32)
            .map_err(|e| VibeVoiceError::AudioError(e.to_string()))?
            .flatten_all()
            .map_err(|e| VibeVoiceError::AudioError(e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e| VibeVoiceError::AudioError(e.to_string()))?;

        Ok(Self {
            samples,
            sample_rate,
        })
    }

    /// Create AudioData from raw samples.
    pub fn from_samples(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
        }
    }

    /// Get raw audio samples.
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Get sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get duration in seconds.
    pub fn duration_secs(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }

    /// Get number of samples.
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }

    /// Save audio to WAV file.
    pub fn save_wav(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(path, spec)
            .map_err(|e| VibeVoiceError::IoError(e.to_string()))?;

        for &sample in &self.samples {
            let amplitude = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            writer
                .write_sample(amplitude)
                .map_err(|e| VibeVoiceError::IoError(e.to_string()))?;
        }

        writer
            .finalize()
            .map_err(|e| VibeVoiceError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Concatenate multiple AudioData together.
    pub fn concat(chunks: &[AudioData]) -> Result<Self> {
        if chunks.is_empty() {
            return Err(VibeVoiceError::AudioError(
                "No audio chunks to concatenate".to_string(),
            ));
        }

        let sample_rate = chunks[0].sample_rate;

        // Verify all chunks have same sample rate
        for chunk in chunks {
            if chunk.sample_rate != sample_rate {
                return Err(VibeVoiceError::AudioError(format!(
                    "Sample rate mismatch: {} vs {}",
                    chunk.sample_rate, sample_rate
                )));
            }
        }

        let total_samples: usize = chunks.iter().map(|c| c.samples.len()).sum();
        let mut samples = Vec::with_capacity(total_samples);

        for chunk in chunks {
            samples.extend_from_slice(&chunk.samples);
        }

        Ok(Self {
            samples,
            sample_rate,
        })
    }
}

impl std::fmt::Debug for AudioData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioData")
            .field("samples", &self.samples.len())
            .field("sample_rate", &self.sample_rate)
            .field("duration_secs", &self.duration_secs())
            .finish()
    }
}
