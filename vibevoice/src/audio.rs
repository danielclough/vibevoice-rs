//! Audio data container with convenience methods.

use crate::{Result, VibeVoiceError};
use candle_core::{DType, Tensor};
use std::path::Path;

/// Audio data container with convenience methods.
#[derive(Clone)]
pub struct AudioData {
    /// Raw audio samples (normalized -1.0 to 1.0)
    samples: Vec<f32>,
    /// WavSample rate in Hz (typically 24000)
    sample_rate: u32,
}

impl AudioData {
    /// Create AudioData from a tensor.
    pub fn from_tensor(tensor: &Tensor, sample_rate: u32) -> Result<Self> {
        let samples = tensor
            .to_dtype(DType::F32)
            .map_err(|e| VibeVoiceError::AudioError(e.to_string()))?
            .contiguous()
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

    /// Encode samples as raw PCM bytes (16-bit signed int, little-endian).
    ///
    /// Use this for streaming - send header separately via `wav_header()`.
    pub fn to_pcm_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.samples.len() * 2);
        for &sample in &self.samples {
            let amplitude = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            bytes.extend_from_slice(&amplitude.to_le_bytes());
        }
        bytes
    }

    /// Generate a WAV header for streaming.
    ///
    /// If `total_samples` is None, uses max value to indicate unknown length (streaming).
    /// Browser clients should update the header size fields after receiving all data.
    pub fn wav_header_for_streaming(sample_rate: u32, total_samples: Option<usize>) -> Vec<u8> {
        let num_channels: u16 = 1;
        let bits_per_sample: u16 = 16;
        let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
        let block_align = num_channels * bits_per_sample / 8;

        // Data size: if unknown, use max u32 - 36 (header overhead)
        let data_size: u32 = total_samples
            .map(|n| (n * 2) as u32)
            .unwrap_or(u32::MAX - 36);
        let file_size = data_size + 36; // 44 - 8 (RIFF header)

        let mut header = Vec::with_capacity(44);

        // RIFF header
        header.extend_from_slice(b"RIFF");
        header.extend_from_slice(&file_size.to_le_bytes());
        header.extend_from_slice(b"WAVE");

        // fmt chunk
        header.extend_from_slice(b"fmt ");
        header.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        header.extend_from_slice(&1u16.to_le_bytes());  // PCM format
        header.extend_from_slice(&num_channels.to_le_bytes());
        header.extend_from_slice(&sample_rate.to_le_bytes());
        header.extend_from_slice(&byte_rate.to_le_bytes());
        header.extend_from_slice(&block_align.to_le_bytes());
        header.extend_from_slice(&bits_per_sample.to_le_bytes());

        // data chunk
        header.extend_from_slice(b"data");
        header.extend_from_slice(&data_size.to_le_bytes());

        header
    }

    /// Encode audio as WAV bytes (for HTTP binary response).
    pub fn to_wav_bytes(&self) -> Result<Vec<u8>> {
        let mut buffer = std::io::Cursor::new(Vec::new());
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::new(&mut buffer, spec)
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
        Ok(buffer.into_inner())
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
                    "WavSample rate mismatch: {} vs {}",
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
