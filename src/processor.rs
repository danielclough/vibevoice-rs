use crate::utils::{load_audio_wav, normalize_audio_db_fs};
use candle_core::{Device, Result, Tensor};
use regex::Regex;
use serde_json;
use std::path::{Path, PathBuf};
use tokenizers::{AddedToken, Tokenizer};
use tracing::{debug, warn};

/// Normalize typographic characters to ASCII equivalents for tokenization.
/// This ensures parity with Python's text handling, preventing bad audio
/// output when text contains smart/curly quotes from word processors.
fn normalize_text(text: &str) -> String {
    text.replace(['\u{2019}', '\u{2018}'], "'") // LEFT SINGLE QUOTATION MARK → apostrophe
        .replace(['\u{201C}', '\u{201D}'], "\"") // RIGHT DOUBLE QUOTATION MARK → quote
        .replace('\u{2014}', "--") // EM DASH → double hyphen
        .replace('\u{2013}', "-") // EN DASH → hyphen
        .replace('\u{2026}', "...") // HORIZONTAL ELLIPSIS → three periods
}

/// Output from the VibeVoiceProcessor.process() method
pub struct ProcessorOutput {
    pub input_ids: Tensor,
    pub attention_mask: Option<Tensor>,
    pub voice_embeds: Option<Tensor>,
    pub voice_masks: Option<Tensor>, // Attention masks for voice samples
    pub speech_input_mask: Option<Tensor>, // Boolean mask marking voice injection positions
    /// Per-speaker VAE token validity masks: [num_speakers][max_vae_tokens]
    /// True = valid token, False = padding (for speakers with shorter audio)
    /// Used to filter padded tokens before injection, matching Python's speech_masks
    pub speech_masks: Option<Vec<Vec<bool>>>,
}

/// VibeVoice processor for text and voice sample processing
/// Matches Python VibeVoiceProcessor API
pub struct VibeVoiceProcessor {
    pub tokenizer: Tokenizer,
    pub sample_rate: u32,
    pub device: Device,
    pub speech_tok_compress_ratio: usize,
    pub db_normalize: bool,
}

impl VibeVoiceProcessor {
    /// Load processor from pretrained model path
    pub fn from_pretrained(model_path: &Path, device: &Device) -> Result<Self> {
        debug!("📦 Loading VibeVoice Processor");

        // Determine tokenizer source from preprocessor_config.json
        let tokenizer_repo_id =
            match std::fs::read_to_string(model_path.join("preprocessor_config.json")) {
                Ok(content) => {
                    let config: serde_json::Value =
                        serde_json::from_str(&content).map_err(|e| {
                            candle_core::Error::Msg(format!(
                                "Failed to parse preprocessor_config.json: {}",
                                e
                            ))
                        })?;
                    config
                        .get("language_model_pretrained_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Qwen/Qwen2.5-1.5B")
                        .to_string()
                }
                Err(_) => "Qwen/Qwen2.5-1.5B".to_string(),
            };

        // Download tokenizer from the language model repo
        let api = candle_hf_hub::api::sync::Api::new()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create HF API: {}", e)))?;
        let tokenizer_repo = api.model(tokenizer_repo_id.clone());
        let tokenizer_path = tokenizer_repo.get("tokenizer.json").map_err(|e| {
            candle_core::Error::Msg(format!(
                "Failed to download tokenizer from {}: {}",
                tokenizer_repo_id, e
            ))
        })?;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))?;

        // Add VibeVoice-specific special tokens (matching Python implementation)
        let special_tokens = vec![
            "<|vision_start|>".to_string(), // Speech start
            "<|vision_end|>".to_string(),   // Speech end
            "<|vision_pad|>".to_string(),   // Speech diffusion pad
        ];

        for token in special_tokens {
            if tokenizer.token_to_id(&token).is_none() {
                // Add token if it doesn't exist
                let added_token = AddedToken::from(token.clone(), false);
                tokenizer.add_special_tokens(&[added_token]);
                debug!("Added special token: {}", token);
            } else {
                debug!("Special token already exists: {}", token);
            }
        }

        debug!("✓ Tokenizer loaded with special tokens");

        // Try to read preprocessor_config.json for configuration
        let (speech_tok_compress_ratio, db_normalize) =
            match std::fs::read_to_string(model_path.join("preprocessor_config.json")) {
                Ok(content) => {
                    let config: serde_json::Value =
                        serde_json::from_str(&content).map_err(|e| {
                            candle_core::Error::Msg(format!(
                                "Failed to parse preprocessor_config.json: {}",
                                e
                            ))
                        })?;
                    let compress_ratio = config
                        .get("speech_tok_compress_ratio")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(3200) as usize;
                    let db_norm = config
                        .get("db_normalize")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(true);
                    (compress_ratio, db_norm)
                }
                Err(_) => {
                    // Default values matching Python
                    (3200, true)
                }
            };

        Ok(Self {
            tokenizer,
            sample_rate: 24000, // VibeVoice uses 24kHz
            device: device.clone(),
            speech_tok_compress_ratio,
            db_normalize,
        })
    }

    /// Process text and voice samples into model inputs
    /// Matches Python: processor(text=[...], voice_samples=[...], padding=True, truncation=True, max_length=2048, return_attention_mask=True)
    pub fn process(
        &self,
        text: Vec<String>,
        voice_samples: Option<Vec<Vec<PathBuf>>>,
        padding: bool,
        return_attention_mask: bool,
    ) -> Result<ProcessorOutput> {
        self.process_with_options(
            text,
            voice_samples,
            padding,
            false,
            None,
            return_attention_mask,
        )
    }
    pub fn parse_script(&self, script: &str) -> Result<ParsedScript> {
        // Normalize typographic characters (smart quotes, em dashes, etc.) to ASCII
        // to ensure identical tokenization with Python
        let script = normalize_text(script);

        let speaker_pattern = Regex::new(r"^Speaker\s+(\d+)\s*:\s*(.*)$")
            .map_err(|e| candle_core::Error::Msg(format!("Regex error: {}", e)))?;

        let lines: Vec<&str> = script.trim().split('\n').collect();
        let mut parsed_lines = Vec::new();
        let mut speaker_ids = Vec::new();

        // First pass: parse all lines and collect speaker IDs
        for line in lines {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Some(captures) = speaker_pattern.captures(line) {
                let speaker_id: usize =
                    captures.get(1).unwrap().as_str().parse().map_err(|e| {
                        candle_core::Error::Msg(format!("Invalid speaker ID: {}", e))
                    })?;

                let text = captures.get(2).unwrap().as_str().trim();

                // Add leading space to text (matching Python: text = ' ' + match.group(2).strip())
                let text_with_space = format!(" {}", text);

                parsed_lines.push((speaker_id, text_with_space));
                speaker_ids.push(speaker_id);
            } else {
                warn!("Could not parse line: '{}'", line);
            }
        }

        if parsed_lines.is_empty() {
            return Err(candle_core::Error::Msg(
                "No valid speaker lines found in script".to_string(),
            ));
        }

        // Check if we need to normalize speaker IDs (only if all are > 0)
        // This matches Python's normalization logic
        let min_speaker_id = *speaker_ids.iter().min().unwrap();

        let normalized_lines = if min_speaker_id > 0 {
            // Normalize to start from 0
            debug!(
                "Normalizing speaker IDs: min={}, subtracting 1 from all IDs",
                min_speaker_id
            );
            parsed_lines
                .into_iter()
                .map(|(id, text)| (id - 1, text))
                .collect()
        } else {
            // Keep original IDs
            parsed_lines
        };

        // Get unique speakers
        let unique_speakers: Vec<usize> = {
            let mut speakers: Vec<usize> = normalized_lines.iter().map(|(id, _)| *id).collect();
            speakers.sort();
            speakers.dedup();
            speakers
        };

        Ok(ParsedScript {
            lines: normalized_lines,
            unique_speakers,
        })
    }

    /// Process with full options (padding, truncation, max_length)
    /// Creates integrated token sequence with voice injection masks like Python
    pub fn process_with_options(
        &self,
        text: Vec<String>,
        voice_samples: Option<Vec<Vec<PathBuf>>>,
        _padding: bool,
        _truncation: bool,
        _max_length: Option<usize>,
        return_attention_mask: bool,
    ) -> Result<ProcessorOutput> {
        debug!("🔄 Processing integrated sequence with voice injection");

        if text.len() != 1 {
            return Err(candle_core::Error::Msg(format!(
                "Integrated processing currently supports only single input, got {}",
                text.len()
            )));
        }

        let text_input = &text[0];

        // Parse the script to extract and normalize speaker IDs
        let parsed_script = self.parse_script(text_input)?;

        debug!(
            "📝 Parsed {} speaker lines with {} unique speakers",
            parsed_script.lines.len(),
            parsed_script.unique_speakers.len()
        );

        // Create system prompt with leading space (matches Python)
        let system_prompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n";
        let mut full_tokens = self
            .tokenizer
            .encode(system_prompt, false)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to encode system prompt: {}", e)))?
            .get_ids()
            .to_vec();
        let mut speech_input_mask = vec![false; full_tokens.len()];

        // Process voice samples if provided
        let (voice_embeds, speech_masks) = if let Some(samples) = voice_samples {
            if samples.len() != 1 {
                return Err(candle_core::Error::Msg(format!(
                    "Voice samples should be a single vec for integrated processing, got {} groups",
                    samples.len()
                )));
            }

            debug!("🎤 Creating voice prompt...");
            // Pass the number of unique speakers so voice prompt uses normalized IDs
            let num_speakers = parsed_script.unique_speakers.len().min(samples[0].len());
            let (voice_tokens, voice_audio, voice_mask, speaker_speech_masks) =
                self.create_voice_prompt(&samples[0][..num_speakers])?;

            full_tokens.extend(voice_tokens);
            speech_input_mask.extend(voice_mask);

            (Some(voice_audio), Some(speaker_speech_masks))
        } else {
            (None, None)
        };

        // Add text input section
        let text_prefix = " Text input:\n";
        let text_prefix_tokens = self
            .tokenizer
            .encode(text_prefix, false)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to encode text prefix: {}", e)))?
            .get_ids()
            .to_vec();
        let text_prefix_len = text_prefix_tokens.len();
        full_tokens.extend(text_prefix_tokens);
        speech_input_mask.extend(vec![false; text_prefix_len]);

        // Add the parsed speaker lines with normalized IDs
        for (speaker_id, speaker_text) in &parsed_script.lines {
            let speaker_line = format!(" Speaker {}:{}\n", speaker_id, speaker_text);
            let line_tokens = self
                .tokenizer
                .encode(speaker_line.as_str(), false)
                .map_err(|e| {
                    candle_core::Error::Msg(format!("Failed to encode speaker line: {}", e))
                })?
                .get_ids()
                .to_vec();
            let line_len = line_tokens.len();

            full_tokens.extend(line_tokens);
            speech_input_mask.extend(vec![false; line_len]);
        }

        // Add speech output section
        let output_prefix = " Speech output:\n";
        let speech_start_id = self
            .tokenizer
            .token_to_id("<|vision_start|>")
            .unwrap_or(151652);

        let output_prefix_tokens = self
            .tokenizer
            .encode(output_prefix, false)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to encode output prefix: {}", e)))?
            .get_ids()
            .to_vec();
        let output_prefix_len = output_prefix_tokens.len();

        full_tokens.extend(output_prefix_tokens);
        full_tokens.push(speech_start_id);
        speech_input_mask.extend(vec![false; output_prefix_len + 1]);

        debug!(
            "✓ Created integrated sequence: {} tokens",
            full_tokens.len()
        );

        // Debug token sequence details
        if tracing::enabled!(tracing::Level::DEBUG) {
            let speech_start_count = full_tokens.iter().filter(|&&t| t == 151652).count();
            let speech_diffusion_count = full_tokens.iter().filter(|&&t| t == 151654).count();
            let speech_end_count = full_tokens.iter().filter(|&&t| t == 151653).count();
            debug!(
                "Token counts: start={}, diffusion={}, end={}",
                speech_start_count, speech_diffusion_count, speech_end_count
            );
            debug!(
                "First 10 tokens: {:?}",
                &full_tokens[..10.min(full_tokens.len())]
            );
        }

        // Create tensors
        let batch_size = 1;
        let seq_len = full_tokens.len();

        let input_ids = Tensor::from_vec(full_tokens.clone(), (batch_size, seq_len), &self.device)?;

        let attention_mask = if return_attention_mask {
            Some(Tensor::ones(
                (batch_size, seq_len),
                candle_core::DType::U32,
                &self.device,
            )?)
        } else {
            None
        };

        let speech_input_mask_tensor = Tensor::from_vec(
            speech_input_mask
                .into_iter()
                .map(|b| if b { 1u8 } else { 0u8 })
                .collect::<Vec<u8>>(),
            (batch_size, seq_len),
            &self.device,
        )?;

        let voice_masks = if voice_embeds.is_some() {
            Some(Tensor::ones(
                (seq_len, 1),
                candle_core::DType::U32,
                &self.device,
            )?)
        } else {
            None
        };

        Ok(ProcessorOutput {
            input_ids,
            attention_mask,
            voice_embeds,
            voice_masks,
            speech_input_mask: Some(speech_input_mask_tensor),
            speech_masks,
        })
    }

    /// Create voice prompt tokens and masks for voice injection
    /// Uses 0-indexed speaker IDs (matching Python normalization)
    /// Returns: (voice_tokens, voice_audio, voice_mask, speech_masks)
    /// - voice_tokens: token IDs for the voice prompt section
    /// - voice_audio: stacked audio tensor [num_speakers, 1, max_samples]
    /// - voice_mask: flat boolean mask marking VAE token positions in voice_tokens
    /// - speech_masks: per-speaker validity masks [num_speakers][max_vae_tokens]
    fn create_voice_prompt(
        &self,
        voice_samples: &[PathBuf],
    ) -> Result<(Vec<u32>, Tensor, Vec<bool>, Vec<Vec<bool>>)> {
        let vae_token_id = self
            .tokenizer
            .token_to_id("<|vision_pad|>")
            .unwrap_or(151654);
        let speech_start_id = self
            .tokenizer
            .token_to_id("<|vision_start|>")
            .unwrap_or(151652);
        let speech_end_id = self
            .tokenizer
            .token_to_id("<|vision_end|>")
            .unwrap_or(151653);

        let mut voice_tokens = self
            .tokenizer
            .encode(" Voice input:\n", false)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to encode voice prefix: {}", e)))?
            .get_ids()
            .to_vec();
        let mut voice_mask = vec![false; voice_tokens.len()];

        let mut voice_audio_list = Vec::new();
        // Track each speaker's actual VAE token count (before padding)
        let mut vae_token_counts: Vec<usize> = Vec::new();

        for (speaker_id, voice_path) in voice_samples.iter().enumerate() {
            let speaker_prefix = format!(" Speaker {}:", speaker_id);
            let prefix_tokens = self
                .tokenizer
                .encode(speaker_prefix.as_str(), false)
                .map_err(|e| {
                    candle_core::Error::Msg(format!("Failed to encode speaker prefix: {}", e))
                })?
                .get_ids()
                .to_vec();
            let prefix_len = prefix_tokens.len();

            voice_tokens.extend(prefix_tokens);
            voice_mask.extend(vec![false; prefix_len]);

            // Load audio
            let audio = load_audio_wav(
                voice_path
                    .to_str()
                    .ok_or_else(|| candle_core::Error::Msg("Invalid voice path".to_string()))?,
                self.sample_rate,
            )
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load voice audio: {}", e)))?;

            // Normalize audio
            let normalized = normalize_audio_db_fs(&audio, -25.0).map_err(|e| {
                candle_core::Error::Msg(format!("Failed to normalize voice audio: {}", e))
            })?;

            // Get the actual sample count AFTER normalization
            // The audio tensor shape is [batch, channels, samples] or [channels, samples]
            let audio_dims = normalized.dims();
            let sample_count = match audio_dims.len() {
                1 => audio_dims[0], // [samples]
                2 => audio_dims[1], // [channels, samples]
                3 => audio_dims[2], // [batch, channels, samples]
                _ => {
                    return Err(candle_core::Error::Msg(format!(
                        "Unexpected audio shape: {:?}",
                        audio_dims
                    )));
                }
            };

            debug!(
                "Speaker {}: audio shape {:?}, {} samples",
                speaker_id, audio_dims, sample_count
            );

            // CRITICAL: Calculate VAE tokens from the NORMALIZED audio's actual sample count
            // This should match Python's wav.shape[0]
            let vae_tok_len =
                (sample_count as f32 / self.speech_tok_compress_ratio as f32).ceil() as usize;
            let vae_tok_len = vae_tok_len.max(1);

            debug!(
                "Speaker {}: calculated {} VAE tokens from {} samples (ratio: {})",
                speaker_id, vae_tok_len, sample_count, self.speech_tok_compress_ratio
            );

            // Track this speaker's VAE token count for speech_masks
            vae_token_counts.push(vae_tok_len);

            // Add speech tokens
            voice_tokens.push(speech_start_id);
            voice_mask.push(false);

            voice_tokens.extend(vec![vae_token_id; vae_tok_len]);
            voice_mask.extend(vec![true; vae_tok_len]);

            voice_tokens.push(speech_end_id);
            voice_mask.push(false);

            // Add newline
            let newline_tokens = self
                .tokenizer
                .encode("\n", false)
                .map_err(|e| candle_core::Error::Msg(format!("Failed to encode newline: {}", e)))?
                .get_ids()
                .to_vec();
            let newline_len = newline_tokens.len();
            voice_tokens.extend(newline_tokens);
            voice_mask.extend(vec![false; newline_len]);

            // Move to device AFTER calculating sample count
            let audio_on_device = normalized.to_device(&self.device)?;
            voice_audio_list.push(audio_on_device);
        }

        let voice_audio = if voice_audio_list.is_empty() {
            return Err(candle_core::Error::Msg(
                "No voice samples provided".to_string(),
            ));
        } else if voice_audio_list.len() == 1 {
            voice_audio_list[0].clone()
        } else {
            // Multiple speakers: need to pad to same length and stack properly
            // Each audio is [1, 1, samples] - we need final shape [num_speakers, 1, samples]
            let max_samples = voice_audio_list
                .iter()
                .map(|a| a.dims()[a.dims().len() - 1])
                .max()
                .unwrap_or(0);

            let mut padded_audios = Vec::new();
            for audio in &voice_audio_list {
                let dims = audio.dims();
                let current_samples = dims[dims.len() - 1];

                // Squeeze to [1, samples] (remove extra batch dim if present)
                let squeezed = if dims.len() == 4 {
                    // [1, 1, 1, samples] -> [1, samples]
                    audio.squeeze(0)?.squeeze(0)?
                } else if dims.len() == 3 && dims[0] == 1 {
                    // [1, 1, samples] -> [1, samples]
                    audio.squeeze(0)?
                } else {
                    audio.clone()
                };

                // Pad if needed
                let padded = if current_samples < max_samples {
                    let pad_size = max_samples - current_samples;
                    let padding =
                        Tensor::zeros((1, pad_size), squeezed.dtype(), squeezed.device())?;
                    squeezed.device().synchronize()?;
                    Tensor::cat(&[&squeezed, &padding], 1)?
                } else {
                    squeezed
                };

                padded_audios.push(padded);
            }

            // Stack along batch dim: [1, samples] * N -> [N, 1, samples]
            // First, cat along dim 0 to get [N, samples], then unsqueeze
            let stacked = Tensor::cat(&padded_audios.iter().collect::<Vec<_>>(), 0)?;
            stacked.unsqueeze(1)? // [N, samples] -> [N, 1, samples]
        };

        debug!(
            "✓ Created voice prompt: {} tokens ({} VAE tokens)",
            voice_tokens.len(),
            voice_tokens.iter().filter(|&&t| t == vae_token_id).count()
        );

        // Create per-speaker speech_masks for filtering padded VAE tokens
        // This matches Python's prepare_speech_inputs which creates speech_masks: [num_speakers, max_vae_tokens]
        let max_vae_tokens = vae_token_counts.iter().copied().max().unwrap_or(0);
        let speech_masks: Vec<Vec<bool>> = vae_token_counts
            .iter()
            .map(|&count| {
                let mut mask = vec![true; count];
                mask.extend(vec![false; max_vae_tokens - count]);
                mask
            })
            .collect();

        debug!(
            "✓ Created speech_masks: {} speakers, max {} VAE tokens, counts: {:?}",
            speech_masks.len(),
            max_vae_tokens,
            vae_token_counts
        );

        Ok((voice_tokens, voice_audio, voice_mask, speech_masks))
    }

    /// Load and process voice samples into audio tensors
    /// Returns (voice_tensors, voice_masks) tuple
    /// voice_tensors: raw audio that will be encoded by the model's VAE
    /// voice_masks: attention masks for voice samples
    pub fn process_voice_samples(&self, samples: &[Vec<PathBuf>]) -> Result<(Tensor, Tensor)> {
        use crate::utils::{load_audio_wav, normalize_audio_db_fs};

        debug!("Loading {} voice sample groups", samples.len());

        let mut all_audio = Vec::new();

        for (i, sample_group) in samples.iter().enumerate() {
            for sample_path in sample_group {
                debug!(
                    "     [{}/{}] Loading {:?}",
                    i + 1,
                    samples.len(),
                    sample_path.file_name().unwrap()
                );

                // Load audio at 24kHz
                let audio = load_audio_wav(
                    sample_path
                        .to_str()
                        .ok_or_else(|| candle_core::Error::Msg("Invalid path".to_string()))?,
                    self.sample_rate,
                )
                .map_err(|e| candle_core::Error::Msg(format!("Failed to load audio: {}", e)))?;

                // Normalize to -25 dB FS (matches Python processor)
                let normalized = normalize_audio_db_fs(&audio, -25.0).map_err(|e| {
                    candle_core::Error::Msg(format!("Failed to normalize audio: {}", e))
                })?;

                // Move to target device
                let audio_on_device = normalized.to_device(&self.device)?;

                all_audio.push(audio_on_device);
            }
        }

        // Group audio samples by batch item
        // samples[i] contains voice files for batch item i
        let mut batch_voice_tensors = Vec::new();
        let mut batch_voice_masks = Vec::new();

        let mut audio_idx = 0;
        for (batch_idx, sample_group) in samples.iter().enumerate() {
            if sample_group.is_empty() {
                return Err(candle_core::Error::Msg(format!(
                    "Batch item {} has no voice samples",
                    batch_idx
                )));
            }

            // Support multiple voice samples by concatenating along time axis
            let end_idx = audio_idx + sample_group.len();
            let voice_tensor = if sample_group.len() == 1 {
                all_audio[audio_idx].clone()
            } else {
                // Concatenate multiple voice samples for this batch item
                let samples: Vec<&Tensor> = all_audio[audio_idx..end_idx].iter().collect();
                Tensor::cat(&samples, 0)? // Concatenate along time dimension
            };
            audio_idx = end_idx;

            // Create voice mask (all 1s since all audio is valid)
            let audio_len = voice_tensor.dims()[0];
            let voice_mask = Tensor::ones((1, audio_len), candle_core::DType::U32, &self.device)?;

            batch_voice_tensors.push(voice_tensor);
            batch_voice_masks.push(voice_mask);
        }

        debug!(
            "✓ Processed {} batch items with voice samples",
            batch_voice_tensors.len()
        );

        // Stack batch items
        let stacked_voices = Tensor::stack(&batch_voice_tensors, 0)?;
        let stacked_masks = Tensor::stack(&batch_voice_masks, 0)?;

        Ok((stacked_voices, stacked_masks))
    }

    /// Save audio tensor to WAV file at 24kHz
    /// Matches Python: processor.save_audio(audio, output_path)
    pub fn save_audio(&self, audio: &Tensor, output_path: &Path) -> Result<()> {
        debug!("💾 Saving audio to {:?}", output_path);

        // Convert tensor to f32 vec
        let audio_data = audio.flatten_all()?.to_vec1::<f32>()?;

        // Create WAV spec
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        // Write WAV file
        let mut writer = hound::WavWriter::create(output_path, spec)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create WAV writer: {}", e)))?;

        // Convert f32 samples to i16
        for sample in audio_data {
            let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
            writer
                .write_sample(sample_i16)
                .map_err(|e| candle_core::Error::Msg(format!("Failed to write sample: {}", e)))?;
        }

        writer
            .finalize()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to finalize WAV: {}", e)))?;

        debug!("✓ Audio saved successfully");

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ParsedScript {
    pub lines: Vec<(usize, String)>, // (speaker_id, text)
    pub unique_speakers: Vec<usize>,
}
