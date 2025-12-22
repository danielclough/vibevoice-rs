//! High-level VibeVoice API for text-to-speech synthesis.

use crate::{
    AudioData, Result, VibeVoiceError, model::VibeVoiceModel, processor::VibeVoiceProcessor, pytorch_rng::set_all_seeds, realtime::{model::VibeVoiceRealtimeModel, voice_cache::SafetensorCache}, utils::{
        create_remapped_varbuilder, download_model_files, download_realtime_model_files,
        get_device, resolve_voice_path, detect_voice_type, VoiceType,
    }, voice_mapper::{VoiceMapper, parse_txt_script}
};
use candle_core::{Device as CandleDevice, Tensor};
use std::path::{Path, PathBuf};
use tracing::info;

/// Model variant selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ModelVariant {
    /// 1.5B parameter batch model (default, good quality/speed balance)
    #[default]
    Batch1_5B,
    /// 7B parameter batch model (highest quality, slower)
    Batch7B,
    /// 0.5B parameter realtime streaming model (fastest, streaming support)
    Realtime,
}

/// Device selection for model execution.
#[derive(Debug, Clone, Default)]
pub enum Device {
    /// Automatic device selection (GPU if available, else CPU)
    #[default]
    Auto,
    /// CPU execution
    Cpu,
    /// CUDA GPU with device ordinal
    Cuda(usize),
    /// Metal GPU (Apple Silicon)
    Metal,
}

impl Device {
    /// Automatically select the best available device.
    pub fn auto() -> Self {
        Self::Auto
    }
}

/// Progress information during generation.
#[derive(Debug, Clone)]
pub struct Progress {
    /// Current step (audio chunk index)
    pub step: usize,
    /// Estimated total steps (may be None if unknown)
    pub total_steps: Option<usize>,
    /// Current audio chunk (if available)
    pub audio_chunk: Option<AudioData>,
}

/// Progress callback for streaming generation.
pub type ProgressCallback = Box<dyn FnMut(Progress) + Send>;

/// High-level VibeVoice API for text-to-speech synthesis.
///
/// This is the primary interface for frontend applications.
///
/// # Example
///
/// ```no_run
/// use vibevoice::{VibeVoice, ModelVariant, Device};
///
/// let mut vv = VibeVoice::new(ModelVariant::Batch1_5B, Device::auto())?;
/// let audio = vv.synthesize("Hello, world!", None)?;
/// audio.save_wav("output.wav")?;
/// # Ok::<(), vibevoice::VibeVoiceError>(())
/// ```
pub struct VibeVoice {
    variant: ModelVariant,
    device: CandleDevice,
    inner: ModelInner,
    processor: Option<VibeVoiceProcessor>,
    seed: u64,
    cfg_scale: f32,
}

enum ModelInner {
    Batch(VibeVoiceModel),
    Realtime(VibeVoiceRealtimeModel),
}

impl VibeVoice {
    /// Create a new VibeVoice instance with the specified model variant and device.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use vibevoice::{VibeVoice, ModelVariant, Device};
    ///
    /// let vv = VibeVoice::new(ModelVariant::Batch1_5B, Device::auto())?;
    /// # Ok::<(), vibevoice::VibeVoiceError>(())
    /// ```
    pub fn new(variant: ModelVariant, device: Device) -> Result<Self> {
        VibeVoiceBuilder::new()
            .variant(variant)
            .device(device)
            .build()
    }

    /// Create a builder for more configuration options.
    pub fn builder() -> VibeVoiceBuilder {
        VibeVoiceBuilder::new()
    }

    /// Synthesize speech from text.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to synthesize
    /// * `voice_path` - Optional path to voice sample for cloning (batch models)
    ///                  or voice cache file (realtime model)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use vibevoice::{VibeVoice, ModelVariant, Device};
    /// # let mut vv = VibeVoice::new(ModelVariant::Batch1_5B, Device::auto())?;
    /// let audio = vv.synthesize("Hello, world!", None)?;
    /// audio.save_wav("output.wav")?;
    /// # Ok::<(), vibevoice::VibeVoiceError>(())
    /// ```
    pub fn synthesize(&mut self, text: &str, voice_path: Option<&str>) -> Result<AudioData> {
        self.synthesize_with_callback(text, voice_path, None)
    }

    /// Synthesize speech with streaming callback.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to synthesize
    /// * `voice_path` - Optional path to voice sample or voice cache
    /// * `callback` - Optional callback for streaming progress
    pub fn synthesize_with_callback(
        &mut self,
        text: &str,
        voice_path: Option<&str>,
        callback: Option<ProgressCallback>,
    ) -> Result<AudioData> {
        // Reset seed for reproducibility
        set_all_seeds(self.seed, &self.device)
            .map_err(|e| VibeVoiceError::InitializationError(e.to_string()))?;

        // Validate voice file type matches model
        if let Some(voice) = voice_path {
            if let Some(detected_type) = detect_voice_type(voice) {
                match (self.variant, detected_type) {
                    (ModelVariant::Realtime, VoiceType::WavSample) => {
                        return Err(VibeVoiceError::VoiceError(
                            "Realtime model requires .safetensors voice cache, not .wav".to_string(),
                        ));
                    }
                    (ModelVariant::Batch1_5B | ModelVariant::Batch7B, VoiceType::SafetensorCache) => {
                        return Err(VibeVoiceError::VoiceError(
                            "Batch models require .wav voice samples, not .safetensors".to_string(),
                        ));
                    }
                    _ => {}
                }
            }
        }

        match self.variant {
            ModelVariant::Batch1_5B | ModelVariant::Batch7B => {
                self.synthesize_batch_impl(text, voice_path, callback)
            }
            ModelVariant::Realtime => self.synthesize_realtime_impl(text, voice_path, callback),
        }
    }

    /// Synthesize multi-speaker dialogue from a script file.
    ///
    /// # Arguments
    ///
    /// * `script_path` - Path to script file with "Speaker N: text" format
    /// * `voices_dir` - Optional directory containing voice samples
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use vibevoice::{VibeVoice, ModelVariant, Device};
    /// # let mut vv = VibeVoice::new(ModelVariant::Batch1_5B, Device::auto())?;
    /// let audio = vv.synthesize_script("dialogue.txt", Some("voices/"))?;
    /// audio.save_wav("dialogue.wav")?;
    /// # Ok::<(), vibevoice::VibeVoiceError>(())
    /// ```
    pub fn synthesize_script(
        &mut self,
        script_path: &str,
        voices_dir: Option<&str>,
    ) -> Result<AudioData> {
        self.synthesize_script_with_callback(script_path, voices_dir, None)
    }

    /// Synthesize multi-speaker dialogue with streaming callback.
    pub fn synthesize_script_with_callback(
        &mut self,
        script_path: &str,
        voices_dir: Option<&str>,
        callback: Option<ProgressCallback>,
    ) -> Result<AudioData> {
        if self.variant == ModelVariant::Realtime {
            return Err(VibeVoiceError::UnsupportedOperation(
                "Multi-speaker synthesis not yet supported for realtime model".to_string(),
            ));
        }

        // Reset seed for reproducibility
        set_all_seeds(self.seed, &self.device)
            .map_err(|e| VibeVoiceError::InitializationError(e.to_string()))?;

        let script_content = std::fs::read_to_string(script_path)
            .map_err(|e| VibeVoiceError::IoError(e.to_string()))?;

        let (scripts, speaker_numbers) = parse_txt_script(&script_content)
            .map_err(|e| VibeVoiceError::ProcessingError(e.to_string()))?;

        // Determine voices directory
        let voices_path = voices_dir
            .map(PathBuf::from)
            .or_else(|| {
                Path::new(script_path)
                    .parent()
                    .map(|p| p.join("voices"))
                    .filter(|p| p.exists())
            })
            .unwrap_or_else(|| PathBuf::from("./voices"));

        let voice_mapper = VoiceMapper::new(&voices_path)
            .map_err(|e| VibeVoiceError::VoiceError(e.to_string()))?;

        let voice_samples = voice_mapper
            .map_speakers_to_voices(&speaker_numbers)
            .map_err(|e| VibeVoiceError::VoiceError(e.to_string()))?;

        // Join scripts and synthesize
        let full_script = scripts.join("\n");
        let all_voices: Vec<PathBuf> = voice_samples.into_iter().flatten().collect();

        self.synthesize_batch_multi_speaker_impl(&full_script, all_voices, callback)
    }

    /// Get the current model variant.
    pub fn variant(&self) -> ModelVariant {
        self.variant
    }

    /// Get the current device.
    pub fn device(&self) -> &CandleDevice {
        &self.device
    }

    /// Set the random seed for reproducibility.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    /// Set the CFG (Classifier-Free Guidance) scale.
    pub fn set_cfg_scale(&mut self, scale: f32) {
        self.cfg_scale = scale;
        if let ModelInner::Batch(model) = &mut self.inner {
            model.set_cfg_scale(scale);
        }
    }

    /// Set the number of diffusion steps (realtime model only).
    pub fn set_diffusion_steps(&mut self, steps: usize) -> Result<()> {
        match &mut self.inner {
            ModelInner::Realtime(model) => {
                model.set_diffusion_steps(steps);
                Ok(())
            }
            ModelInner::Batch(_) => Err(VibeVoiceError::UnsupportedOperation(
                "Diffusion steps can only be changed for realtime model".to_string(),
            )),
        }
    }

    // === Private implementation methods ===

    fn synthesize_batch_impl(
        &mut self,
        text: &str,
        voice_path: Option<&str>,
        callback: Option<ProgressCallback>,
    ) -> Result<AudioData> {
        let processor = self.processor.as_ref().ok_or_else(|| {
            VibeVoiceError::InitializationError("Processor not initialized".to_string())
        })?;

        // Format text with speaker prefix if needed
        let text_with_speaker = if text.trim().starts_with("Speaker ") {
            text.to_string()
        } else {
            format!("Speaker 1: {}", text.trim())
        };

        let (scripts, speaker_numbers) = parse_txt_script(&text_with_speaker)
            .map_err(|e| VibeVoiceError::ProcessingError(e.to_string()))?;

        // Get unique speakers
        let mut seen = std::collections::HashSet::new();
        let unique_speakers: Vec<_> = speaker_numbers
            .iter()
            .filter(|s| seen.insert(s.to_string()))
            .collect();

        // Load voice samples if provided
        let voice_samples = if let Some(voice_input) = voice_path {
            let resolved_path = resolve_voice_path(voice_input, None, VoiceType::WavSample)
                .map_err(|e| VibeVoiceError::VoiceError(e.to_string()))?;
            let all_voices: Vec<PathBuf> = unique_speakers
                .iter()
                .map(|_| resolved_path.clone())
                .collect();
            Some(vec![all_voices])
        } else {
            None
        };

        let full_script = scripts.join("\n");

        // Process inputs
        let processed_inputs = processor
            .process(vec![full_script], voice_samples, false, true)
            .map_err(|e| VibeVoiceError::ProcessingError(e.to_string()))?;

        // Get the model
        let model = match &mut self.inner {
            ModelInner::Batch(m) => m,
            _ => {
                return Err(VibeVoiceError::InitializationError(
                    "Expected batch model".to_string(),
                ));
            }
        };

        // Generate audio with optional callback
        let audio_chunks = if let Some(mut cb) = callback {
            let mut step = 0;
            model
                .generate_processed(
                    &processed_inputs.input_ids,
                    processed_inputs.voice_embeds.as_ref(),
                    processed_inputs.speech_input_mask.as_ref(),
                    processed_inputs.attention_mask.as_ref(),
                    processed_inputs.speech_masks.as_ref(),
                    None::<usize>,
                    true,
                    Some(move |chunk: &Tensor| {
                        step += 1;
                        let audio_chunk = AudioData::from_tensor(chunk, 24000).ok();
                        cb(Progress {
                            step,
                            total_steps: None,
                            audio_chunk,
                        });
                        Ok(())
                    }),
                )
                .map_err(|e| VibeVoiceError::GenerationError(e.to_string()))?
        } else {
            model
                .generate_processed(
                    &processed_inputs.input_ids,
                    processed_inputs.voice_embeds.as_ref(),
                    processed_inputs.speech_input_mask.as_ref(),
                    processed_inputs.attention_mask.as_ref(),
                    processed_inputs.speech_masks.as_ref(),
                    None::<usize>,
                    true,
                    None::<fn(&Tensor) -> anyhow::Result<()>>,
                )
                .map_err(|e| VibeVoiceError::GenerationError(e.to_string()))?
        };

        if audio_chunks.is_empty() {
            return Err(VibeVoiceError::GenerationError(
                "No audio chunks generated".to_string(),
            ));
        }

        let chunk_refs: Vec<&Tensor> = audio_chunks.iter().collect();
        let concatenated = Tensor::cat(&chunk_refs, 2)?;

        AudioData::from_tensor(&concatenated, 24000)
    }

    fn synthesize_batch_multi_speaker_impl(
        &mut self,
        script: &str,
        all_voices: Vec<PathBuf>,
        callback: Option<ProgressCallback>,
    ) -> Result<AudioData> {
        let processor = self.processor.as_ref().ok_or_else(|| {
            VibeVoiceError::InitializationError("Processor not initialized".to_string())
        })?;

        // Process inputs
        let processed_inputs = processor
            .process(
                vec![script.to_string()],
                Some(vec![all_voices]),
                false,
                true,
            )
            .map_err(|e| VibeVoiceError::ProcessingError(e.to_string()))?;

        // Get the model
        let model = match &mut self.inner {
            ModelInner::Batch(m) => m,
            _ => {
                return Err(VibeVoiceError::InitializationError(
                    "Expected batch model".to_string(),
                ));
            }
        };

        // Generate audio with optional callback
        let audio_chunks = if let Some(mut cb) = callback {
            let mut step = 0;
            model
                .generate_processed(
                    &processed_inputs.input_ids,
                    processed_inputs.voice_embeds.as_ref(),
                    processed_inputs.speech_input_mask.as_ref(),
                    processed_inputs.attention_mask.as_ref(),
                    processed_inputs.speech_masks.as_ref(),
                    None::<usize>,
                    true,
                    Some(move |chunk: &Tensor| {
                        step += 1;
                        let audio_chunk = AudioData::from_tensor(chunk, 24000).ok();
                        cb(Progress {
                            step,
                            total_steps: None,
                            audio_chunk,
                        });
                        Ok(())
                    }),
                )
                .map_err(|e| VibeVoiceError::GenerationError(e.to_string()))?
        } else {
            model
                .generate_processed(
                    &processed_inputs.input_ids,
                    processed_inputs.voice_embeds.as_ref(),
                    processed_inputs.speech_input_mask.as_ref(),
                    processed_inputs.attention_mask.as_ref(),
                    processed_inputs.speech_masks.as_ref(),
                    None::<usize>,
                    true,
                    None::<fn(&Tensor) -> anyhow::Result<()>>,
                )
                .map_err(|e| VibeVoiceError::GenerationError(e.to_string()))?
        };

        if audio_chunks.is_empty() {
            return Err(VibeVoiceError::GenerationError(
                "No audio chunks generated".to_string(),
            ));
        }

        let chunk_refs: Vec<&Tensor> = audio_chunks.iter().collect();
        let concatenated = Tensor::cat(&chunk_refs, 2)?;

        AudioData::from_tensor(&concatenated, 24000)
    }

    fn synthesize_realtime_impl(
        &mut self,
        text: &str,
        voice_cache_path: Option<&str>,
        callback: Option<ProgressCallback>,
    ) -> Result<AudioData> {
        let cache_path = voice_cache_path.ok_or_else(|| {
            VibeVoiceError::VoiceError("Voice cache path required for realtime model".to_string())
        })?;

        // Resolve voice cache path
        let resolved_path = resolve_voice_path(cache_path, None, VoiceType::SafetensorCache)
            .map_err(|e| VibeVoiceError::VoiceError(e.to_string()))?;

        let voice_cache = SafetensorCache::from_safetensors(
            resolved_path.to_str().ok_or_else(|| {
                VibeVoiceError::VoiceError("Invalid voice cache path".to_string())
            })?,
            &self.device,
        )
        .map_err(|e| VibeVoiceError::VoiceError(e.to_string()))?;

        // Get the model
        let model = match &mut self.inner {
            ModelInner::Realtime(m) => m,
            _ => {
                return Err(VibeVoiceError::InitializationError(
                    "Expected realtime model".to_string(),
                ));
            }
        };

        // Generate audio with optional callback
        let audio = if let Some(mut cb) = callback {
            let mut step = 0;
            model
                .generate(text, &voice_cache, move |chunk| {
                    step += 1;
                    // Convert chunk to AudioData - log errors but don't fail generation
                    let audio_chunk = match AudioData::from_tensor(chunk, 24000) {
                        Ok(data) => Some(data),
                        Err(e) => {
                            tracing::error!("Chunk {} conversion failed: {}", step, e);
                            None
                        }
                    };
                    cb(Progress {
                        step,
                        total_steps: None,
                        audio_chunk,
                    });
                    Ok(())
                })
                .map_err(|e| VibeVoiceError::GenerationError(e.to_string()))?
        } else {
            model
                .generate(text, &voice_cache, |_| Ok(()))
                .map_err(|e| VibeVoiceError::GenerationError(e.to_string()))?
        };

        AudioData::from_tensor(&audio, 24000)
    }
}

/// Builder for VibeVoice with additional configuration options.
#[derive(Default)]
pub struct VibeVoiceBuilder {
    variant: ModelVariant,
    device: Device,
    seed: u64,
    cfg_scale: f32,
    model_path: Option<String>,
    diffusion_steps: Option<usize>,
    restore_rng_after_voice_embedding: bool,
}

impl VibeVoiceBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            variant: ModelVariant::default(),
            device: Device::Auto,
            seed: 524242,
            cfg_scale: 1.3,
            model_path: None,
            diffusion_steps: None,
            restore_rng_after_voice_embedding: false,
        }
    }

    /// Set the model variant.
    pub fn variant(mut self, variant: ModelVariant) -> Self {
        self.variant = variant;
        self
    }

    /// Set the device.
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Set the random seed for reproducibility.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the CFG (Classifier-Free Guidance) scale.
    pub fn cfg_scale(mut self, scale: f32) -> Self {
        self.cfg_scale = scale;
        self
    }

    /// Use a local model path instead of downloading from HuggingFace.
    pub fn model_path(mut self, path: impl Into<String>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Set the number of diffusion steps (realtime model only).
    pub fn diffusion_steps(mut self, steps: usize) -> Self {
        self.diffusion_steps = Some(steps);
        self
    }

    /// Set whether to restore RNG state after voice embedding.
    ///
    /// When `false` (default): Diffusion continues from where voice embedding left off.
    /// This matches Python's quality pattern where some voices work well.
    ///
    /// When `true`: RNG is restored after voice embedding, so diffusion starts from position 0.
    /// This may work better for some voices but worse for others.
    pub fn restore_rng_after_voice_embedding(mut self, restore: bool) -> Self {
        self.restore_rng_after_voice_embedding = restore;
        self
    }

    /// Build the VibeVoice instance.
    pub fn build(self) -> Result<VibeVoice> {
        // Convert Device enum to CandleDevice
        let device = match self.device {
            Device::Auto => {
                get_device(Some(0)).map_err(|e| VibeVoiceError::DeviceError(e.to_string()))?
            }
            Device::Cpu => CandleDevice::Cpu,
            Device::Cuda(ordinal) => CandleDevice::new_cuda(ordinal)
                .map_err(|e| VibeVoiceError::DeviceError(e.to_string()))?,
            Device::Metal => CandleDevice::new_metal(0)
                .map_err(|e| VibeVoiceError::DeviceError(e.to_string()))?,
        };

        set_all_seeds(self.seed, &device)
            .map_err(|e| VibeVoiceError::InitializationError(e.to_string()))?;

        let (inner, processor) = match self.variant {
            ModelVariant::Batch1_5B | ModelVariant::Batch7B => {
                let model_id = match self.variant {
                    ModelVariant::Batch1_5B => "vibevoice/VibeVoice-1.5B",
                    ModelVariant::Batch7B => "vibevoice/VibeVoice-7B",
                    _ => unreachable!(),
                };

                info!("Downloading model from HuggingFace: {}", model_id);
                let (model_dir, config_path, tokenizer_path) = download_model_files(model_id)
                    .map_err(|e| VibeVoiceError::DownloadError(e.to_string()))?;

                info!("Loading model weights...");
                let vb = create_remapped_varbuilder(&model_dir, &device)
                    .map_err(|e| VibeVoiceError::InitializationError(e.to_string()))?;

                let mut model =
                    VibeVoiceModel::new(vb, device.clone(), &config_path, &tokenizer_path)
                        .map_err(|e| VibeVoiceError::InitializationError(e.to_string()))?;

                model.set_cfg_scale(self.cfg_scale);
                model.set_seed(self.seed);
                model.set_restore_rng_after_voice_embedding(self.restore_rng_after_voice_embedding);
                if let Some(steps) = self.diffusion_steps {
                    model.set_ddpm_inference_steps(steps);
                }

                let processor = VibeVoiceProcessor::from_pretrained(&model_dir, &device)
                    .map_err(|e| VibeVoiceError::InitializationError(e.to_string()))?;

                info!("Model loaded successfully");
                (ModelInner::Batch(model), Some(processor))
            }
            ModelVariant::Realtime => {
                let model_path = if let Some(path) = self.model_path {
                    PathBuf::from(path)
                } else {
                    info!("Downloading realtime model from HuggingFace...");
                    download_realtime_model_files("VibeVoice/VibeVoice-Realtime-0.5B")
                        .map_err(|e| VibeVoiceError::DownloadError(e.to_string()))?
                };

                info!("Loading realtime model...");
                let mut model = VibeVoiceRealtimeModel::from_pretrained(&model_path, &device)
                    .map_err(|e| VibeVoiceError::InitializationError(e.to_string()))?;

                if let Some(steps) = self.diffusion_steps {
                    model.set_diffusion_steps(steps);
                }

                info!("Realtime model loaded successfully");
                (ModelInner::Realtime(model), None)
            }
        };

        Ok(VibeVoice {
            variant: self.variant,
            device,
            inner,
            processor,
            seed: self.seed,
            cfg_scale: self.cfg_scale,
        })
    }
}
