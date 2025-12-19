use anyhow::{Error as AnyErr, Result};
use candle_core::{D, DType, Device, Tensor};
use candle_hf_hub::api::sync::Api;
use candle_nn::VarBuilder;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use serde_json;
use std::fs::File;
use std::path::PathBuf;
use std::{fs, path::Path};
use tracing::{debug, error, info, warn};

/// Resolve a voice path from user input.
///
/// Accepts:
/// - Absolute path: used directly if it exists
/// - Relative path: resolved from current directory if it exists
/// - Voice name (with or without .wav): searched in:
///   1. Script directory's `voices/` folder (if script_dir provided)
///   2. `./voices/` folder (current working directory)
///   3. Executable directory's `voices/` folder
///
/// Returns the resolved PathBuf or an error if the voice cannot be found.
pub fn resolve_voice_path(voice_input: &str, script_dir: Option<&Path>) -> Result<PathBuf> {
    let voice_path = Path::new(voice_input);

    // 1. If it's an absolute path, use it directly
    if voice_path.is_absolute() {
        if voice_path.exists() {
            debug!("Voice resolved as absolute path: {:?}", voice_path);
            return Ok(voice_path.to_path_buf());
        } else {
            return Err(AnyErr::msg(format!(
                "Voice file not found at absolute path: {}",
                voice_input
            )));
        }
    }

    // 2. If it exists as a relative path from current directory, use it
    if voice_path.exists() {
        debug!("Voice resolved as relative path: {:?}", voice_path);
        return Ok(voice_path.to_path_buf());
    }

    // 2b. Try relative to script directory and its parent (project root) if provided
    if let Some(script_dir) = script_dir {
        let relative_to_script = script_dir.join(voice_path);
        if relative_to_script.exists() {
            debug!(
                "Voice resolved as path relative to script dir: {:?}",
                relative_to_script
            );
            return Ok(relative_to_script);
        }
        // Also try relative to script's parent directory (common for project root)
        if let Some(script_parent) = script_dir.parent() {
            let relative_to_parent = script_parent.join(voice_path);
            if relative_to_parent.exists() {
                debug!(
                    "Voice resolved as path relative to script parent dir: {:?}",
                    relative_to_parent
                );
                return Ok(relative_to_parent);
            }
        }
    }

    // 2c. Try relative to executable directory
    if let Ok(exe_path) = std::env::current_exe()
        && let Some(exe_dir) = exe_path.parent()
        && let relative_to_exe = exe_dir.join(voice_path)
        && relative_to_exe.exists()
    {
        debug!(
            "Voice resolved as path relative to exe dir: {:?}",
            relative_to_exe
        );
        return Ok(relative_to_exe);
    }

    // 3. Treat as a voice name - search in voices directories
    // Ensure we have a .wav extension for searching
    let voice_name = if voice_input.ends_with(".wav") {
        voice_input.to_string()
    } else {
        format!("{}.wav", voice_input)
    };

    // Build list of directories to search
    let mut search_dirs: Vec<PathBuf> = Vec::new();

    // 3a. Script directory's voices folder (highest priority)
    if let Some(script_dir) = script_dir {
        search_dirs.push(script_dir.join("voices"));
        // Also check parent of script dir (e.g., project root)
        if let Some(script_parent) = script_dir.parent() {
            search_dirs.push(script_parent.join("voices"));
        }
    }

    // 3b. Current working directory's voices folder
    search_dirs.push(PathBuf::from("./voices"));

    // 3c. Executable directory's voices folder
    if let Ok(exe_path) = std::env::current_exe()
        && let Some(exe_dir) = exe_path.parent()
    {
        search_dirs.push(exe_dir.join("voices"));
    }

    // Search each directory
    for dir in &search_dirs {
        let candidate = dir.join(&voice_name);
        if candidate.exists() {
            debug!("Voice '{}' resolved to: {:?}", voice_input, candidate);
            return Ok(candidate);
        }
    }

    // Not found - provide helpful error message
    let searched_locations: Vec<String> = search_dirs
        .iter()
        .map(|d| d.display().to_string())
        .collect();

    Err(AnyErr::msg(format!(
        "Voice '{}' not found. Searched:\n  - As path: {}\n  - In directories: {}",
        voice_input,
        voice_input,
        searched_locations.join(", ")
    )))
}
/// Device selection with CUDA and Metal support
pub fn get_device(cuda_device: Option<usize>) -> Result<Device> {
    if let Some(ordinal) = cuda_device {
        match Device::new_cuda(ordinal) {
            Ok(device) => {
                info!("Using CUDA device {}", ordinal);
                return Ok(device);
            }
            Err(e) => info!("CUDA not available: {}, trying Metal...", e),
        }
    }
    match Device::new_metal(0) {
        Ok(device) => {
            info!("Using Metal device");
            Ok(device)
        }
        Err(_) => {
            info!("Metal not available, falling back to CPU");
            Ok(Device::Cpu)
        }
    }
}

pub fn download_model_files(repo_id: &str) -> Result<(PathBuf, PathBuf, PathBuf)> {
    info!("Downloading model files from {}...", repo_id);

    let api = Api::new().map_err(|e| AnyErr::msg(format!("Failed to create HF API: {}", e)))?;
    let repo = api.model(repo_id.to_string());

    // Download config.json - use its parent as the model directory (HF cache snapshot dir)
    let config_path = repo
        .get("config.json")
        .map_err(|e| AnyErr::msg(format!("Failed to download config.json: {}", e)))?;
    let model_dir = config_path.parent().unwrap().to_path_buf();
    debug!("Model directory (HF cache): {:?}", model_dir);

    // Try to read preprocessor_config.json to determine tokenizer source
    let tokenizer_repo_id = match repo.get("preprocessor_config.json") {
        Ok(preprocessor_path) => {
            let preprocessor_content = fs::read_to_string(&preprocessor_path)?;
            let preprocessor_config: serde_json::Value =
                serde_json::from_str(&preprocessor_content)?;
            let language_model_name = preprocessor_config
                .get("language_model_pretrained_name")
                .and_then(|v| v.as_str())
                .unwrap_or("Qwen/Qwen2.5-1.5B");
            debug!("Using tokenizer from: {}", language_model_name);
            language_model_name.to_string()
        }
        Err(_) => {
            debug!("preprocessor_config.json not found, using default tokenizer");
            "Qwen/Qwen2.5-1.5B".to_string()
        }
    };

    let index_path = repo
        .get("model.safetensors.index.json")
        .map_err(|e| AnyErr::msg(format!("Failed to download index: {}", e)))?;
    // Parse index to get actual shard filenames
    let index_content = std::fs::read_to_string(&index_path)?;
    let index_json: serde_json::Value = serde_json::from_str(&index_content)?;
    let weight_map = index_json["weight_map"]
        .as_object()
        .ok_or_else(|| AnyErr::msg("Invalid index.json: missing weight_map"))?;
    // Get unique shard filenames
    let shard_filenames_set: std::collections::HashSet<String> = weight_map
        .values()
        .filter_map(|v| v.as_str())
        .map(|s| s.to_string())
        .collect();
    let mut shard_filenames: Vec<String> = shard_filenames_set.into_iter().collect();
    shard_filenames.sort();

    // Download tokenizer from language model (e.g., Qwen)
    let tokenizer_repo = api.model(tokenizer_repo_id.clone());
    let tokenizer_path = tokenizer_repo.get("tokenizer.json").map_err(|e| {
        AnyErr::msg(format!(
            "Failed to download tokenizer.json from {}: {}",
            tokenizer_repo_id, e
        ))
    })?;

    // Download optional tokenizer files from base model
    for file in &[
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ] {
        let _ = tokenizer_repo.get(file);
    }

    // Download model shards to HF cache
    info!("📦 Downloading {} model shards...", shard_filenames.len());
    for (i, filename) in shard_filenames.iter().enumerate() {
        debug!("  Shard {}/{}: {}", i + 1, shard_filenames.len(), filename);
        repo.get(filename)
            .map_err(|e| AnyErr::msg(format!("Failed to download {}: {}", filename, e)))?;
    }

    info!("✓ Model cached in {:?}", model_dir);

    Ok((model_dir, config_path, tokenizer_path))
}

/// Download realtime model files from HuggingFace.
///
/// Downloads the VibeVoice-Realtime-0.5B model files:
/// - config.json - Model configuration
/// - model.safetensors - Model weights (single file)
/// - tokenizer.json - Downloaded from Qwen repo and copied to model dir
///
/// Returns the model directory path (HuggingFace cache directory).
pub fn download_realtime_model_files(repo_id: &str) -> Result<PathBuf> {
    info!("Downloading realtime model files from {}...", repo_id);

    let api = Api::new().map_err(|e| AnyErr::msg(format!("Failed to create HF API: {}", e)))?;
    let repo = api.model(repo_id.to_string());

    // Download config.json - use its parent as the model directory
    info!("📥 Downloading config.json...");
    let config_path = repo
        .get("config.json")
        .map_err(|e| AnyErr::msg(format!("Failed to download config.json: {}", e)))?;
    let model_dir = config_path.parent().unwrap().to_path_buf();
    debug!("Model directory (HF cache): {:?}", model_dir);

    // Download model weights (single file for realtime model)
    info!("📥 Downloading model.safetensors...");
    repo.get("model.safetensors")
        .map_err(|e| AnyErr::msg(format!("Failed to download model.safetensors: {}", e)))?;

    // Try to get tokenizer from model repo first, fall back to Qwen
    let tokenizer_dest = model_dir.join("tokenizer.json");
    if !tokenizer_dest.exists() {
        // Try model repo first
        match repo.get("tokenizer.json") {
            Ok(_) => {
                info!("📥 Using tokenizer from model repo");
            }
            Err(_) => {
                // Fall back to Qwen tokenizer
                info!("📥 Downloading tokenizer from Qwen/Qwen2.5-0.5B...");
                let tokenizer_repo = api.model("Qwen/Qwen2.5-0.5B".to_string());
                let tokenizer_src = tokenizer_repo.get("tokenizer.json").map_err(|e| {
                    AnyErr::msg(format!(
                        "Failed to download tokenizer.json from Qwen: {}",
                        e
                    ))
                })?;

                // Copy tokenizer to model directory for easy access
                fs::copy(&tokenizer_src, &tokenizer_dest).map_err(|e| {
                    AnyErr::msg(format!("Failed to copy tokenizer to model dir: {}", e))
                })?;
                info!("✓ Tokenizer copied to model directory");
            }
        }
    }

    info!("✓ Realtime model cached in {:?}", model_dir);

    Ok(model_dir)
}

/// Sinusoidal timestep embedding for diffusion
pub fn timestep_embedding(timesteps: &Tensor, dim: usize) -> Result<Tensor> {
    let device = timesteps.device();
    let half_dim = dim / 2;
    let emb_scale = -(10000.0_f32.ln()) / half_dim as f32;
    let positions = Tensor::arange(0f32, half_dim as f32, device)?;
    let emb = positions.affine(emb_scale as f64, 0.0)?.exp()?;
    let emb = timesteps
        .unsqueeze(D::Minus1)?
        .broadcast_mul(&emb.unsqueeze(0)?)?;
    let emb_sin = emb.sin()?;
    let emb_cos = emb.cos()?;
    Ok(Tensor::cat(&[&emb_cos, &emb_sin], D::Minus1)?)
}

/// Tensor statistics for debugging (matches Python output format).
///
/// Returns a formatted string with shape, mean, std, min, max.
/// Example: "shape=[1, 5, 896], mean=0.003293, std=0.365257, min=-4.322391, max=5.308450"
pub fn tensor_stats(t: &Tensor) -> String {
    let shape = format!("{:?}", t.dims());

    // Flatten and convert to f32 for stats
    let flat = match t.flatten_all().and_then(|f| f.to_dtype(DType::F32)) {
        Ok(f) => f,
        Err(e) => return format!("shape={}, error computing stats: {}", shape, e),
    };

    let data: Vec<f32> = match flat.to_vec1() {
        Ok(d) => d,
        Err(e) => return format!("shape={}, error converting to vec: {}", shape, e),
    };

    if data.is_empty() {
        return format!("shape={}, (empty tensor)", shape);
    }

    let n = data.len() as f64;
    let sum: f64 = data.iter().map(|&x| x as f64).sum();
    let mean = sum / n;

    let variance: f64 = data
        .iter()
        .map(|&x| {
            let diff = x as f64 - mean;
            diff * diff
        })
        .sum::<f64>()
        / n;
    let std = variance.sqrt();

    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    format!(
        "shape={}, mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
        shape, mean, std, min, max
    )
}

use std::collections::HashMap;
pub fn create_remapped_varbuilder<'a>(
    model_dir: &PathBuf,
    device: &'a Device,
) -> Result<VarBuilder<'a>> {
    let mut shard_files: Vec<PathBuf> = std::fs::read_dir(model_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        })
        .collect();
    shard_files.sort();
    // Load all tensors and remap names
    let mut all_tensors: HashMap<String, Tensor> = HashMap::new();
    let mut encoder_weights_found = Vec::new();
    let mut decoder_weights_found = Vec::new();

    let mut bf16_count = 0;

    for shard_file in &shard_files {
        let tensors = candle_core::safetensors::load(shard_file, device)?;
        for (name, tensor) in tensors {
            // Track encoder and decoder components
            if name.contains("acoustic_tokenizer.encoder") {
                encoder_weights_found.push(name.clone());
            }
            if name.contains("acoustic_tokenizer.decoder") {
                decoder_weights_found.push(name.clone());
            }

            // Convert BF16 weights to F32 to match Python's torch_dtype=torch.float32
            let tensor = if tensor.dtype() == DType::BF16 {
                bf16_count += 1;
                tensor.to_dtype(DType::F32)?
            } else {
                tensor
            };

            let new_name = if name.starts_with("model.language_model.") {
                name.replace("model.language_model.", "model.")
            } else if name.starts_with("model.acoustic_tokenizer.") {
                // Acoustic tokenizer: model.acoustic_tokenizer.X → acoustic_tokenizer.X
                name.strip_prefix("model.").unwrap().to_string()
            } else if name.starts_with("model.acoustic_connector.") {
                // Acoustic connector: model.acoustic_connector.X → acoustic_connector.X
                name.strip_prefix("model.").unwrap().to_string()
            } else if name.starts_with("model.semantic_") {
                // Semantic stuff: model.semantic_X → semantic_X
                name.strip_prefix("model.").unwrap().to_string()
            } else {
                // Everything else (like lm_head): keep as-is
                name
            };
            all_tensors.insert(new_name, tensor);
        }
    }

    debug!("Converted {} BF16 tensors to F32", bf16_count);

    // Validate VAE components
    if decoder_weights_found.is_empty() {
        error!("❌ NO VAE DECODER WEIGHTS FOUND!");
    }
    if encoder_weights_found.is_empty() {
        error!("❌ NO VAE ENCODER WEIGHTS FOUND!");
        warn!("⚠️  Voice cloning will NOT be possible without encoder!");
    }

    debug!(
        "🔍 VAE components: {} encoder weights, {} decoder weights",
        encoder_weights_found.len(),
        decoder_weights_found.len()
    );

    // Use BF16 for GPU, F32 for CPU
    let dtype = DType::F32;

    // Create VarBuilder from remapped tensors
    Ok(VarBuilder::from_tensors(all_tensors, dtype, device))
}

/// Create a VarBuilder for the realtime (streaming) model with remapped tensor names.
///
/// The realtime model stores weights at:
/// - `model.language_model.*` - Lower 4-layer Qwen model (no final norm - uses Identity)
/// - `model.tts_language_model.*` - Upper 20-layer Qwen model
///
/// Qwen2Model expects weights at `model.*` internally, so we remap:
/// - `model.language_model.X` → `model.language_model.model.X` (for Qwen2Model)
/// - `model.tts_language_model.X` → `model.tts_language_model.model.X` (for Qwen2Model)
/// - Other tensors are kept as-is
///
/// Also adds a dummy norm.weight for language_model since Python uses nn.Identity().
/// The norm is skipped during inference via `forward_from_embeds_no_norm()`.
pub fn create_realtime_remapped_varbuilder<'a>(
    model_dir: &Path,
    device: &'a Device,
) -> Result<VarBuilder<'a>> {
    let weights_path = model_dir.join("model.safetensors");

    let tensors = candle_core::safetensors::load(&weights_path, device)?;
    let mut all_tensors: HashMap<String, Tensor> = HashMap::new();

    // Track hidden_size from embed_tokens for creating dummy norm
    let mut hidden_size: Option<usize> = None;

    let mut bf16_count = 0;
    let mut f32_count = 0;

    for (name, tensor) in tensors {
        // Capture hidden_size from embed_tokens
        if name == "model.language_model.embed_tokens.weight" {
            hidden_size = Some(tensor.dim(1)?);
        }

        // Convert BF16 weights to F32 for consistency
        let tensor = if tensor.dtype() == DType::BF16 {
            bf16_count += 1;
            tensor.to_dtype(DType::F32)?
        } else {
            if tensor.dtype() == DType::F32 {
                f32_count += 1;
            }
            tensor
        };

        let new_name = if name.starts_with("model.language_model.") {
            // model.language_model.X → model.language_model.model.X
            name.replace("model.language_model.", "model.language_model.model.")
        } else if name.starts_with("model.tts_language_model.") {
            // model.tts_language_model.X → model.tts_language_model.model.X
            name.replace(
                "model.tts_language_model.",
                "model.tts_language_model.model.",
            )
        } else {
            // Keep other tensors as-is (acoustic_connector, tts_input_types, etc.)
            name
        };
        all_tensors.insert(new_name, tensor);
    }

    debug!(
        "Converted {} BF16 tensors to F32, {} already F32",
        bf16_count, f32_count
    );

    // Add dummy norm.weight for language_model (Python uses nn.Identity())
    // This tensor is required by Qwen2Model::new() but skipped during inference
    if let Some(hs) = hidden_size {
        let dummy_norm = Tensor::ones(&[hs], DType::F32, device)?;
        all_tensors.insert(
            "model.language_model.model.norm.weight".to_string(),
            dummy_norm,
        );
        debug!(
            "Added dummy norm.weight for language_model (hidden_size={})",
            hs
        );
    }

    debug!(
        "Loaded {} tensors from realtime model (remapped for Qwen2Model)",
        all_tensors.len()
    );

    Ok(VarBuilder::from_tensors(all_tensors, DType::F32, device))
}

/// Normalize audio to target dB FS level
/// Matches Python AudioNormalizer exactly:
///   1. tailor_dB_FS: scalar = 10^(target_dB_FS/20) / (rms + eps)
///   2. avoid_clipping: if max > 1.0, divide by max
pub fn normalize_audio_db_fs(audio: &Tensor, target_db_fs: f32) -> Result<Tensor> {
    debug!("Audio normalization input shape: {:?}", audio.dims());

    const EPS: f32 = 1e-6; // Match Python eps

    // Step 1: tailor_dB_FS - Calculate RMS and scale to target dB FS
    let squared = audio.sqr()?;
    let mean = squared.mean_all()?.to_scalar::<f32>()?;
    let rms = mean.sqrt();

    // Convert target dB FS to linear scale: amplitude = 10^(dB_FS / 20)
    let target_amplitude = 10.0_f32.powf(target_db_fs / 20.0);

    // Scale factor with eps to avoid division by zero (matches Python)
    let scale_factor = target_amplitude / (rms + EPS);
    let normalized = audio.affine(scale_factor as f64, 0.0)?;

    // Step 2: avoid_clipping - Scale down if max amplitude > 1.0
    let max_abs = normalized.abs()?.max_all()?.to_scalar::<f32>()?;
    let final_audio = if max_abs > 1.0 {
        let clip_scale = 1.0 / (max_abs + EPS);
        normalized.affine(clip_scale as f64, 0.0)?
    } else {
        normalized
    };

    debug!("Audio normalization output shape: {:?}", final_audio.dims());
    Ok(final_audio)
}
pub fn load_audio_wav(path: &str, target_sample_rate: u32) -> Result<Tensor> {
    debug!("Loading audio from: {}", path);

    let mut reader = hound::WavReader::open(path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to open WAV: {}", e)))?;

    let spec = reader.spec();
    debug!(
        "WAV spec: channels={}, sample_rate={}, bits_per_sample={}",
        spec.channels, spec.sample_rate, spec.bits_per_sample
    );

    // Read samples based on bit depth
    let samples: Vec<f32> = if spec.bits_per_sample == 16 {
        reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect()
    } else if spec.bits_per_sample == 32 {
        reader
            .samples::<i32>()
            .map(|s| s.unwrap() as f32 / 2147483648.0)
            .collect()
    } else {
        return Err(anyhow::Error::msg(format!(
            "Unsupported bit depth: {}. Only 16 and 32 bits are supported.",
            spec.bits_per_sample
        )));
    };

    debug!("Loaded {} raw samples", samples.len());

    // Convert to mono if stereo
    let mono_samples: Vec<f32> = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
            .collect()
    } else if spec.channels == 1 {
        samples
    } else {
        // For multi-channel (>2), average all channels
        let chunk_size = spec.channels as usize;
        samples
            .chunks(chunk_size)
            .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
            .collect()
    };

    debug!("After mono conversion: {} samples", mono_samples.len());

    // Resample if needed
    let final_samples = if spec.sample_rate != target_sample_rate {
        debug!(
            "Resampling from {} Hz to {} Hz",
            spec.sample_rate, target_sample_rate
        );

        let resampled = resample_audio(&mono_samples, spec.sample_rate, target_sample_rate)
            .map_err(|e| candle_core::Error::Msg(format!("Resampling failed: {}", e)))?;

        debug!("After resampling: {} samples", resampled.len());
        resampled
    } else {
        mono_samples
    };

    debug!("Final sample count: {}", final_samples.len());
    debug!(
        "Expected VAE tokens: {}",
        (final_samples.len() as f32 / 3200.0).ceil()
    );

    // Create tensor [1, 1, samples] to match Python's shape
    let tensor = Tensor::from_vec(
        final_samples.clone(),
        (1, 1, final_samples.len()),
        &Device::Cpu,
    )?;

    info!(
        "✓ Loaded audio: {} samples @ {}Hz",
        final_samples.len(),
        target_sample_rate
    );

    Ok(tensor)
}

fn resample_audio(
    samples: &[f32],
    source_rate: u32,
    target_rate: u32,
) -> std::result::Result<Vec<f32>, String> {
    if source_rate == target_rate {
        return Ok(samples.to_vec());
    }

    // Calculate the expected output length (matching librosa's calculation)
    let resample_ratio = target_rate as f64 / source_rate as f64;
    let expected_len = (samples.len() as f64 * resample_ratio).round() as usize;

    // Use high-quality sinc interpolation parameters matching librosa's kaiser_best
    let params = SincInterpolationParameters {
        sinc_len: 256, // High quality
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    // Create resampler
    let mut resampler = SincFixedIn::<f32>::new(
        resample_ratio,
        2.0, // max_resample_ratio_relative
        params,
        samples.len(),
        1, // mono channel
    )
    .map_err(|e| format!("Failed to create resampler: {}", e))?;

    // Prepare input as Vec<Vec<f32>> (channel-first format)
    let input_channels = vec![samples.to_vec()];

    // Perform resampling
    let output_channels = resampler
        .process(&input_channels, None)
        .map_err(|e| format!("Resampling failed: {}", e))?;

    // Extract mono channel
    let mut resampled = output_channels
        .first()
        .ok_or("No output channel from resampler")?
        .clone();

    // Ensure output length matches expected (pad or trim to match librosa's output)
    // This is critical for matching Python's VAE token count
    if resampled.len() < expected_len {
        // Pad with zeros at the end
        resampled.resize(expected_len, 0.0);
    } else if resampled.len() > expected_len {
        // Trim to expected length
        resampled.truncate(expected_len);
    }

    Ok(resampled)
}

/// Initialize file-based logging for test binaries.
///
/// All tracing output will be written to `debug/logs/test_outputs/rust/{test_name}.log`.
/// A single println! indicates the log file location to stdout.
///
/// # Arguments
/// * `test_name` - Name of the test (used for log filename)
/// * `enable_logging` - If false, logging is disabled entirely (no-op tracing subscriber)
///
/// # Returns
/// Path to the log file (even if logging is disabled)
pub fn init_file_logging(test_name: &str) -> PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};

    let log_dir = Path::new("debug/logs/test_outputs/rust");
    std::fs::create_dir_all(log_dir).expect("Failed to create log directory");

    // Get current timestamp
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs();

    let log_path = log_dir.join(format!("{}_{}.log", test_name, timestamp));

    let log_file = File::create(&log_path).expect("Failed to create log file");

    // Use RUST_LOG env var if set, otherwise default to info
    let filter = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());

    tracing_subscriber::registry()
        .with(EnvFilter::new(&filter))
        .with(fmt::layer().with_writer(log_file).with_ansi(false))
        .init();

    println!("📝 Log: {}", log_path.display());

    log_path
}
