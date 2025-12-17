use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use clap::{Args, Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::time::Instant;

use tracing::{debug, error, info, warn};
use vibevoice_rs::model::VibeVoiceModel;
use vibevoice_rs::processor::VibeVoiceProcessor;
use vibevoice_rs::realtime::{VibeVoiceRealtimeModel, VoiceCache};
use vibevoice_rs::utils::{
    create_remapped_varbuilder, download_model_files, download_realtime_model_files, get_device,
    init_file_logging, resolve_voice_path, save_audio_wav, set_all_seeds,
};
use vibevoice_rs::voice_converter;
use vibevoice_rs::voice_mapper::{VoiceMapper, parse_txt_script};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[command(flatten)]
    generate: GenerateArgs,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Convert voice cache files from .pt to .safetensors format
    ConvertVoice {
        /// Input .pt file (if not provided, converts all in voices/streaming_model)
        input: Option<PathBuf>,
        /// Output .safetensors file (auto-generated from input name if not provided)
        output: Option<PathBuf>,
    },
}

#[derive(Args, Debug)]
struct GenerateArgs {
    /// Model variant: "1.5B" or "7B" (batch) or "realtime" (0.5B streaming)
    #[arg(short, long, default_value = "1.5B")]
    model: String,
    /// Text to synthesize
    #[arg(
        short,
        long,
        default_value = "Hello, this is a test of the VibeVoice text-to-speech system."
    )]
    text: String,
    /// Output file path
    #[arg(short, long, default_value = "output.wav")]
    output: String,
    /// Voice sample WAV file for voice cloning (optional, single speaker)
    #[arg(short, long)]
    voice: Option<String>,
    /// Multiple voice samples for multi-speaker synthesis (use with --script)
    /// Order matters: first voice → Speaker 1, second voice → Speaker 2, etc.
    #[arg(long, num_args = 1..)]
    voices: Option<Vec<String>>,
    /// Script file with "Speaker X: text" format for multi-speaker synthesis (optional)
    #[arg(short, long)]
    script: Option<String>,
    /// CFG (Classifier-Free Guidance) scale for diffusion generation
    #[arg(long, default_value = "1.3")]
    cfg_scale: f32,
    /// Random seed for deterministic output (default: 524242)
    #[arg(long, default_value = "524242")]
    seed: u64,
    /// Enable file logging (writes to debug/logs/test_outputs/rust/)
    #[arg(long, default_value = "false")]
    tracing: bool,

    // === Realtime model options ===
    /// Use realtime (0.5B streaming) model
    #[arg(long)]
    realtime: bool,
    /// Pre-computed voice cache file (safetensors) for realtime mode
    #[arg(long)]
    voice_cache: Option<PathBuf>,
    /// Path to realtime model directory (optional, will download if not provided)
    #[arg(long)]
    realtime_model_path: Option<PathBuf>,
    /// Number of diffusion steps for realtime model (default: 5)
    #[arg(long, default_value = "5")]
    steps: usize,
}
fn main() -> Result<()> {
    let cli = Cli::parse();

    // Handle subcommands first
    if let Some(command) = cli.command {
        return match command {
            Commands::ConvertVoice { input, output } => {
                run_convert_voice(input.as_deref(), output.as_deref())
            }
        };
    }

    // Default: run generate/inference mode
    let args = cli.generate;

    // Initialize file-based logging (only if --tracing flag is set)

    if args.tracing {
        let _log_path = init_file_logging("vibevoice");
    }

    // ============================================================================
    // CRITICAL: Set all random seeds FIRST for deterministic output
    // ============================================================================
    info!("🎲 Setting random seed to {}", args.seed);
    let device = get_device(Some(0))?;
    set_all_seeds(args.seed, &device)?;
    info!("✓ All random seeds set for deterministic output");

    // Check for realtime mode
    if args.realtime || args.model.to_lowercase() == "realtime" {
        return run_realtime_inference(&args, &device);
    }

    let model_id = match args.model.as_str() {
        "1.5B" | "1.5b" => "vibevoice/VibeVoice-1.5B",
        "7B" | "7b" => "vibevoice/VibeVoice-7B",
        _ => {
            error!(
                "Error: Invalid model variant '{}'. Use '1.5B', '7B', or 'realtime'",
                args.model
            );
            std::process::exit(1);
        }
    };

    info!("╔════════════════════════════════════════════╗");
    info!("║   VibeVoice Inference with Candle         ║");
    info!(
        "║   Model: {:<33}║",
        model_id.split('/').next_back().unwrap()
    );
    info!("╚════════════════════════════════════════════╝");

    info!("✓ Device: {:?}", device);

    let load_start = Instant::now();

    info!("📥 Downloading model from HuggingFace...");
    let (model_dir, config_path, tokenizer_path) = download_model_files(model_id)?;
    info!("✓ Model directory: {:?}", model_dir);

    info!("⚙️  Loading model weights...");
    let device_ref = device.clone();
    let vb = create_remapped_varbuilder(&model_dir, &device_ref)?;

    info!(
        "🔧 Initializing {}...",
        model_id.split('/').next_back().unwrap()
    );
    let mut model = VibeVoiceModel::new(vb, device.clone(), &config_path, &tokenizer_path)?;
    info!(
        "✓ Model loaded in {:.1}s",
        load_start.elapsed().as_secs_f32()
    );

    // Create processor for input processing
    let processor = VibeVoiceProcessor::from_pretrained(&model_dir, &device)?;
    info!("✓ Processor initialized!");

    // Batch model uses 20 steps from config (matching Python default)
    model.set_cfg_scale(args.cfg_scale);
    info!(
        "✓ Diffusion: {} steps, CFG scale {:.1}",
        model.num_diffusion_steps(),
        args.cfg_scale
    );

    // === HANDLE SCRIPT FILE OR TEXT INPUT ===
    let (texts, voice_samples) = if let Some(script_path) = &args.script {
        // Script mode: parse script file and map speakers to voices
        info!("📄 Reading script from: {}", script_path);
        let script_content = std::fs::read_to_string(script_path)?;

        let (scripts, speaker_numbers) = parse_txt_script(&script_content)?;
        info!("Found {} speaker segments:", scripts.len());
        for (i, (speaker, text)) in speaker_numbers.iter().zip(scripts.iter()).enumerate() {
            let preview = if text.len() > 50 {
                format!("{}...", &text[..50])
            } else {
                text.clone()
            };
            info!("  {}. Speaker {}", i + 1, speaker);
            info!("     Text preview: {}", preview);
        }

        // Get unique speakers (in order of first appearance)
        let mut seen = std::collections::HashSet::new();
        let unique_speakers: Vec<_> = speaker_numbers
            .iter()
            .filter(|s| seen.insert(s.to_string()))
            .collect();

        // Get voice files: prefer explicit --voices or --voice, fall back to VoiceMapper
        let script_dir = Path::new(script_path).parent();

        // Collect explicit voice paths from either --voices or --voice
        let explicit_voice_inputs: Option<Vec<String>> = if let Some(voice_paths) = &args.voices {
            Some(voice_paths.clone())
        } else if let Some(voice_path) = &args.voice {
            Some(vec![voice_path.clone()])
        } else {
            None
        };

        let all_voices: Vec<PathBuf> = if let Some(voice_inputs) = explicit_voice_inputs {
            // Explicit voices provided via CLI - use in order
            info!("🎤 Using {} explicit voice files:", voice_inputs.len());
            let voices: Vec<PathBuf> = voice_inputs
                .iter()
                .map(|v| resolve_voice_path(v, script_dir))
                .collect::<Result<Vec<_>>>()?;

            if voices.len() < unique_speakers.len() {
                warn!(
                    "⚠️ Only {} voices provided for {} unique speakers - some speakers will share voices",
                    voices.len(),
                    unique_speakers.len()
                );
            } else if voices.len() > unique_speakers.len() {
                info!(
                    "ℹ️ {} voices provided for {} speakers - only using first {} voices",
                    voices.len(),
                    unique_speakers.len(),
                    unique_speakers.len()
                );
            }

            // Only use as many voices as there are unique speakers (matching Python behavior)
            let voices_to_use: Vec<PathBuf> = unique_speakers
                .iter()
                .enumerate()
                .map(|(i, speaker)| {
                    let voice_path = voices.get(i).unwrap_or(&voices[0]);
                    info!(
                        "  Speaker {} -> {:?}",
                        speaker,
                        voice_path.file_name().unwrap_or_default()
                    );
                    voice_path.clone()
                })
                .collect();

            voices_to_use
        } else {
            // No explicit voices - use VoiceMapper to scan directory
            let voices_dir = if let Some(parent) = Path::new(script_path).parent() {
                let demo_voices = parent.join("voices");
                if demo_voices.exists() {
                    demo_voices
                } else {
                    PathBuf::from("./voices")
                }
            } else {
                PathBuf::from("./voices")
            };

            info!("🎤 Scanning voices directory: {:?}", voices_dir);
            let voice_mapper = VoiceMapper::new(&voices_dir)?;
            let voice_samples_per_speaker =
                voice_mapper.map_speakers_to_voices(&speaker_numbers)?;

            info!("\nSpeaker mapping ({} unique):", unique_speakers.len());
            for (i, speaker) in unique_speakers.iter().enumerate() {
                if let Some(voice_paths) = voice_samples_per_speaker.get(i)
                    && let Some(voice_path) = voice_paths.first()
                {
                    info!(
                        "  Speaker {} -> {:?}",
                        speaker,
                        voice_path.file_name().unwrap()
                    );
                }
            }

            voice_samples_per_speaker.into_iter().flatten().collect()
        };

        let voice_samples = vec![all_voices];

        // Join all scripts into ONE string (like Python's '\n'.join(scripts))
        let full_script = scripts.join("\n");

        (vec![full_script], Some(voice_samples))
    } else {
        // Single text mode - use same path as script mode
        // Format text to include "Speaker 1:" prefix if not already present
        let text_with_speaker = if args.text.trim().starts_with("Speaker ") {
            args.text.clone()
        } else {
            format!("Speaker 1: {}", args.text.trim())
        };

        info!("📝 Parsing text input...");

        // Parse through same path as script mode
        let (scripts, speaker_numbers) = parse_txt_script(&text_with_speaker)?;

        // Get unique speakers (in order of first appearance)
        let mut seen = std::collections::HashSet::new();
        let unique_speakers: Vec<_> = speaker_numbers
            .iter()
            .filter(|s| seen.insert(s.to_string()))
            .collect();

        info!("Found {} speaker segments:", scripts.len());
        for (i, (speaker, text)) in speaker_numbers.iter().zip(scripts.iter()).enumerate() {
            let preview = if text.len() > 50 {
                format!("{}...", &text[..50])
            } else {
                text.clone()
            };
            info!("  {}. Speaker {}", i + 1, speaker);
            info!("     Text preview: {}", preview);
        }

        // Load voice samples if provided
        let voice_samples = if let Some(voice_input) = &args.voice {
            let resolved_path = resolve_voice_path(voice_input, None)?;
            info!("🎤 Using voice sample: {:?}", resolved_path);

            // Create voice list for all unique speakers (reuse same voice)
            let all_voices: Vec<PathBuf> = unique_speakers
                .iter()
                .map(|_| resolved_path.clone())
                .collect();

            Some(vec![all_voices])
        } else if let Some(voice_paths) = &args.voices {
            // Multiple explicit voices provided
            let voices: Vec<PathBuf> = voice_paths
                .iter()
                .map(|v| resolve_voice_path(v, None))
                .collect::<Result<Vec<_>>>()?;

            let all_voices: Vec<PathBuf> = unique_speakers
                .iter()
                .enumerate()
                .map(|(i, _)| voices.get(i).unwrap_or(&voices[0]).clone())
                .collect();

            Some(vec![all_voices])
        } else {
            None
        };

        // Join all scripts into ONE string (like script mode)
        let full_script = scripts.join("\n");

        (vec![full_script], voice_samples)
    };

    // === PROCESS INPUTS ===
    info!("🔄 Processing inputs...");
    let processed_inputs = processor.process(
        texts,
        voice_samples,
        false, // padding
        true,  // return_attention_mask
    )?;

    info!("✓ Inputs processed, tokenizing...");
    let input_ids = processed_inputs.input_ids;
    let attention_mask = processed_inputs.attention_mask;
    let speech_input_mask = processed_inputs.speech_input_mask;
    let voice_embeds = processed_inputs.voice_embeds;
    let speech_masks = processed_inputs.speech_masks;

    // === GENERATE AUDIO ===
    info!("🎵 Generating audio...");
    let gen_start = Instant::now();

    let audio_chunks = model.generate_processed(
        &input_ids,
        voice_embeds.as_ref(),
        speech_input_mask.as_ref(),
        attention_mask.as_ref(),
        speech_masks.as_ref(),
        None::<usize>,
        true,
        None::<fn(&Tensor) -> Result<()>>,
    )?;

    let gen_elapsed = gen_start.elapsed().as_secs_f32();

    if audio_chunks.is_empty() {
        warn!("No audio chunks generated!");
    } else {
        for (i, chunk) in audio_chunks.iter().enumerate() {
            debug!("   Chunk {}: shape={:?}", i, chunk.dims());
        }

        let chunk_refs: Vec<&Tensor> = audio_chunks.iter().collect();
        let concatenated = Tensor::cat(&chunk_refs, 2)?;

        let total_samples = concatenated.dims().iter().product::<usize>();
        let audio_duration = total_samples as f32 / 24000.0;

        // Audio stats at debug level (F16 compatibility)
        if let Ok(audio_vec) = concatenated
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()
        {
            let mean = audio_vec.iter().sum::<f32>() / audio_vec.len() as f32;
            let max = audio_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = audio_vec.iter().cloned().fold(f32::INFINITY, f32::min);
            debug!(
                "   Audio stats: mean={:.6}, range=[{:.4}, {:.4}]",
                mean, min, max
            );
        }

        info!(
            "✓ Generated {} chunks ({:.1}s audio) in {:.1}s",
            audio_chunks.len(),
            audio_duration,
            gen_elapsed
        );

        save_audio_wav(&concatenated, &args.output, 24000)?;
        info!("💾 Saved to {}", args.output);
    }

    info!("✅ Complete!");

    Ok(())
}

/// HuggingFace repo ID for the realtime model.
const REALTIME_MODEL_REPO: &str = "VibeVoice/VibeVoice-Realtime-0.5B";

/// Run inference with the realtime (0.5B streaming) model.
fn run_realtime_inference(args: &GenerateArgs, device: &Device) -> Result<()> {
    info!("╔════════════════════════════════════════════╗");
    info!("║   VibeVoice Realtime (0.5B Streaming)     ║");
    info!("╚════════════════════════════════════════════╝");

    // Check for required voice cache
    let voice_cache_path = args.voice_cache.as_ref().ok_or_else(|| {
        anyhow!(
            "--voice-cache is required for realtime mode.\n\
             Convert a voice cache with: vibevoice-rs convert-voice\n\
             Or use: python scripts/convert_voice_cache.py input.pt output.safetensors"
        )
    })?;

    // Get model path - download from HuggingFace if not provided
    let model_path = if let Some(path) = &args.realtime_model_path {
        path.clone()
    } else {
        info!("📥 Downloading realtime model from HuggingFace...");
        download_realtime_model_files(REALTIME_MODEL_REPO)?
    };

    let load_start = Instant::now();

    // Load voice cache
    info!("📥 Loading voice cache from {:?}...", voice_cache_path);
    let voice_cache = VoiceCache::from_safetensors(voice_cache_path, device)?;
    info!("{}", voice_cache.summary()?);

    // Load model
    info!("⚙️  Loading realtime model from {:?}...", model_path);
    let mut model = VibeVoiceRealtimeModel::from_pretrained(&model_path, device)?;

    // Override diffusion steps from CLI (default: 5, matching Python)
    model.set_diffusion_steps(args.steps);

    info!(
        "✓ Model loaded in {:.1}s ({} diffusion steps)",
        load_start.elapsed().as_secs_f32(),
        model.num_diffusion_steps()
    );

    // Get text to synthesize
    let text = if let Some(script_path) = &args.script {
        std::fs::read_to_string(script_path)?
    } else {
        args.text.clone()
    };

    info!("📝 Text: \"{}\"", &text[..text.len().min(100)]);
    if text.len() > 100 {
        info!("   ... ({} total chars)", text.len());
    }

    // Generate audio
    let gen_start = Instant::now();
    info!("🎵 Generating audio (streaming)...");

    let mut chunk_count = 0;
    let audio = model.generate(&text, &voice_cache, |_chunk| {
        chunk_count += 1;
        if chunk_count % 10 == 0 {
            debug!("  Generated {} chunks...", chunk_count);
        }
        Ok(())
    })?;

    let gen_elapsed = gen_start.elapsed().as_secs_f32();
    let audio_samples = audio.dim(2)?;
    let audio_duration = audio_samples as f32 / 24000.0;

    info!(
        "✓ Generated {} audio samples ({:.1}s) in {:.1}s",
        audio_samples, audio_duration, gen_elapsed
    );
    info!("   Real-time factor: {:.2}x", audio_duration / gen_elapsed);

    // Save audio
    save_audio_wav(&audio, &args.output, 24000)?;
    info!("💾 Saved to {}", args.output);

    info!("✅ Complete!");
    Ok(())
}

/// Run voice cache conversion from .pt to .safetensors format.
fn run_convert_voice(input: Option<&Path>, output: Option<&Path>) -> Result<()> {
    // Initialize basic logging for conversion
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    info!("╔════════════════════════════════════════════╗");
    info!("║   Voice Cache Conversion                  ║");
    info!("╚════════════════════════════════════════════╝");

    if let Some(inp) = input {
        info!("Input: {:?}", inp);
        if let Some(out) = output {
            info!("Output: {:?}", out);
        }
    } else {
        info!(
            "Converting all .pt files in {}",
            voice_converter::DEFAULT_VOICE_DIR
        );
    }

    let report = voice_converter::convert_voice_caches(input, output)?;

    // Print summary
    info!("────────────────────────────────────────────");
    info!(
        "Conversion complete: {} succeeded, {} failed",
        report.converted, report.failed
    );

    if !report.success_paths.is_empty() {
        info!("Converted files:");
        for path in &report.success_paths {
            info!("  ✓ {:?}", path.file_name().unwrap_or_default());
        }
    }

    if !report.failures.is_empty() {
        warn!("Failed conversions:");
        for (path, error) in &report.failures {
            warn!("  ✗ {:?}: {}", path.file_name().unwrap_or_default(), error);
        }
        return Err(anyhow!(
            "{} file(s) failed to convert",
            report.failures.len()
        ));
    }

    info!("✅ Complete!");
    Ok(())
}
