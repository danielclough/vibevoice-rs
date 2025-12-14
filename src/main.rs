use anyhow::Result;
use candle_core::{DType, Tensor};
use clap::Parser;
use std::path::{Path, PathBuf};
use std::time::Instant;

use tracing::{debug, error, info, warn};
use vibevoice_rs::model::VibeVoiceModel;
use vibevoice_rs::processor::VibeVoiceProcessor;
use vibevoice_rs::utils::{
    create_remapped_varbuilder, download_model_files, get_device, init_file_logging,
    resolve_voice_path, save_audio_wav, set_all_seeds,
};
use vibevoice_rs::voice_mapper::{VoiceMapper, parse_txt_script};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model variant: "1.5B" or "7B"
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
}
fn main() -> Result<()> {
    let args = Args::parse();

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

    let model_id = match args.model.as_str() {
        "1.5B" | "1.5b" => "vibevoice/VibeVoice-1.5B",
        "7B" | "7b" => "vibevoice/VibeVoice-7B",
        _ => {
            error!(
                "Error: Invalid model variant '{}'. Use '1.5B' or '7B'",
                args.model
            );
            std::process::exit(1);
        }
    };

    info!("╔════════════════════════════════════════════╗");
    info!("║   VibeVoice Inference with Candle         ║");
    info!("║   Model: {:<33}║", model_id.split('/').last().unwrap());
    info!("╚════════════════════════════════════════════╝");

    info!("✓ Device: {:?}", device);

    let load_start = Instant::now();

    info!("📥 Downloading model from HuggingFace...");
    let (model_dir, config_path, tokenizer_path) = download_model_files(model_id)?;
    info!("✓ Model directory: {:?}", model_dir);

    info!("⚙️  Loading model weights...");
    let device_ref = device.clone();
    let vb = create_remapped_varbuilder(&model_dir, &device_ref)?;

    info!("🔧 Initializing {}...", model_id.split('/').last().unwrap());
    let mut model = VibeVoiceModel::new(vb, device.clone(), &config_path, &tokenizer_path)?;
    info!(
        "✓ Model loaded in {:.1}s",
        load_start.elapsed().as_secs_f32()
    );

    // Create processor for input processing
    let processor = VibeVoiceProcessor::from_pretrained(&model_dir, &device)?;
    info!("✓ Processor initialized!");

    model.set_ddpm_inference_steps(10);
    model.set_cfg_scale(args.cfg_scale);
    info!("✓ Diffusion: 10 steps, CFG scale {:.1}", args.cfg_scale);

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
                if let Some(voice_paths) = voice_samples_per_speaker.get(i) {
                    if let Some(voice_path) = voice_paths.first() {
                        info!(
                            "  Speaker {} -> {:?}",
                            speaker,
                            voice_path.file_name().unwrap()
                        );
                    }
                }
            }

            voice_samples_per_speaker.into_iter().flatten().collect()
        };

        let voice_samples = vec![all_voices];

        // Join all scripts into ONE string (like Python's '\n'.join(scripts))
        let full_script = scripts.join("\n");

        (vec![full_script], Some(voice_samples))
    } else {
        // Single text mode
        // Format text to include "Speaker 1:" prefix if not already present
        let text_with_speaker = if args.text.trim().starts_with("Speaker ") {
            args.text.clone()
        } else {
            format!("Speaker 1: {}", args.text.trim())
        };

        info!("📝 Using text: {}", text_with_speaker);

        // Load single voice sample if provided
        let voice_samples = if let Some(voice_input) = &args.voice {
            let resolved_path = resolve_voice_path(voice_input, None)?;
            info!("🎤 Using voice sample: {:?}", resolved_path);
            Some(vec![vec![resolved_path]])
        } else {
            None
        };

        (vec![text_with_speaker], voice_samples)
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
