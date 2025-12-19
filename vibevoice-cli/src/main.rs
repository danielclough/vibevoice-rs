use anyhow::{Result, anyhow};
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;
use tracing::{debug, info};
use vibevoice::{Device, ModelVariant, VibeVoice};

#[derive(Parser, Debug)]
#[command(name = "vibevoice")]
#[command(
    author,
    version,
    about = "High-quality text-to-speech with voice cloning"
)]
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
    /// Model variant: "1.5B", "7B", or "realtime"
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

    /// Voice sample WAV file for voice cloning (batch models) or voice cache file (realtime)
    #[arg(short, long)]
    voice: Option<String>,

    /// Script file with "Speaker X: text" format for multi-speaker synthesis
    #[arg(short, long)]
    script: Option<String>,

    /// Directory containing voice samples for multi-speaker scripts
    #[arg(long)]
    voices_dir: Option<String>,

    /// CFG (Classifier-Free Guidance) scale for diffusion generation
    #[arg(long, default_value = "1.3")]
    cfg_scale: f32,

    /// Random seed for deterministic output
    #[arg(long, default_value = "524242")]
    seed: u64,

    /// Restore RNG after voice embedding (may help some voices)
    #[arg(long)]
    restore_rng: bool,

    /// Number of diffusion steps (default: 5 for realtime, 10 for batch)
    #[arg(long)]
    steps: Option<usize>,

    /// Enable tracing logs
    #[arg(long)]
    tracing: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Handle subcommands first
    if let Some(command) = cli.command {
        return match command {
            Commands::ConvertVoice { input, output } => {
                // Initialize basic logging for conversion
                tracing_subscriber::fmt()
                    .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
                    .init();

                info!("Voice Cache Conversion");
                info!("────────────────────────────────────────────");

                let report = vibevoice::voice_converter::convert_voice_caches(
                    input.as_deref(),
                    output.as_deref(),
                )?;

                info!(
                    "Conversion complete: {} succeeded, {} failed",
                    report.converted, report.failed
                );

                if !report.failures.is_empty() {
                    return Err(anyhow!(
                        "{} file(s) failed to convert",
                        report.failures.len()
                    ));
                }

                Ok(())
            }
        };
    }

    // Default: run generate/inference mode
    let args = cli.generate;

    if args.tracing {
        _ = vibevoice::init_file_logging("vibevoice");
    }

    // Parse model variant
    let variant = match args.model.to_lowercase().as_str() {
        "1.5b" => ModelVariant::Batch1_5B,
        "7b" => ModelVariant::Batch7B,
        "realtime" | "0.5b" => ModelVariant::Realtime,
        _ => {
            return Err(anyhow!(
                "Invalid model: {}. Use '1.5B', '7B', or 'realtime'",
                args.model
            ));
        }
    };

    info!("VibeVoice Text-to-Speech");
    info!("────────────────────────────────────────────");
    info!("Model: {:?}", variant);

    // Build VibeVoice instance
    let mut builder = VibeVoice::builder()
        .variant(variant)
        .device(Device::auto())
        .seed(args.seed)
        .cfg_scale(args.cfg_scale)
        .restore_rng_after_voice_embedding(args.restore_rng);

    // Use variant-specific default for diffusion steps
    let steps = args.steps.unwrap_or(match variant {
        ModelVariant::Realtime => 5,
        ModelVariant::Batch1_5B | ModelVariant::Batch7B => 10, // Match Python's default
    });
    builder = builder.diffusion_steps(steps);

    let mut vv = builder.build()?;

    // Generate audio
    let audio = if let Some(script_path) = args.script {
        info!("Synthesizing from script: {}", script_path);
        vv.synthesize_script(&script_path, args.voices_dir.as_deref())?
    } else {
        let text_preview = if args.text.len() > 60 {
            format!("{}...", &args.text[..60])
        } else {
            args.text.clone()
        };
        info!("Text: \"{}\"", text_preview);

        let mut chunk_count = 0;
        vv.synthesize_with_callback(
            &args.text,
            args.voice.as_deref(),
            Some(Box::new(move |progress| {
                chunk_count += 1;
                if chunk_count % 10 == 0 {
                    debug!("Generated {} chunks...", progress.step);
                }
            })),
        )?
    };

    // Save output
    audio.save_wav(&args.output)?;

    info!("────────────────────────────────────────────");
    info!(
        "Saved: {} ({:.1}s audio)",
        args.output,
        audio.duration_secs()
    );

    Ok(())
}
