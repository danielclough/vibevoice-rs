//! Streaming realtime model example.
//!
//! The realtime model generates audio chunk-by-chunk for low latency applications.
//!
//! Run with:
//! ```sh
//! cargo run --example streaming -p vibevoice --features metal -- voice_cache.safetensors
//! cargo run --example streaming -p vibevoice --features metal -- cache.safetensors "Custom text"
//! ```
//!
//! Convert a voice cache first:
//! ```sh
//! cargo run -p vibevoice-cli --features metal -- convert-voice input.pt output.safetensors
//! ```

use std::env;
use std::io::Write;
use std::time::Instant;
use vibevoice::{Device, ModelVariant, Progress, VibeVoice};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();

    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        println!("Streaming realtime model example");
        println!();
        println!("Usage: streaming <VOICE_CACHE> [TEXT]");
        println!();
        println!("Arguments:");
        println!("  <VOICE_CACHE>  Path to voice cache (.safetensors)");
        println!("  [TEXT]         Text to synthesize");
        println!();
        println!("Convert a voice cache first:");
        println!(
            "  cargo run -p vibevoice-cli --features metal -- convert-voice input.pt output.safetensors"
        );
        println!();
        println!("Example:");
        println!(
            "  cargo run --example streaming -p vibevoice --features metal -- cache.safetensors"
        );
        return Ok(());
    }

    let voice_cache_path = &args[0];
    let text = args.get(1).map(|s| s.as_str()).unwrap_or(
        "Hello! This is the realtime streaming model. It generates audio chunk by chunk for low latency applications."
    );

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    println!("Loading realtime model...");
    let load_start = Instant::now();

    let mut vv = VibeVoice::builder()
        .variant(ModelVariant::Realtime)
        .device(Device::auto())
        .diffusion_steps(5)
        .build()?;

    println!("Model loaded in {:.1}s", load_start.elapsed().as_secs_f32());
    println!("Voice cache: {}", voice_cache_path);
    println!("Text: \"{}\"", &text[..text.len().min(60)]);
    if text.len() > 60 {
        println!("      ... ({} chars total)", text.len());
    }
    println!();
    println!("Generating with streaming progress:");

    let gen_start = Instant::now();

    let audio = vv.synthesize_with_callback(
        text,
        Some(voice_cache_path),
        Some(Box::new(move |progress: Progress| {
            print!("\r  Chunks: {} ", progress.step);
            std::io::stdout().flush().ok();
        })),
    )?;

    let gen_time = gen_start.elapsed();
    let rtf = audio.duration_secs() / gen_time.as_secs_f32();

    println!();
    println!();

    audio.save_wav("streaming_output.wav")?;

    println!("---");
    println!("Audio duration: {:.1}s", audio.duration_secs());
    println!("Generation time: {:.1}s", gen_time.as_secs_f32());
    println!("Real-time factor: {:.2}x", rtf);
    println!("Saved to: streaming_output.wav");

    if rtf > 1.0 {
        println!();
        println!("Faster than real-time - suitable for streaming!");
    }

    Ok(())
}
