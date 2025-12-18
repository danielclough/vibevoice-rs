//! Simple text-to-speech example.
//!
//! Run with:
//! ```sh
//! cargo run --example simple_tts -p vibevoice --features metal
//! cargo run --example simple_tts -p vibevoice --features metal -- "Custom text here"
//! cargo run --example simple_tts -p vibevoice --features metal -- --model 7B "Higher quality"
//! ```

use std::env;
use std::time::Instant;
use vibevoice::{Device, ModelVariant, VibeVoice};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse simple arguments
    let args: Vec<String> = env::args().skip(1).collect();

    let (model_variant, text) = parse_args(&args);

    // Initialize tracing for logs (set RUST_LOG=info for details)
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    println!("Loading {:?} model...", model_variant);
    let load_start = Instant::now();

    let mut vv = VibeVoice::new(model_variant, Device::auto())?;

    let load_time = load_start.elapsed();
    println!("Model loaded in {:.1}s", load_time.as_secs_f32());

    println!("Generating speech for: \"{}\"", text);
    let gen_start = Instant::now();

    let audio = vv.synthesize(&text, None)?;

    let gen_time = gen_start.elapsed();
    let rtf = audio.duration_secs() / gen_time.as_secs_f32();

    audio.save_wav("output/simple_tts.wav")?;

    println!("---");
    println!("Audio duration: {:.1}s", audio.duration_secs());
    println!("Generation time: {:.1}s", gen_time.as_secs_f32());
    println!("Real-time factor: {:.2}x", rtf);
    println!("Saved to: output/simple_tts.wav");

    Ok(())
}

fn parse_args(args: &[String]) -> (ModelVariant, String) {
    let default_text = "Hello! This is a test of the VibeVoice text-to-speech system.";

    if args.is_empty() {
        return (ModelVariant::Batch1_5B, default_text.to_string());
    }

    let mut model = ModelVariant::Batch1_5B;
    let mut text_parts = Vec::new();
    let mut skip_next = false;

    for (i, arg) in args.iter().enumerate() {
        if skip_next {
            skip_next = false;
            continue;
        }

        match arg.as_str() {
            "--model" | "-m" => {
                if let Some(m) = args.get(i + 1) {
                    model = match m.to_lowercase().as_str() {
                        "7b" => ModelVariant::Batch7B,
                        "realtime" | "0.5b" => ModelVariant::Realtime,
                        _ => ModelVariant::Batch1_5B,
                    };
                    skip_next = true;
                }
            }
            "--help" | "-h" => {
                println!("Usage: simple_tts [OPTIONS] [TEXT]");
                println!();
                println!("Options:");
                println!("  -m, --model <MODEL>  Model variant: 1.5B (default), 7B, realtime");
                println!("  -h, --help           Show this help");
                println!();
                println!("Example:");
                println!(
                    "  cargo run --example simple_tts -p vibevoice --features metal -- \"Hello world\""
                );
                std::process::exit(0);
            }
            _ => text_parts.push(arg.clone()),
        }
    }

    let text = if text_parts.is_empty() {
        default_text.to_string()
    } else {
        text_parts.join(" ")
    };

    (model, text)
}
