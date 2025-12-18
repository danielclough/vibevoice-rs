//! Voice cloning example.
//!
//! Run with:
//! ```sh
//! cargo run --example voice_cloning -p vibevoice --features metal -- voice.wav
//! cargo run --example voice_cloning -p vibevoice --features metal -- voice.wav "Custom text"
//! cargo run --example voice_cloning -p vibevoice --features metal -- voice.wav -o output.wav
//! ```

use std::env;
use std::time::Instant;
use vibevoice::{Device, ModelVariant, VibeVoice};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let config = parse_args(&args)?;

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    println!("Loading model...");
    let load_start = Instant::now();

    let mut vv = VibeVoice::builder()
        .variant(ModelVariant::Batch1_5B)
        .device(Device::auto())
        .seed(config.seed)
        .cfg_scale(config.cfg_scale)
        .build()?;

    println!("Model loaded in {:.1}s", load_start.elapsed().as_secs_f32());

    println!("Voice sample: {}", config.voice_path);
    println!("Text: \"{}\"", config.text);
    println!("CFG scale: {:.1}, Seed: {}", config.cfg_scale, config.seed);

    let gen_start = Instant::now();

    let audio = vv.synthesize(&config.text, Some(&config.voice_path))?;

    let gen_time = gen_start.elapsed();
    let rtf = audio.duration_secs() / gen_time.as_secs_f32();

    audio.save_wav(&config.output_path)?;

    println!("---");
    println!("Audio duration: {:.1}s", audio.duration_secs());
    println!("Generation time: {:.1}s", gen_time.as_secs_f32());
    println!("Real-time factor: {:.2}x", rtf);
    println!("Saved to: {}", config.output_path);

    Ok(())
}

struct Config {
    voice_path: String,
    text: String,
    output_path: String,
    cfg_scale: f32,
    seed: u64,
}

fn parse_args(args: &[String]) -> Result<Config, &'static str> {
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        println!("Voice cloning example");
        println!();
        println!("Usage: voice_cloning <VOICE_FILE> [OPTIONS] [TEXT]");
        println!();
        println!("Arguments:");
        println!("  <VOICE_FILE>         Path to voice sample WAV file");
        println!("  [TEXT]               Text to synthesize (default: demo text)");
        println!();
        println!("Options:");
        println!("  -o, --output <FILE>  Output path (default: cloned_output.wav)");
        println!("  --cfg-scale <FLOAT>  CFG scale (default: 1.5)");
        println!("  --seed <INT>         Random seed (default: 42)");
        println!("  -h, --help           Show this help");
        println!();
        println!("Example:");
        println!(
            "  cargo run --example voice_cloning -p vibevoice --features metal -- voice.wav \"Hello world\""
        );
        std::process::exit(0);
    }

    let voice_path = args[0].clone();
    let mut text = "This is my cloned voice speaking. Pretty cool, right?".to_string();
    let mut output_path = "cloned_output.wav".to_string();
    let mut cfg_scale = 1.3;
    let mut seed = 524242;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                output_path = args.get(i + 1).cloned().unwrap_or(output_path);
                i += 2;
            }
            "--cfg-scale" => {
                cfg_scale = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(cfg_scale);
                i += 2;
            }
            "--seed" => {
                seed = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(seed);
                i += 2;
            }
            s if !s.starts_with('-') => {
                text = s.to_string();
                i += 1;
            }
            _ => i += 1,
        }
    }

    Ok(Config {
        voice_path,
        text,
        output_path,
        cfg_scale,
        seed,
    })
}
