//! Multi-speaker dialogue synthesis example.
//!
//! Run with:
//! ```sh
//! cargo run --example multi_speaker -p vibevoice --features metal -- script.txt
//! cargo run --example multi_speaker -p vibevoice --features metal -- script.txt --voices-dir voices/
//! ```
//!
//! Script format (script.txt):
//! ```text
//! Speaker 1: Hello, welcome to the show.
//! Speaker 2: Thanks for having me!
//! Speaker 1: Let's dive into today's topic.
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

    // Read and preview script
    let script_content = std::fs::read_to_string(&config.script_path)?;
    let lines: Vec<&str> = script_content.lines().take(5).collect();
    println!("Script: {}", config.script_path);
    for line in &lines {
        println!("  {}", line);
    }
    if script_content.lines().count() > 5 {
        println!("  ... ({} lines total)", script_content.lines().count());
    }

    if let Some(ref dir) = config.voices_dir {
        println!("Voices directory: {}", dir);
    }

    println!();
    println!("Generating multi-speaker audio...");
    let gen_start = Instant::now();

    let audio = vv.synthesize_script(&config.script_path, config.voices_dir.as_deref())?;

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
    script_path: String,
    voices_dir: Option<String>,
    output_path: String,
    cfg_scale: f32,
    seed: u64,
}

fn parse_args(args: &[String]) -> Result<Config, &'static str> {
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        println!("Multi-speaker dialogue synthesis example");
        println!();
        println!("Usage: multi_speaker <SCRIPT_FILE> [OPTIONS]");
        println!();
        println!("Arguments:");
        println!("  <SCRIPT_FILE>           Path to script file with 'Speaker N: text' format");
        println!();
        println!("Options:");
        println!("  --voices-dir <DIR>      Directory containing voice samples");
        println!("  -o, --output <FILE>     Output path (default: dialogue_output.wav)");
        println!("  --cfg_scale <FLOAT>     CFG scale (default: 1.3)");
        println!("  --seed <INT>            Random seed (default: 524242)");
        println!("  -h, --help              Show this help");
        println!();
        println!("Script format:");
        println!("  Speaker 1: Hello, welcome to the show.");
        println!("  Speaker 2: Thanks for having me!");
        println!();
        println!("Voice mapping:");
        println!("  Voices are mapped to speakers alphabetically by filename.");
        println!("  Place voice files (e.g., 1_alice.wav, 2_bob.wav) in the voices directory.");
        println!();
        println!("Example:");
        println!(
            "  cargo run --example multi_speaker -p vibevoice --features metal -- dialogue.txt"
        );
        std::process::exit(0);
    }

    let script_path = args[0].clone();
    let mut voices_dir = None;
    let mut output_path = "dialogue_output.wav".to_string();
    let mut cfg_scale = 1.3;
    let mut seed = 524242;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--voices-dir" => {
                voices_dir = args.get(i + 1).cloned();
                i += 2;
            }
            "-o" | "--output" => {
                output_path = args.get(i + 1).cloned().unwrap_or(output_path);
                i += 2;
            }
            "--cfg_scale" => {
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
            _ => i += 1,
        }
    }

    Ok(Config {
        script_path,
        voices_dir,
        output_path,
        cfg_scale,
        seed,
    })
}
