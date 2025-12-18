//! Batch processing example - synthesize multiple texts efficiently.
//!
//! Run with:
//! ```sh
//! cargo run --example batch_processing -p vibevoice --features metal -- "Text one" "Text two" "Text three"
//! cargo run --example batch_processing -p vibevoice --features metal -- --file texts.txt
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

    let mut vv = VibeVoice::new(ModelVariant::Batch1_5B, Device::auto())?;

    println!("Model loaded in {:.1}s", load_start.elapsed().as_secs_f32());
    println!();
    println!("Processing {} texts...", config.texts.len());
    println!();

    let mut total_audio_duration = 0.0;
    let mut total_gen_time = 0.0;

    for (i, text) in config.texts.iter().enumerate() {
        let output_path = if config.texts.len() == 1 {
            config.output_prefix.clone() + ".wav"
        } else {
            format!("{}_{:02}.wav", config.output_prefix, i + 1)
        };

        let preview = if text.len() > 50 {
            format!("{}...", &text[..50])
        } else {
            text.clone()
        };
        println!("[{}/{}] \"{}\"", i + 1, config.texts.len(), preview);

        let gen_start = Instant::now();

        let audio = vv.synthesize(text, config.voice_path.as_deref())?;

        let gen_time = gen_start.elapsed().as_secs_f32();
        total_audio_duration += audio.duration_secs();
        total_gen_time += gen_time;

        audio.save_wav(&output_path)?;

        println!(
            "       -> {} ({:.1}s audio in {:.1}s)",
            output_path,
            audio.duration_secs(),
            gen_time
        );
    }

    println!();
    println!("---");
    println!("Total audio: {:.1}s", total_audio_duration);
    println!("Total time: {:.1}s", total_gen_time);
    println!("Average RTF: {:.2}x", total_audio_duration / total_gen_time);

    Ok(())
}

struct Config {
    texts: Vec<String>,
    voice_path: Option<String>,
    output_prefix: String,
}

fn parse_args(args: &[String]) -> Result<Config, &'static str> {
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        println!("Batch processing example - synthesize multiple texts");
        println!();
        println!("Usage: batch_processing [OPTIONS] <TEXT>...");
        println!("       batch_processing --file <FILE>");
        println!();
        println!("Arguments:");
        println!("  <TEXT>...              One or more texts to synthesize");
        println!();
        println!("Options:");
        println!("  -f, --file <FILE>      Read texts from file (one per line)");
        println!("  -v, --voice <FILE>     Voice sample for all outputs");
        println!("  -o, --output <PREFIX>  Output prefix (default: batch_output)");
        println!("  -h, --help             Show this help");
        println!();
        println!("Examples:");
        println!("  batch_processing \"Hello\" \"World\" \"Test\"");
        println!("  batch_processing --file texts.txt --voice speaker.wav");
        std::process::exit(0);
    }

    let mut texts = Vec::new();
    let mut voice_path = None;
    let mut output_prefix = "batch_output".to_string();
    let mut file_path = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-f" | "--file" => {
                file_path = args.get(i + 1).cloned();
                i += 2;
            }
            "-v" | "--voice" => {
                voice_path = args.get(i + 1).cloned();
                i += 2;
            }
            "-o" | "--output" => {
                output_prefix = args.get(i + 1).cloned().unwrap_or(output_prefix);
                i += 2;
            }
            s if !s.starts_with('-') => {
                texts.push(s.to_string());
                i += 1;
            }
            _ => i += 1,
        }
    }

    // If file specified, read texts from it
    if let Some(path) = file_path {
        let content = std::fs::read_to_string(&path).map_err(|_| "Failed to read texts file")?;
        texts = content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.to_string())
            .collect();
    }

    if texts.is_empty() {
        return Err("No texts provided. Use --help for usage.");
    }

    Ok(Config {
        texts,
        voice_path,
        output_prefix,
    })
}
