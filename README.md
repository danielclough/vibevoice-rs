# VibeVoice-RS

Rust implementation of VibeVoice text-to-speech with voice cloning and multi-speaker synthesis.

## Features

- High-quality text-to-speech synthesis
- Voice cloning from audio samples
- Multi-speaker dialogue synthesis
- GPU acceleration (Metal/CUDA)
- Streaming audio generation (realtime model)

## Crate Structure

```
vibevoice-rs/
├── vibevoice/        # Library crate
├── vibevoice-cli/    # CLI binary
└── examples/         # Example code
```

## Requirements

- Rust 1.75+
- HuggingFace account and token
- GPU recommended (Metal on Apple Silicon, CUDA on NVIDIA)

## Setup

### HuggingFace Token (Required)

```bash
# Create the cache directory
mkdir -p ~/.cache/huggingface

# Paste your token (get from https://huggingface.co/settings/tokens)
echo "hf_yourTokenHere" > ~/.cache/huggingface/token

# Secure it
chmod 600 ~/.cache/huggingface/token
```

### Build

```bash
# Build CLI
cargo build -p vibevoice-cli --release --features metal  # Apple Silicon
cargo build -p vibevoice-cli --release --features cuda   # NVIDIA GPU
cargo build -p vibevoice-cli --release                   # CPU only

# Or install globally
cargo install --path vibevoice-cli --features metal
```

## Library Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
vibevoice = { git = "https://github.com/danielclough/vibevoice-rs" }
```

### Basic Example

```rust
use vibevoice::{VibeVoice, ModelVariant, Device};

fn main() -> Result<(), vibevoice::VibeVoiceError> {
    // Create with default 1.5B model
    let mut vv = VibeVoice::new(ModelVariant::Batch1_5B, Device::auto())?;

    // Synthesize speech
    let audio = vv.synthesize("Hello, world!", None)?;
    audio.save_wav("output.wav")?;

    Ok(())
}
```

### Voice Cloning

```rust
let audio = vv.synthesize("Hello!", Some("path/to/voice.wav"))?;
```

### Builder Pattern

```rust
let mut vv = VibeVoice::builder()
    .variant(ModelVariant::Realtime)
    .device(Device::Metal)
    .seed(42)
    .cfg_scale(1.3)
    .diffusion_steps(5)
    .build()?;
```

### Run Examples

All examples support `--help` for usage details.

```bash
# Simple TTS
cargo run --release --example simple_tts -p vibevoice --features metal
cargo run --release --example simple_tts -p vibevoice --features metal -- "Hello world"

# Voice cloning
cargo run --release --example voice_cloning -p vibevoice --features metal -- voices/en-Alice_woman.wav
cargo run --release --example voice_cloning -p vibevoice --features metal -- voices/en-Carter_man.wav "Custom text"

# Multi-speaker dialogue
cargo run --release --example multi_speaker -p vibevoice --features metal -- text_examples/2p_short.txt
cargo run --release --example multi_speaker -p vibevoice --features metal -- text_examples/4p_short.txt --voices-dir voices/

# Batch processing
cargo run --release --example batch_processing -p vibevoice --features metal -- "Text one" "Text two"

# Streaming realtime model (requires voice cache)
cargo run --release --example streaming -p vibevoice --features metal -- voices/streaming_model/jp-Spk1_woman.safetensors
```

**Note**: Replace `--features metal` with `--features cuda` for NVIDIA GPUs.

### Available Test Files

**Voice samples** (`voices/`):
- `en-Alice_woman.wav`, `en-Carter_man.wav`, `en-Frank_man.wav`, `en-Maya_woman.wav`
- `zh-Bowen_man.wav`, `zh-Xinran_woman.wav`, `in-Samuel_man.wav`

**Scripts** (`text_examples/`):
- `1p.txt` - Single speaker, short
- `2p_short.txt` - Two speakers, short dialogue
- `4p_short.txt` - Four speakers, short
- `2p_music.txt`, `3p_gpt5.txt` - Longer conversations

## CLI Usage

### Basic Text-to-Speech

```bash
vibevoice --text "Hello world" --output hello.wav
```

### Voice Cloning (Single Speaker)

```bash
vibevoice \
  --text "Hello, this is a cloned voice." \
  --voice path/to/voice_sample.wav
```

### Multi-Speaker Dialogue

1. Create a script file (e.g., `dialogue.txt`):
   ```
   Speaker 1: Hello, welcome to the show.
   Speaker 2: Thanks for having me!
   Speaker 1: Let's dive into today's topic.
   ```

2. Place voice files in a `voices/` directory.

3. Run:
   ```bash
   vibevoice --script dialogue.txt
   ```

**Speaker-to-voice mapping**: Voices are assigned alphabetically by filename to speakers in order of first appearance (Speaker 1 -> 1st voice, Speaker 2 -> 2nd voice, etc.)

### Realtime Streaming Model

```bash
# First, convert a voice cache
vibevoice convert-voice input.pt output.safetensors

# Then use the realtime model
vibevoice --model realtime --voice-cache output.safetensors --text "Hello world"
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model`, `-m` | `1.5B` | Model variant: `1.5B`, `7B`, or `realtime` |
| `--text`, `-t` | - | Text to synthesize |
| `--output`, `-o` | `output.wav` | Output audio file path |
| `--voice`, `-v` | - | Voice sample WAV file for cloning |
| `--script`, `-s` | - | Script file for multi-speaker synthesis |
| `--voices-dir` | - | Directory containing voice samples |
| `--cfg-scale` | `1.3` | CFG scale for diffusion |
| `--seed` | `524242` | Random seed for reproducibility |
| `--steps` | `5` | Diffusion steps (realtime model only) |
| `--tracing` | `false` | Enable logging |

### Subcommands

```bash
# Convert voice cache from PyTorch to safetensors
vibevoice convert-voice [INPUT] [OUTPUT]
```

## Known Issues

### Apple / Metal

Very long inputs may run over the buffer.

> Error: Metal error Failed to create metal resource: Buffer

PyTorch MPS uses optimized SDPA (Scaled Dot Product Attention) that doesn't materialize the full attention matrix. Candle has flash attention but it's CUDA-only, not available for Metal.
