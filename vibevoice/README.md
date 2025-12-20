# vibevoice

Core library for VibeVoice text-to-speech with voice cloning and multi-speaker synthesis.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
vibevoice = { git = "https://github.com/danielclough/vibevoice-rs" }
```

## Usage

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

## Running Examples

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

## Models

| Model | Variant | Use Case |
|-------|---------|----------|
| Batch 1.5B | `ModelVariant::Batch1_5B` | Good quality/speed balance (default) |
| Batch 7B | `ModelVariant::Batch7B` | Highest quality, slower |
| Realtime 0.5B | `ModelVariant::Realtime` | Streaming, lowest latency |

## Available Test Files

**Voice samples** (`voices/`):
- `en-Alice_woman.wav`, `en-Carter_man.wav`, `en-Frank_man.wav`, `en-Maya_woman.wav`
- `zh-Bowen_man.wav`, `zh-Xinran_woman.wav`, `in-Samuel_man.wav`

**Scripts** (`text_examples/`):
- `1p.txt` - Single speaker, short
- `2p_short.txt` - Two speakers, short dialogue
- `4p_short.txt` - Four speakers, short
- `2p_music.txt`, `3p_gpt5.txt` - Longer conversations
