# VibeVoice-RS

Rust implementation of VibeVoice text-to-speech with voice cloning and multi-speaker synthesis.

## Features

- High-quality text-to-speech synthesis
- Voice cloning from audio samples
- Multi-speaker dialogue synthesis
- GPU acceleration (Metal/CUDA)
- Streaming audio generation

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
cargo build --release --features metal  # Apple Silicon
cargo build --release --features cuda   # NVIDIA GPU
cargo build --release                   # CPU only
```

## Usage

### Basic Text-to-Speech

```bash
cargo run --release --features cuda -- --text "Hello world"
```

### Voice Cloning (Single Speaker)

```bash
cargo run --release --features metal -- \
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
   cargo run --release --features metal -- --script dialogue.txt
   ```

**Speaker-to-voice mapping**: Voices are assigned alphabetically by filename to speakers in order of first appearance (Speaker 1 → 1st voice, Speaker 2 → 2nd voice, etc.)

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model`, `-m` | `1.5B` | Model variant: `1.5B` or `7B` |
| `--text`, `-t` | - | Text to synthesize |
| `--output`, `-o` | `output.wav` | Output audio file path |
| `--voice`, `-v` | Alphabetical | Voice sample WAV file for cloning (single speaker) |
| `--voices` | Alphabetical | Multiple voice files for multi-speaker (use with `--script`) |
| `--script`, `-s` | - | Script file for multi-speaker synthesis |
| `--cfg-scale` | `1.3` | CFG (Classifier-Free Guidance) scale for diffusion |
| `--seed` | `524242` | Random seed for reproducibility (makes a big difference in final result) |
| `--tracing` | `false` | Enable file logging (writes to `debug/logs/test_outputs/rust/`) |

## Examples

```bash
# Generate with 7B model (higher quality)
cargo run --release --features metal -- --model 7B --text "Hello world" --output hello.wav

# Clone a specific voice
cargo run --release --features metal -- \
  --text "This sounds like the reference speaker." \
  --voice voices/en-Alice_woman.wav

# Multi-speaker podcast
cargo run --release --features metal -- \
  --script text_examples/2p_short.txt \
  --output podcast.wav
```

## Known Issues

### Apple / Metal

> Error: Metal error Failed to create metal resource: Buffer

PyTorch MPS uses optimized SDPA (Scaled Dot Product Attention) that doesn't materialize the full attention matrix. Candle has flash attention but it's CUDA-only, not available for Metal.