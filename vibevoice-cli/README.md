# vibevoice-cli

Command-line interface for VibeVoice text-to-speech with voice cloning.

## Building

```bash
# macOS (Metal + Accelerate)
cargo build --release -p vibevoice-cli --features metal

# Linux with NVIDIA GPU
cargo build --release -p vibevoice-cli --features cuda

# CPU only
cargo build --release -p vibevoice-cli

# Install globally
cargo install --path vibevoice-cli --features metal
```

## Usage

### Basic Text-to-Speech

```bash
vibevoice --text "Hello world" --output hello.wav
```

### Voice Cloning

```bash
vibevoice --text "Hello, this is a cloned voice." --voice path/to/voice_sample.wav
```

### Multi-Speaker Dialogue

Create a script file (e.g., `dialogue.txt`):
```
Speaker 1: Hello, welcome to the show.
Speaker 2: Thanks for having me!
Speaker 1: Let's dive into today's topic.
```

Run with voice samples:
```bash
vibevoice --script dialogue.txt --voices-dir voices/
```

Voices are assigned alphabetically by filename to speakers in order of first appearance.

### Realtime Streaming Model

```bash
# First, convert a voice cache (if needed)
vibevoice convert-voice input.pt output.safetensors

# Then use the realtime model
vibevoice --model realtime --voice output.safetensors --text "Hello world"
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --model` | `1.5B` | Model variant: `1.5B`, `7B`, or `realtime` |
| `-t, --text` | (default message) | Text to synthesize |
| `-o, --output` | `output.wav` | Output audio file path |
| `-v, --voice` | - | Voice sample WAV (batch) or .safetensors cache (realtime) |
| `-s, --script` | - | Script file with "Speaker X: text" format |
| `--voices-dir` | - | Directory containing voice samples for multi-speaker |
| `--cfg-scale` | `1.3` | CFG scale for diffusion |
| `--seed` | `524242` | Random seed for reproducibility |
| `--restore-rng` | `false` | Restore RNG after voice embedding |
| `--steps` | 5 (realtime) / 10 (batch) | Diffusion steps |
| `--tracing` | `false` | Enable logging |

## Subcommands

### convert-voice

Convert voice cache files from PyTorch `.pt` to `.safetensors` format:

```bash
# Convert a single file
vibevoice convert-voice input.pt output.safetensors

# Convert all .pt files in default directory (voices/streaming_model)
vibevoice convert-voice
```

## Models

| Model | Description |
|-------|-------------|
| `1.5B` | Good quality/speed balance (default) |
| `7B` | Highest quality, slower |
| `realtime` | Fastest, streaming support, requires .safetensors voice cache |