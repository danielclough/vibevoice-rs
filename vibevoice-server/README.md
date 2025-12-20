# vibevoice-server

HTTP server for VibeVoice text-to-speech with SSE streaming support.

## Building

```bash
# macOS (Metal + Accelerate)
cargo build --release -p vibevoice-server --features macos

# Linux with NVIDIA GPU
cargo build --release -p vibevoice-server --features linux-gpu

# CPU only
cargo build --release -p vibevoice-server
```

## Configuration

Create a `config.yaml` file (see `config.example.yaml`):

```yaml
# Server binding
host: 0.0.0.0
port: 3000

# Directory containing voice .safetensors files (realtime model)
voices_dir: /path/to/voices/streaming_model

# Directory containing .wav samples for cloning (batch models)
samples_dir: /path/to/wav/samples

# Optional output directory
output_dir: /path/to/output

# Directory containing built vibevoice-web frontend to serve at /
web_dir: /path/to/vibevoice-web/dist
```

## Running

```bash
# With config file
vibevoice-server --config config.yaml

# Override settings via CLI
vibevoice-server --config config.yaml --port 8080 --model realtime --cors
```

### CLI Options

```
Options:
  -c, --config <PATH>    Path to YAML config file
  -p, --port <PORT>      Port (overrides config)
  -m, --model <MODEL>    Default model: 1.5B, 7B, realtime [default: realtime]
      --cors             Enable CORS for all origins
  -h, --help             Print help
```

The `--model` flag sets the default model loaded at startup. Requests can override this per-request (see Runtime Model Switching).

## API Endpoints

### GET /health

Health check.

```bash
curl http://localhost:3000/health
```

Response:
```json
{"status": "ok"}
```

### GET /voices

List available voices from configured directories.

```bash
curl http://localhost:3000/voices
```

Response:
```json
{
  "voices": ["en-Emma_woman", "en-Mike_man", ...],
  "samples": ["sample1", "sample2", ...]
}
```

### POST /synthesize

Synthesize speech, returns WAV audio.

```bash
curl -X POST http://localhost:3000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "en-Emma_woman"}' \
  -o output.wav

# With specific model
curl -X POST http://localhost:3000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "en-Emma_woman", "model": "7B"}' \
  -o output.wav
```

### POST /synthesize/json

Synthesize speech, returns JSON with base64-encoded WAV.

```bash
curl -X POST http://localhost:3000/synthesize/json \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "en-Emma_woman"}'

# With specific model
curl -X POST http://localhost:3000/synthesize/json \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "en-Emma_woman", "model": "1.5B"}'
```

Response:
```json
{
  "audio_base64": "UklGR...",
  "duration_secs": 1.5,
  "sample_rate": 24000
}
```

### POST /synthesize/stream

SSE streaming synthesis. Sends WAV header first, then raw PCM chunks.

```bash
curl -sN -X POST http://localhost:3000/synthesize/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "en-Emma_woman"}'

# With specific model
curl -sN -X POST http://localhost:3000/synthesize/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "en-Emma_woman", "model": "realtime"}'
```

SSE Events:

```
event: header
data: {"wav_header": "<base64 44-byte WAV header>", "sample_rate": 24000}

event: chunk
data: {"step": 1, "pcm_chunk": "<base64 raw PCM>"}

event: chunk
data: {"step": 2, "pcm_chunk": "<base64 raw PCM>"}

event: complete
data: {"duration_secs": 1.5, "total_steps": 10, "total_pcm_bytes": 72000}
```

## Streaming Playback

Pipe SSE stream to ffplay:

```bash
curl -sN -X POST http://localhost:3000/synthesize/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello! This is the realtime streaming model. It generates audio chunk by chunk for low latency applications.", "voice": "en-Emma_woman"}' \
  | python3 scripts/sse_to_wav.py | ffplay -autoexit -nodisp -
```

Or inline:

```bash
curl -sN -X POST http://localhost:3000/synthesize/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "en-Emma_woman"}' \
  | python3 -c "
import sys, json, base64
for line in sys.stdin:
    if line.startswith('data:'):
        try:
            d = json.loads(line[5:])
            if h := d.get('wav_header'):
                sys.stdout.buffer.write(base64.b64decode(h))
            elif c := d.get('pcm_chunk'):
                sys.stdout.buffer.write(base64.b64decode(c))
        except: pass
" | ffplay -autoexit -nodisp -
```

## Browser Usage

```javascript
const eventSource = new EventSource('/synthesize/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'Hello', voice: 'en-Emma_woman' })
});

const chunks = [];
let wavHeader = null;

eventSource.addEventListener('header', (e) => {
  const { wav_header } = JSON.parse(e.data);
  wavHeader = Uint8Array.from(atob(wav_header), c => c.charCodeAt(0));
});

eventSource.addEventListener('chunk', (e) => {
  const { pcm_chunk } = JSON.parse(e.data);
  if (pcm_chunk) {
    chunks.push(Uint8Array.from(atob(pcm_chunk), c => c.charCodeAt(0)));
  }
});

eventSource.addEventListener('complete', (e) => {
  const { total_pcm_bytes } = JSON.parse(e.data);
  // Concatenate: wavHeader + all chunks = valid WAV
  // Play with Web Audio API or create blob URL
});
```

## Voice Resolution

The `voice` field in requests can be:

1. **Absolute path**: Used as-is
   ```json
   {"voice": "/full/path/to/voice.safetensors"}
   ```

2. **Voice name**: Resolved from `voices_dir` or `samples_dir`
   ```json
   {"voice": "en-Emma_woman"}
   ```
   Tries: `voices_dir/en-Emma_woman.safetensors`, then `samples_dir/en-Emma_woman.wav`

## Runtime Model Switching

You can select a different model per-request by including the optional `model` field:

```json
{
  "text": "Hello world",
  "voice": "en-Emma_woman",
  "model": "7B"
}
```

### Available Models

| Model | Value | Description |
|-------|-------|-------------|
| Realtime 0.5B | `"realtime"` | Fastest, supports streaming (default) |
| Batch 1.5B | `"1.5B"` | Good quality/speed balance |
| Batch 7B | `"7B"` | Highest quality, slower |

### Behavior

- If `model` is omitted, uses the server's `--model` CLI default
- Models are loaded lazily on first request (takes a few seconds)
- Once loaded, models stay cached in memory for subsequent requests
- Loading multiple models requires significant VRAM (realtime ~2-4GB, 1.5B ~5-7GB, 7B ~20-24GB)
