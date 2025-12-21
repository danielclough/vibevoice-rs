# vibevoice-web

Leptos 0.8 CSR web frontend for VibeVoice text-to-speech.

## Features

- Model selection (realtime, 1.5B, 7B)
- Voice selection (voice caches for realtime, WAV samples for batch models)
- SSE streaming with progress indicator
- Audio playback and download
- Settings persistence via LocalStorage

## Requirements

- Rust 1.85+
- [Trunk](https://trunkrs.dev/): `cargo install trunk`
- wasm32 target: `rustup target add wasm32-unknown-unknown`

## Development

```bash
cd vibevoice-web
trunk serve
```

Opens at http://127.0.0.1:8908 with hot reload (watches `src/`, `index.html`, `style.css`).

Requires a running vibevoice-server (default: http://localhost:3908).

## Building

```bash
trunk build --release
```

Output is written to `dist/` directory.

## Deployment

### Standalone

Serve the `dist/` directory with any static file server:

```bash
python3 -m http.server -d dist 8908
```

### With vibevoice-server

Point the server's `web_dir` config to the built frontend:

```yaml
# config.yaml
web_dir: /path/to/vibevoice-web/dist
```

The server will serve the frontend at `/`.

### With vibevoice-tauri

The Tauri app embeds and serves the frontend automatically.
