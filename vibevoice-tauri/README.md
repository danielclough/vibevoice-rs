# VibeVoice Tauri

Desktop application for VibeVoice text-to-speech, built with Tauri v2.

## Features

- System tray with quick access menu
- Global hotkey support (`Ctrl+Shift+V` by default)
- Embedded server mode (runs the TTS server locally)
- Remote server mode (connects to a server on another device)
- Desktop notifications on synthesis complete

## Requirements

- Rust 1.85+
- Tauri CLI v2: `cargo install tauri-cli --version "^2"`
- [Trunk](https://trunkrs.dev/): `cargo install trunk`
- System dependencies for Tauri (see [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/))
- wasm32 Target: `rustup target add wasm32-unknown-unknown`

## Development

```bash
# Run in development mode
cargo tauri dev

# With GPU acceleration
cargo tauri dev --features metal      # macOS
cargo tauri dev --features cuda       # Linux/Windows NVIDIA
```

## Building

```bash
# Build release binary
cargo tauri build

# With GPU features
cargo tauri build --features metal    # macOS
cargo tauri build --features linux-gpu  # Linux with CUDA
```

### Debian Aarch64

```sh
sudo dpkg --add-architecture arm64
sudo apt update
sudo apt install libgtk-3-dev:arm64 libglib2.0-dev:arm64 libcairo2-dev:arm64 libpango1.0-dev:arm64 libgdk-pixbuf-2.0-dev:arm64 libatk1.0-dev:arm64
```

## Configuration

The app uses a TOML config file located at:
- **macOS**: `~/Library/Application Support/com.vibevoice.VibeVoice/config.toml`
- **Linux**: `~/.config/vibevoice/config.toml`
- **Windows**: `%APPDATA%\vibevoice\VibeVoice\config.toml`

### Example Configuration

```toml
# Use embedded server (default) or connect to remote
embedded_server = true

# Port for embedded server
server_port = 3908

# Remote server URL (used when embedded_server = false)
remote_server_url = "http://192.168.1.100:3908"

# Voice files directory
voices_dir = "/path/to/voices"
samples_dir = "/path/to/samples"

# Default model: "realtime", "1.5B", or "7B"
default_model = "realtime"

# Global hotkey to show window
hotkey_show = "CommandOrControl+Shift+V"

# Start minimized to system tray
start_minimized = false

# Show notifications when synthesis completes
show_notifications = true
```

### Using a Remote Server

To connect to a VibeVoice server running on another device:

1. Create/edit the config file at the path above
2. Set `embedded_server = false`
3. Set `remote_server_url` to the remote server's address:
   ```toml
   embedded_server = false
   remote_server_url = "http://192.168.1.100:3908"
   ```
4. Update the CSP in `tauri.conf.json` to allow the connection:
   ```json
   "connect-src": "'self' http://192.168.1.11:3908"
   ```

#### Create a config file to disable the embedded server on Linux
```
mkdir ~/.config/vibevoice
cat << EOF > ~/.config/vibevoice/config.toml
embedded_server = false
remote_server_url = "http://192.168.1.11:3908"
EOF
```

## GPU Features

| Feature | Description |
|---------|-------------|
| `metal` | Apple Metal acceleration |
| `cuda` | NVIDIA CUDA support |
| `cudnn` | NVIDIA cuDNN support |
| `flash-attn` | Flash Attention (CUDA only) |
| `accelerate` | Apple Accelerate framework |
| `mkl` | Intel MKL |

### Platform Convenience Features

| Feature | Includes |
|---------|----------|
| `macos` | `metal`, `accelerate` |
| `linux-gpu` | `cuda`, `cudnn`, `flash-attn` |
| `windows-gpu` | `cuda`, `cudnn` |

## Architecture

```
vibevoice-tauri/
├── src/
│   ├── main.rs      # Application entry point
│   ├── config.rs    # Configuration management
│   ├── server.rs    # Embedded server integration
│   ├── tray.rs      # System tray setup
│   └── hotkeys.rs   # Global hotkey handling
├── icons/           # Application icons
└── tauri.conf.json  # Tauri configuration
```

The app embeds `vibevoice-server` and serves the `vibevoice-web` frontend. When `embedded_server` is enabled, the TTS server runs in-process. Otherwise, the frontend connects to the configured remote server.
