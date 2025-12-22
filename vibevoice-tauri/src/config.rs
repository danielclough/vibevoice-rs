//! Platform-specific configuration paths and settings.

use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Failed to determine config directory")]
    NoConfigDir,
    #[error("Failed to read config file: {0}")]
    ReadError(#[from] std::io::Error),
    #[error("Failed to parse config: {0}")]
    ParseError(#[from] toml::de::Error),
    #[error("Failed to serialize config: {0}")]
    SerializeError(#[from] toml::ser::Error),
}

/// Desktop app configuration stored in platform-specific location.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DesktopConfig {
    /// Whether to start the embedded server (set to false to use remote server)
    pub embedded_server: bool,

    /// Server port (embedded server binds to this)
    pub server_port: u16,

    /// Remote server URL (used when embedded_server is false)
    pub remote_server_url: Option<String>,

    /// Directory containing voice safetensors files (for realtime model)
    pub voices_dir: Option<PathBuf>,

    /// Directory containing WAV samples for voice cloning (for batch models)
    pub samples_dir: Option<PathBuf>,

    /// Optional directory to save output WAV files
    pub output_dir: Option<PathBuf>,

    /// Default model variant: "realtime", "1.5B", or "7B"
    pub default_model: String,

    /// Global hotkey for showing window (e.g., "CommandOrControl+Shift+V")
    pub hotkey_show: Option<String>,

    /// Start minimized to tray
    pub start_minimized: bool,

    /// Show notifications on synthesis complete
    pub show_notifications: bool,
}

impl Default for DesktopConfig {
    fn default() -> Self {
        Self {
            embedded_server: true,
            server_port: 3908,
            remote_server_url: None,
            voices_dir: None,
            samples_dir: None,
            output_dir: None,
            default_model: "realtime".to_string(),
            hotkey_show: Some("CommandOrControl+Shift+V".to_string()),
            start_minimized: false,
            show_notifications: true,
        }
    }
}

impl DesktopConfig {
    /// Get the platform-specific config directory.
    /// - macOS: ~/Library/Application Support/com.vibevoice.VibeVoice/
    /// - Linux: ~/.config/vibevoice/
    /// - Windows: %APPDATA%\vibevoice\VibeVoice\
    pub fn config_dir() -> Result<PathBuf, ConfigError> {
        ProjectDirs::from("com", "vibevoice", "VibeVoice")
            .map(|dirs| dirs.config_dir().to_path_buf())
            .ok_or(ConfigError::NoConfigDir)
    }

    /// Get the full path to the config file.
    pub fn config_path() -> Result<PathBuf, ConfigError> {
        Ok(Self::config_dir()?.join("config.toml"))
    }

    /// Load config from file, or return defaults if file doesn't exist.
    pub fn load() -> Result<Self, ConfigError> {
        let path = Self::config_path()?;

        if !path.exists() {
            return Ok(Self::default());
        }

        let contents = std::fs::read_to_string(&path)?;
        let config: Self = toml::from_str(&contents)?;
        Ok(config)
    }

    /// Convert to vibevoice-server Config format.
    pub fn to_server_config(&self) -> vibevoice_server::Config {
        vibevoice_server::Config {
            host: Some("127.0.0.1".to_string()),
            port: Some(self.server_port),
            voices_dir: self.voices_dir.clone(),
            samples_dir: self.samples_dir.clone(),
            output_dir: self.output_dir.clone(),
            web_dir: None, // Tauri serves the frontend
            cors_origins: vec![], // Not needed for embedded server
        }
    }
}
