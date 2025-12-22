//! Platform-specific configuration paths and settings.
//!
//! Uses the unified Config from vibevoice-server, stored as YAML.

use directories::ProjectDirs;
use std::path::PathBuf;
use thiserror::Error;

// Re-export from vibevoice_server for convenience
pub use vibevoice_server::{Config, DesktopSettings};

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Failed to determine config directory")]
    NoConfigDir,
    #[error("Failed to read config file: {0}")]
    ReadError(#[from] std::io::Error),
    #[error("Config error: {0}")]
    ConfigError(#[from] anyhow::Error),
}

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
    Ok(config_dir()?.join("config.yaml"))
}

/// Load config from file, or return defaults if file doesn't exist.
pub fn load_config() -> Result<Config, ConfigError> {
    let path = config_path()?;

    if !path.exists() {
        return Ok(Config::default());
    }

    Config::from_file(&path).map_err(ConfigError::from)
}

/// Save config to file.
pub fn save_config(config: &Config) -> Result<(), ConfigError> {
    let path = config_path()?;

    // Create config directory if it doesn't exist
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    config.to_file(&path).map_err(ConfigError::from)
}

/// Migrate from old TOML config to new YAML config if needed.
pub fn migrate_config_if_needed() -> Result<(), ConfigError> {
    let yaml_path = config_path()?;
    let toml_path = config_dir()?.join("config.toml");

    // If YAML exists or TOML doesn't exist, no migration needed
    if yaml_path.exists() || !toml_path.exists() {
        return Ok(());
    }

    // Try to migrate from TOML
    if let Ok(contents) = std::fs::read_to_string(&toml_path) {
        // Parse old TOML config
        #[derive(serde::Deserialize)]
        #[serde(default)]
        struct OldConfig {
            embedded_server: bool,
            server_port: u16,
            remote_server_url: Option<String>,
            voices_dir: Option<PathBuf>,
            samples_dir: Option<PathBuf>,
            output_dir: Option<PathBuf>,
            default_model: String,
            hotkey_show: Option<String>,
            start_minimized: bool,
            show_notifications: bool,
        }

        impl Default for OldConfig {
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

        if let Ok(old) = toml::from_str::<OldConfig>(&contents) {
            // Convert to new unified config
            // Note: old voices_dir/samples_dir map to new safetensors_dir/wav_dir
            let new_config = Config {
                host: Some("127.0.0.1".to_string()),
                port: Some(old.server_port),
                safetensors_dir: old.voices_dir,
                wav_dir: old.samples_dir,
                output_dir: old.output_dir,
                web_dir: None,
                cors_origins: vec![],
                desktop: Some(DesktopSettings {
                    embedded_server: old.embedded_server,
                    remote_server_url: old.remote_server_url,
                    default_model: old.default_model,
                    hotkey_show: old.hotkey_show,
                    start_minimized: old.start_minimized,
                    show_notifications: old.show_notifications,
                }),
            };

            // Save new config
            save_config(&new_config)?;

            // Optionally rename old config as backup
            let _ = std::fs::rename(&toml_path, toml_path.with_extension("toml.bak"));

            tracing::info!("Migrated config from TOML to YAML");
        }
    }

    Ok(())
}
