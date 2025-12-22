//! VibeVoice Desktop Application
//!
//! A Tauri v2 application that embeds the vibevoice-server for offline
//! text-to-speech synthesis with voice cloning capabilities.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod config;
mod hotkeys;
mod server;
mod tray;

use config::{load_config, migrate_config_if_needed, save_config as save_config_to_file, Config, DesktopSettings, config_path};
use serde::Deserialize;
use server::EmbeddedServer;
use std::sync::{Mutex, RwLock};
use tauri::{Manager, RunEvent};
use tracing::{error, info, Level};
use tracing_subscriber::FmtSubscriber;
use vibevoice::ModelVariant;

/// Application state managed by Tauri.
struct AppState {
    server: Mutex<Option<EmbeddedServer>>,
    config: RwLock<Config>,
}

/// Parse model variant from string.
fn parse_model_variant(s: &str) -> ModelVariant {
    match s.to_lowercase().as_str() {
        "1.5b" | "batch1_5b" => ModelVariant::Batch1_5B,
        "7b" | "batch7b" => ModelVariant::Batch7B,
        "realtime" | _ => ModelVariant::Realtime,
    }
}

/// Helper to get desktop settings with defaults.
fn get_desktop_settings(config: &Config) -> DesktopSettings {
    config.desktop.clone().unwrap_or_default()
}

/// Tauri command: Get the server URL (embedded or remote).
#[tauri::command]
fn get_server_url(state: tauri::State<AppState>) -> Result<String, String> {
    let config = state.config.read().map_err(|e| e.to_string())?;
    let desktop = get_desktop_settings(&config);

    // If using remote server, return that URL
    if !desktop.embedded_server {
        return Ok(desktop
            .remote_server_url
            .unwrap_or_else(|| "http://localhost:3908".to_string()));
    }

    // Otherwise return embedded server URL
    let port = config.port.unwrap_or(3908);
    Ok(state
        .server
        .lock()
        .ok()
        .and_then(|guard| guard.as_ref().map(|s| s.url()))
        .unwrap_or_else(|| format!("http://127.0.0.1:{}", port)))
}

/// Tauri command: Get current configuration.
#[tauri::command]
fn get_config(state: tauri::State<AppState>) -> Result<Config, String> {
    let config = state.config.read().map_err(|e| e.to_string())?;
    Ok(config.clone())
}

/// Arguments for save_config command (matches frontend TauriConfig struct).
#[derive(Deserialize)]
struct SaveConfigArgs {
    embedded_server: bool,
    server_port: u16,
    remote_server_url: Option<String>,
    safetensors_dir: Option<String>,
    wav_dir: Option<String>,
    output_dir: Option<String>,
    default_model: String,
}

/// Tauri command: Save configuration changes.
#[tauri::command]
fn save_config(state: tauri::State<AppState>, args: SaveConfigArgs) -> Result<(), String> {
    let mut config = state.config.write().map_err(|e| e.to_string())?;

    // Update server settings
    config.port = Some(args.server_port);
    config.safetensors_dir = args.safetensors_dir.map(std::path::PathBuf::from);
    config.wav_dir = args.wav_dir.map(std::path::PathBuf::from);
    config.output_dir = args.output_dir.map(std::path::PathBuf::from);

    // Update desktop settings
    let desktop = config.desktop.get_or_insert_with(DesktopSettings::default);
    desktop.embedded_server = args.embedded_server;
    desktop.remote_server_url = args.remote_server_url;
    desktop.default_model = args.default_model;

    // Save to file
    save_config_to_file(&config).map_err(|e| e.to_string())?;

    info!(
        "Configuration saved: safetensors_dir={:?}, wav_dir={:?}",
        config.safetensors_dir, config.wav_dir
    );
    Ok(())
}

/// Tauri command: Open directory picker dialog.
#[tauri::command]
async fn pick_directory(app: tauri::AppHandle) -> Option<String> {
    use tauri_plugin_dialog::{DialogExt, FilePath};

    let (tx, rx) = std::sync::mpsc::channel();
    app.dialog().file().pick_folder(move |path| {
        let result = path.and_then(|p| match p {
            FilePath::Path(path_buf) => Some(path_buf.to_string_lossy().to_string()),
            _ => None,
        });
        let _ = tx.send(result);
    });

    rx.recv().ok().flatten()
}

/// Tauri command: Start the embedded server.
/// Returns the server URL on success, or an error message.
/// If the server is already running, it will be restarted to apply new config.
#[tauri::command]
fn start_embedded_server(state: tauri::State<AppState>) -> Result<String, String> {
    let mut guard = state.server.lock().map_err(|e| e.to_string())?;

    // Stop existing server if running (to pick up new config)
    if let Some(mut server) = guard.take() {
        info!("Stopping existing server to apply new configuration...");
        server.shutdown();
    }

    // Get current config
    let config = state.config.read().map_err(|e| e.to_string())?;
    let desktop = get_desktop_settings(&config);

    // Start the embedded server with current config
    let model_variant = parse_model_variant(&desktop.default_model);
    info!("Starting embedded server with model {:?}...", model_variant);
    info!("Config: safetensors_dir={:?}, wav_dir={:?}", config.safetensors_dir, config.wav_dir);

    // Clone config for server (it will be wrapped in Arc)
    let server_config = config.clone();
    drop(config); // Release read lock before potentially long operation

    match EmbeddedServer::start(server_config, model_variant) {
        Ok(server) => {
            let url = server.url();
            info!("Embedded server started at {}", url);
            *guard = Some(server);
            Ok(url)
        }
        Err(e) => {
            error!("Failed to start embedded server: {}", e);
            Err(format!("Failed to start server: {}", e))
        }
    }
}

/// Tauri command: Stop the embedded server.
#[tauri::command]
fn stop_embedded_server(state: tauri::State<AppState>) -> Result<(), String> {
    let mut guard = state.server.lock().map_err(|e| e.to_string())?;

    if let Some(mut server) = guard.take() {
        server.shutdown();
        info!("Embedded server stopped");
    }

    Ok(())
}

/// Tauri command: Get the embedded server status.
/// Returns "running", "stopped", or "failed".
#[tauri::command]
fn get_embedded_server_status(state: tauri::State<AppState>) -> String {
    match state.server.lock() {
        Ok(guard) => {
            if guard.is_some() {
                "running".to_string()
            } else {
                "stopped".to_string()
            }
        }
        Err(_) => "failed".to_string(),
    }
}

fn main() {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set tracing subscriber");

    info!("Starting VibeVoice Desktop Application");

    // Migrate config from TOML to YAML if needed
    if let Err(e) = migrate_config_if_needed() {
        error!("Failed to migrate config: {}", e);
    }

    // Load configuration
    let config = match load_config() {
        Ok(c) => {
            info!("Loaded configuration from {:?}", config_path());
            c
        }
        Err(e) => {
            error!("Failed to load config: {}, using defaults", e);
            Config::default()
        }
    };

    // Server is started on-demand via start_embedded_server command
    // No auto-start - user chooses via setup wizard
    let embedded_server: Option<EmbeddedServer> = None;
    info!("Embedded server will start on-demand via setup wizard");

    // Get desktop settings for setup
    let desktop = get_desktop_settings(&config);
    let hotkey_config = desktop.hotkey_show.clone();
    let start_minimized = desktop.start_minimized;

    // Create app state
    let app_state = AppState {
        server: Mutex::new(embedded_server),
        config: RwLock::new(config),
    };

    // Build and run Tauri application
    tauri::Builder::default()
        .manage(app_state)
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .setup(move |app| {
            let handle = app.handle().clone();

            // Setup system tray
            tray::setup_tray(&handle)?;

            // Setup global hotkeys
            if let Err(e) = hotkeys::setup_hotkeys(&handle, hotkey_config.as_deref()) {
                error!("Failed to setup hotkeys: {}", e);
            }

            // Show window unless start_minimized is set
            if !start_minimized {
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.show();
                }
            }

            info!("Tauri application setup complete");
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_server_url,
            get_config,
            save_config,
            pick_directory,
            start_embedded_server,
            stop_embedded_server,
            get_embedded_server_status,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| match event {
            RunEvent::ExitRequested { .. } => {
                info!("Exit requested, performing cleanup...");

                // Shutdown embedded server
                if let Some(state) = app_handle.try_state::<AppState>() {
                    if let Ok(mut guard) = state.server.lock() {
                        if let Some(server) = guard.as_mut() {
                            server.shutdown();
                        }
                    }
                }
            }
            RunEvent::Exit => {
                info!("Application exiting");
            }
            _ => {}
        });
}
