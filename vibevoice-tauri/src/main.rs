//! VibeVoice Desktop Application
//!
//! A Tauri v2 application that embeds the vibevoice-server for offline
//! text-to-speech synthesis with voice cloning capabilities.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod config;
mod hotkeys;
mod server;
mod tray;

use config::DesktopConfig;
use server::EmbeddedServer;
use std::sync::Mutex;
use tauri::{Manager, RunEvent};
use tracing::{error, info, Level};
use tracing_subscriber::FmtSubscriber;
use vibevoice::ModelVariant;

/// Application state managed by Tauri.
struct AppState {
    server: Mutex<Option<EmbeddedServer>>,
    config: DesktopConfig,
}

/// Parse model variant from string.
fn parse_model_variant(s: &str) -> ModelVariant {
    match s.to_lowercase().as_str() {
        "1.5b" | "batch1_5b" => ModelVariant::Batch1_5B,
        "7b" | "batch7b" => ModelVariant::Batch7B,
        "realtime" | _ => ModelVariant::Realtime,
    }
}

/// Tauri command: Get the server URL (embedded or remote).
#[tauri::command]
fn get_server_url(state: tauri::State<AppState>) -> String {
    // If using remote server, return that URL
    if !state.config.embedded_server {
        return state
            .config
            .remote_server_url
            .clone()
            .unwrap_or_else(|| "http://localhost:3908".to_string());
    }

    // Otherwise return embedded server URL
    state
        .server
        .lock()
        .ok()
        .and_then(|guard| guard.as_ref().map(|s| s.url()))
        .unwrap_or_else(|| format!("http://127.0.0.1:{}", state.config.server_port))
}

/// Tauri command: Get current configuration.
#[tauri::command]
fn get_config(state: tauri::State<AppState>) -> DesktopConfig {
    state.config.clone()
}

/// Tauri command: Start the embedded server.
/// Returns the server URL on success, or an error message.
#[tauri::command]
fn start_embedded_server(state: tauri::State<AppState>) -> Result<String, String> {
    let mut guard = state.server.lock().map_err(|e| e.to_string())?;

    // Check if already running
    if guard.is_some() {
        return Ok(guard.as_ref().unwrap().url());
    }

    // Start the embedded server
    let model_variant = parse_model_variant(&state.config.default_model);
    info!("Starting embedded server with model {:?}...", model_variant);

    let server_config = state.config.to_server_config();
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

    // Load configuration
    let config = match DesktopConfig::load() {
        Ok(c) => {
            info!("Loaded configuration from {:?}", DesktopConfig::config_path());
            c
        }
        Err(e) => {
            error!("Failed to load config: {}, using defaults", e);
            DesktopConfig::default()
        }
    };

    // Server is started on-demand via start_embedded_server command
    // No auto-start - user chooses via setup wizard
    let embedded_server: Option<EmbeddedServer> = None;
    info!("Embedded server will start on-demand via setup wizard");

    // Store hotkey config for setup
    let hotkey_config = config.hotkey_show.clone();
    let start_minimized = config.start_minimized;

    // Create app state
    let app_state = AppState {
        server: Mutex::new(embedded_server),
        config,
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
