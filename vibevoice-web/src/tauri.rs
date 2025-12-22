//! Tauri integration for desktop app.
//!
//! When running inside Tauri, this module provides access to Tauri commands.
//! When running in a browser, these functions return None.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["window", "__TAURI__", "core"], js_name = invoke, catch)]
    async fn tauri_invoke(cmd: &str) -> Result<JsValue, JsValue>;

    #[wasm_bindgen(js_namespace = ["window", "__TAURI__", "core"], js_name = invoke, catch)]
    async fn tauri_invoke_with_args(cmd: &str, args: JsValue) -> Result<JsValue, JsValue>;
}

/// Desktop-specific settings nested in the config.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DesktopSettings {
    #[serde(default = "default_true")]
    pub embedded_server: bool,
    pub remote_server_url: Option<String>,
    #[serde(default = "default_model")]
    pub default_model: String,
    pub hotkey_show: Option<String>,
    #[serde(default)]
    pub start_minimized: bool,
    #[serde(default = "default_true")]
    pub show_notifications: bool,
}

fn default_true() -> bool { true }
fn default_model() -> String { "realtime".to_string() }

/// Unified configuration from the Tauri backend.
/// Matches vibevoice_server::Config structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TauriConfig {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub safetensors_dir: Option<String>,
    pub wav_dir: Option<String>,
    pub output_dir: Option<String>,
    pub web_dir: Option<String>,
    #[serde(default)]
    pub cors_origins: Vec<String>,
    pub desktop: Option<DesktopSettings>,
}

impl Default for TauriConfig {
    fn default() -> Self {
        Self {
            host: None,
            port: Some(3908),
            safetensors_dir: None,
            wav_dir: None,
            output_dir: None,
            web_dir: None,
            cors_origins: vec![],
            desktop: Some(DesktopSettings::default()),
        }
    }
}

impl TauriConfig {
    /// Get desktop settings with defaults.
    pub fn desktop_settings(&self) -> DesktopSettings {
        self.desktop.clone().unwrap_or_default()
    }

    /// Get server port with default.
    pub fn server_port(&self) -> u16 {
        self.port.unwrap_or(3908)
    }
}

/// Check if we're running inside Tauri.
pub fn is_tauri() -> bool {
    let result = web_sys::window()
        .and_then(|w| js_sys::Reflect::get(&w, &"__TAURI__".into()).ok())
        .map(|v| !v.is_undefined())
        .unwrap_or(false);
    web_sys::console::log_1(&format!("[tauri] is_tauri: {}", result).into());
    result
}

/// Get the server URL from Tauri config.
/// Returns None if not running in Tauri or if the call fails.
pub async fn get_server_url() -> Option<String> {
    if !is_tauri() {
        return None;
    }

    match tauri_invoke("get_server_url").await {
        Ok(value) => value.as_string(),
        Err(_) => None,
    }
}

/// Get the current configuration from Tauri.
pub async fn get_config() -> Option<TauriConfig> {
    web_sys::console::log_1(&"[tauri] get_config called".into());

    if !is_tauri() {
        web_sys::console::log_1(&"[tauri] get_config: not in tauri".into());
        return None;
    }

    match tauri_invoke("get_config").await {
        Ok(value) => {
            web_sys::console::log_1(&format!("[tauri] get_config raw value: {:?}", value).into());
            match serde_wasm_bindgen::from_value(value) {
                Ok(config) => {
                    web_sys::console::log_1(&format!("[tauri] get_config parsed: {:?}", config).into());
                    Some(config)
                }
                Err(e) => {
                    web_sys::console::log_1(&format!("[tauri] get_config deserialize error: {:?}", e).into());
                    None
                }
            }
        }
        Err(e) => {
            web_sys::console::log_1(&format!("[tauri] get_config invoke error: {:?}", e).into());
            None
        }
    }
}

/// Arguments for save_config - flattened for compatibility with backend.
#[derive(Serialize)]
struct SaveConfigArgs {
    embedded_server: bool,
    server_port: u16,
    remote_server_url: Option<String>,
    safetensors_dir: Option<String>,
    wav_dir: Option<String>,
    output_dir: Option<String>,
    default_model: String,
}

/// Wrapper for save_config args (matches backend parameter name).
#[derive(Serialize)]
struct SaveConfigArgsWrapper {
    args: SaveConfigArgs,
}

/// Save configuration to Tauri backend.
pub async fn save_config(config: &TauriConfig) -> Result<(), String> {
    if !is_tauri() {
        return Err("Not running in Tauri".to_string());
    }

    let desktop = config.desktop_settings();
    let args = SaveConfigArgs {
        embedded_server: desktop.embedded_server,
        server_port: config.server_port(),
        remote_server_url: desktop.remote_server_url,
        safetensors_dir: config.safetensors_dir.clone(),
        wav_dir: config.wav_dir.clone(),
        output_dir: config.output_dir.clone(),
        default_model: desktop.default_model,
    };

    let wrapper = SaveConfigArgsWrapper { args };
    let args_js = serde_wasm_bindgen::to_value(&wrapper)
        .map_err(|e| format!("Failed to serialize config: {}", e))?;

    match tauri_invoke_with_args("save_config", args_js).await {
        Ok(_) => Ok(()),
        Err(e) => Err(e.as_string().unwrap_or_else(|| "Unknown error".to_string())),
    }
}

/// Open native directory picker dialog.
pub async fn pick_directory() -> Option<String> {
    if !is_tauri() {
        return None;
    }

    match tauri_invoke("pick_directory").await {
        Ok(value) => value.as_string(),
        Err(_) => None,
    }
}

/// Start the embedded server (Tauri only).
/// Returns the server URL if successful, or an error message.
pub async fn start_embedded_server() -> Result<String, String> {
    web_sys::console::log_1(&"[tauri] start_embedded_server called".into());

    if !is_tauri() {
        web_sys::console::log_1(&"[tauri] start_embedded_server: not in tauri".into());
        return Err("Not running in Tauri".to_string());
    }

    web_sys::console::log_1(&"[tauri] start_embedded_server: invoking command".into());
    match tauri_invoke("start_embedded_server").await {
        Ok(value) => {
            web_sys::console::log_1(&format!("[tauri] start_embedded_server success: {:?}", value).into());
            value
                .as_string()
                .ok_or_else(|| "Invalid response from server".to_string())
        }
        Err(e) => {
            let err_msg = e.as_string().unwrap_or_else(|| "Unknown error".to_string());
            web_sys::console::log_1(&format!("[tauri] start_embedded_server error: {}", err_msg).into());
            Err(err_msg)
        }
    }
}

/// Stop the embedded server (Tauri only).
/// Returns true if successful.
pub async fn stop_embedded_server() -> bool {
    if !is_tauri() {
        return false;
    }

    tauri_invoke("stop_embedded_server").await.is_ok()
}

/// Get the embedded server status (Tauri only).
/// Returns "running", "stopped", or "failed".
pub async fn get_embedded_server_status() -> Option<String> {
    if !is_tauri() {
        return None;
    }

    match tauri_invoke("get_embedded_server_status").await {
        Ok(value) => value.as_string(),
        Err(_) => None,
    }
}
