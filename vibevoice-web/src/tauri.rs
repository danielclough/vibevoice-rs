//! Tauri integration for desktop app.
//!
//! When running inside Tauri, this module provides access to Tauri commands.
//! When running in a browser, these functions return None.

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["window", "__TAURI__", "core"], js_name = invoke, catch)]
    async fn tauri_invoke(cmd: &str) -> Result<JsValue, JsValue>;
}

/// Check if we're running inside Tauri.
pub fn is_tauri() -> bool {
    web_sys::window()
        .and_then(|w| js_sys::Reflect::get(&w, &"__TAURI__".into()).ok())
        .map(|v| !v.is_undefined())
        .unwrap_or(false)
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