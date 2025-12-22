//! Centralized storage module for localStorage operations.

pub mod history;
pub mod templates;

// Storage keys
pub const STORAGE_SERVER_URL: &str = "vibevoice.server_url";
pub const STORAGE_LAST_SERVER: &str = "vibevoice.last_server";
pub const STORAGE_MODEL: &str = "vibevoice.model";
pub const STORAGE_STREAMING: &str = "vibevoice.use_streaming";
pub const STORAGE_AUDIO_HISTORY: &str = "vibevoice.audio_history";
pub const STORAGE_TEMPLATES: &str = "vibevoice.templates";

/// Generate a unique ID using crypto.getRandomValues (no uuid crate needed)
pub fn generate_id() -> String {
    let array = js_sys::Uint8Array::new_with_length(16);
    if let Some(window) = web_sys::window() {
        if let Ok(crypto) = window.crypto() {
            let _ = crypto.get_random_values_with_array_buffer_view(&array);
        }
    }
    let bytes: Vec<u8> = array.to_vec();
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Get current timestamp in milliseconds using js_sys::Date
pub fn now_millis() -> u64 {
    js_sys::Date::now() as u64
}
