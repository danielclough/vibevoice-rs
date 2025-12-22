//! Audio history storage and management.

use gloo_storage::{LocalStorage, Storage};
use serde::{Deserialize, Serialize};

const MAX_ENTRIES: usize = 50;

/// A single audio history entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub id: String,
    pub created_at: u64,
    pub text: String,
    pub voice: String,
    pub model: String,
    pub audio_b64: String,   // Base64-encoded WAV
    pub server_url: String,  // Server that produced this audio
}

/// Load history entries from localStorage.
pub fn load() -> Vec<HistoryEntry> {
    LocalStorage::get(super::STORAGE_AUDIO_HISTORY).unwrap_or_default()
}

/// Save history entries to localStorage.
pub fn save(entries: &[HistoryEntry]) {
    let _ = LocalStorage::set(super::STORAGE_AUDIO_HISTORY, entries);
}

/// Add an entry to the history (at the beginning), enforcing max entries.
pub fn add(entries: &mut Vec<HistoryEntry>, entry: HistoryEntry) {
    entries.insert(0, entry);
    entries.truncate(MAX_ENTRIES);
}

/// Remove an entry by ID.
pub fn remove(entries: &mut Vec<HistoryEntry>, id: &str) {
    entries.retain(|e| e.id != id);
}
