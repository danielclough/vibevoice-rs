//! Text templates storage (stub for Phase 2).

use gloo_storage::{LocalStorage, Storage};
use serde::{Deserialize, Serialize};

/// A saved text template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextTemplate {
    pub id: String,
    pub name: String,
    pub text: String,
    pub created_at: u64,
}

/// Load templates from localStorage.
pub fn load() -> Vec<TextTemplate> {
    LocalStorage::get(super::STORAGE_TEMPLATES).unwrap_or_default()
}

/// Save templates to localStorage.
pub fn save(templates: &[TextTemplate]) {
    let _ = LocalStorage::set(super::STORAGE_TEMPLATES, templates);
}
