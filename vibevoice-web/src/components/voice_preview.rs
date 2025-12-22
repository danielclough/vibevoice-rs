//! Voice preview component for playing audio samples of selected voices.

use leptos::prelude::*;
use leptos::task::spawn_local;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use wasm_bindgen::JsCast;
use wasm_bindgen::closure::Closure;
use base64::Engine;

use crate::api::client::synthesize_json;
use crate::components::model_selector::Model;

const PREVIEW_TEXT: &str = "Hello, this is a voice preview.";

/// Play audio bytes and return the audio element.
fn play_audio(
    bytes: Vec<u8>,
    current_audio: Rc<RefCell<Option<web_sys::HtmlAudioElement>>>,
    is_playing: RwSignal<bool>,
) {
    // Stop any current playback first
    if let Some(audio) = current_audio.borrow_mut().take() {
        audio.pause().ok();
    }

    // Create blob and audio element
    let array = js_sys::Uint8Array::new_with_length(bytes.len() as u32);
    array.copy_from(&bytes);

    let blob_parts = js_sys::Array::new();
    blob_parts.push(&array.buffer());

    let mut options = web_sys::BlobPropertyBag::new();
    options.type_("audio/wav");

    if let Ok(blob) = web_sys::Blob::new_with_u8_array_sequence_and_options(&blob_parts, &options) {
        if let Ok(url) = web_sys::Url::create_object_url_with_blob(&blob) {
            if let Ok(audio) = web_sys::HtmlAudioElement::new_with_src(&url) {
                // Set up ended handler
                let current_audio_ended = current_audio.clone();
                let url_clone = url.clone();
                let onended = Closure::wrap(Box::new(move || {
                    is_playing.set(false);
                    current_audio_ended.borrow_mut().take();
                    let _ = web_sys::Url::revoke_object_url(&url_clone);
                }) as Box<dyn Fn()>);

                audio.set_onended(Some(onended.as_ref().unchecked_ref()));
                onended.forget();

                audio.play().ok();
                *current_audio.borrow_mut() = Some(audio);
                is_playing.set(true);
            }
        }
    }
}

#[component]
pub fn VoicePreview(
    server_url: Signal<String>,
    voice: Signal<String>,
    model: Signal<Model>,
) -> impl IntoView {
    // Cache keyed by "server_url:voice:model"
    let cache = RwSignal::new(HashMap::<String, Vec<u8>>::new());
    let is_loading = RwSignal::new(false);
    let is_playing = RwSignal::new(false);
    // Use Rc<RefCell> for audio element since HtmlAudioElement is not Send
    let current_audio: Rc<RefCell<Option<web_sys::HtmlAudioElement>>> = Rc::new(RefCell::new(None));

    // Clear cache when server changes (different server = different voices)
    Effect::new(move |prev_url: Option<String>| {
        let url = server_url.get();
        if let Some(prev) = prev_url {
            if prev != url {
                cache.set(HashMap::new());
            }
        }
        url
    });

    let current_audio_clone = current_audio.clone();
    let handle_click = move |_| {
        // If playing, stop
        if is_playing.get_untracked() {
            if let Some(audio) = current_audio_clone.borrow_mut().take() {
                audio.pause().ok();
            }
            is_playing.set(false);
            return;
        }

        let url = server_url.get_untracked();
        let v = voice.get_untracked();
        let m = model.get_untracked();

        if v.is_empty() {
            return;
        }

        let cache_key = format!("{}:{}:{}", url, v, m.as_str());

        // Check cache first
        if let Some(audio_bytes) = cache.get_untracked().get(&cache_key).cloned() {
            play_audio(audio_bytes, current_audio_clone.clone(), is_playing);
            return;
        }

        // Synthesize preview
        is_loading.set(true);
        let current_audio_for_spawn = current_audio_clone.clone();
        spawn_local(async move {
            match synthesize_json(&url, PREVIEW_TEXT, &v, Some(m.as_str())).await {
                Ok(response) => {
                    if let Ok(bytes) = base64::engine::general_purpose::STANDARD
                        .decode(&response.audio_base64)
                    {
                        // Cache the result
                        cache.update(|c| {
                            c.insert(cache_key.clone(), bytes.clone());
                        });
                        play_audio(bytes, current_audio_for_spawn, is_playing);
                    }
                }
                Err(_) => {
                    // Silent fail for preview - user can try again
                }
            }
            is_loading.set(false);
        });
    };

    let button_content = move || {
        if is_loading.get() {
            "..."
        } else if is_playing.get() {
            "Stop"
        } else {
            "Preview"
        }
    };

    let button_title = move || {
        if is_loading.get() {
            "Loading preview..."
        } else if is_playing.get() {
            "Stop preview"
        } else {
            "Preview voice"
        }
    };

    view! {
        <button
            class="preview-button"
            class:loading=move || is_loading.get()
            class:playing=move || is_playing.get()
            on:click=handle_click
            disabled=move || voice.get().is_empty() || is_loading.get()
            title=button_title
        >
            {button_content}
        </button>
    }
}
