//! Audio history component for displaying and managing saved audio clips.

use leptos::prelude::*;
use wasm_bindgen::JsCast;
use crate::storage::history::HistoryEntry;

#[component]
pub fn AudioHistory(
    history: Signal<Vec<HistoryEntry>>,
    current_server_url: Signal<String>,
    #[prop(into)] on_play: Callback<HistoryEntry>,
    #[prop(into)] on_delete: Callback<String>,
    #[prop(into)] on_reuse: Callback<HistoryEntry>,
) -> impl IntoView {
    // Filter history to show only entries from the current server
    let filtered_history = Memo::new(move |_| {
        let url = current_server_url.get();
        history.get().into_iter()
            .filter(|e| e.server_url == url)
            .collect::<Vec<_>>()
    });

    view! {
        <div class="history-section">
            <h3 class="section-title">"Audio History"</h3>
            <HistoryList
                entries=filtered_history.into()
                on_play=on_play
                on_delete=on_delete
                on_reuse=on_reuse
            />
        </div>
    }
}

#[component]
fn HistoryList(
    entries: Signal<Vec<HistoryEntry>>,
    #[prop(into)] on_play: Callback<HistoryEntry>,
    #[prop(into)] on_delete: Callback<String>,
    #[prop(into)] on_reuse: Callback<HistoryEntry>,
) -> impl IntoView {
    view! {
        <div class="history-list">
            {move || {
                let entries_vec = entries.get();
                if entries_vec.is_empty() {
                    view! { <p class="history-empty">"No history yet"</p> }.into_any()
                } else {
                    entries_vec.into_iter().map(|entry| {
                        view! {
                            <HistoryItem
                                entry=entry
                                on_play=on_play.clone()
                                on_delete=on_delete.clone()
                                on_reuse=on_reuse.clone()
                            />
                        }
                    }).collect_view().into_any()
                }
            }}
        </div>
    }
}

#[component]
fn HistoryItem(
    entry: HistoryEntry,
    #[prop(into)] on_play: Callback<HistoryEntry>,
    #[prop(into)] on_delete: Callback<String>,
    #[prop(into)] on_reuse: Callback<HistoryEntry>,
) -> impl IntoView {
    let entry_play = entry.clone();
    let entry_reuse = entry.clone();
    let entry_download = entry.clone();
    let id_delete = entry.id.clone();

    let display_text: String = entry.text.chars().take(50).collect();
    let display_text = if entry.text.len() > 50 {
        format!("{}...", display_text)
    } else {
        display_text
    };
    let full_text = entry.text.clone();

    let download_audio = move |_| {
        let entry = entry_download.clone();
        if let Ok(bytes) = base64::Engine::decode(
            &base64::engine::general_purpose::STANDARD,
            &entry.audio_b64,
        ) {
            // Create blob and download
            let array = js_sys::Uint8Array::from(&bytes[..]);
            let blob_parts = js_sys::Array::new();
            blob_parts.push(&array);

            if let Ok(blob) = web_sys::Blob::new_with_u8_array_sequence_and_options(
                &blob_parts,
                web_sys::BlobPropertyBag::new().type_("audio/wav"),
            ) {
                if let Ok(url) = web_sys::Url::create_object_url_with_blob(&blob) {
                    let window = web_sys::window().unwrap();
                    let document = window.document().unwrap();
                    let a: web_sys::HtmlAnchorElement = document
                        .create_element("a")
                        .unwrap()
                        .unchecked_into();
                    a.set_href(&url);
                    // Create filename from first few words of text
                    let filename: String = entry.text
                        .split_whitespace()
                        .take(3)
                        .collect::<Vec<_>>()
                        .join("_");
                    let filename = format!("{}.wav", if filename.is_empty() { "audio" } else { &filename });
                    a.set_download(&filename);
                    a.click();
                    let _ = web_sys::Url::revoke_object_url(&url);
                }
            }
        }
    };

    view! {
        <div class="history-entry">
            <div class="history-entry-text" title=full_text>
                {display_text}
            </div>
            <div class="history-entry-meta">
                <span class="history-voice">{entry.voice.clone()}</span>
                " · "
                <span class="history-model">{entry.model.clone()}</span>
            </div>
            <div class="history-entry-actions">
                <div>
                    <button
                        class="history-btn play-btn"
                        on:click=move |_| on_play.run(entry_play.clone())
                        title="Play audio"
                    >
                        "Play"
                    </button>
                    <button
                        class="history-btn download-btn"
                        on:click=download_audio
                        title="Download audio"
                    >
                        "Download"
                    </button>
                </div>
                <div>
                    <button
                        class="history-btn reuse-btn"
                        on:click=move |_| on_reuse.run(entry_reuse.clone())
                        title="Load text and settings"
                    >
                        "Reuse"
                    </button>
                    <button
                        class="history-btn delete-btn"
                        on:click=move |_| on_delete.run(id_delete.clone())
                        title="Delete from history"
                    >
                        "Delete"
                    </button>
                </div>
            </div>
        </div>
    }
}
