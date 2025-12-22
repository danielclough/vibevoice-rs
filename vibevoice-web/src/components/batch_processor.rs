//! Batch processing component for synthesizing multiple text lines.

use leptos::prelude::*;
use leptos::task::spawn_local;
use wasm_bindgen::JsCast;
use base64::Engine;

use crate::api::client::synthesize_json;
use crate::components::model_selector::Model;

/// Status of a batch item.
#[derive(Debug, Clone, PartialEq)]
pub enum BatchItemStatus {
    Pending,
    Processing,
    Completed,
    Failed(String),
}

/// A single batch item.
#[derive(Debug, Clone)]
pub struct BatchItem {
    pub id: usize,
    pub text: String,
    pub status: BatchItemStatus,
    pub audio_data: Option<Vec<u8>>,
}

#[component]
pub fn BatchProcessor(
    server_url: Signal<String>,
    voice: Signal<String>,
    model: Signal<Model>,
) -> impl IntoView {
    let input_text = RwSignal::new(String::new());
    let items = RwSignal::new(Vec::<BatchItem>::new());
    let is_processing = RwSignal::new(false);
    let cancel_flag = RwSignal::new(false);

    // Count non-empty lines
    let line_count = Memo::new(move |_| {
        input_text.get().lines().filter(|l| !l.trim().is_empty()).count()
    });

    // Count completed items
    let completed_count = Memo::new(move |_| {
        items.get().iter().filter(|i| matches!(i.status, BatchItemStatus::Completed)).count()
    });

    let parse_items = move |_| {
        let text = input_text.get_untracked();
        let parsed: Vec<BatchItem> = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .enumerate()
            .map(|(i, line)| BatchItem {
                id: i,
                text: line.trim().to_string(),
                status: BatchItemStatus::Pending,
                audio_data: None,
            })
            .collect();
        items.set(parsed);
        cancel_flag.set(false);
    };

    let process_all = move |_| {
        let url = server_url.get_untracked();
        let v = voice.get_untracked();
        let m = model.get_untracked();

        if v.is_empty() {
            return;
        }

        is_processing.set(true);
        cancel_flag.set(false);

        spawn_local(async move {
            let total = items.get_untracked().len();

            for idx in 0..total {
                // Check cancel flag
                if cancel_flag.get_untracked() {
                    break;
                }

                // Mark current item as processing
                items.update(|list| {
                    if let Some(item) = list.get_mut(idx) {
                        item.status = BatchItemStatus::Processing;
                    }
                });

                let text = items.get_untracked()[idx].text.clone();

                // Synthesize using JSON endpoint
                let result = synthesize_json(&url, &text, &v, Some(m.as_str())).await;

                match result {
                    Ok(response) => {
                        if let Ok(bytes) = base64::engine::general_purpose::STANDARD
                            .decode(&response.audio_base64)
                        {
                            items.update(|list| {
                                if let Some(item) = list.get_mut(idx) {
                                    item.status = BatchItemStatus::Completed;
                                    item.audio_data = Some(bytes);
                                }
                            });
                        } else {
                            items.update(|list| {
                                if let Some(item) = list.get_mut(idx) {
                                    item.status = BatchItemStatus::Failed("Decode error".to_string());
                                }
                            });
                        }
                    }
                    Err(e) => {
                        items.update(|list| {
                            if let Some(item) = list.get_mut(idx) {
                                item.status = BatchItemStatus::Failed(e.message.clone());
                            }
                        });
                    }
                }
            }

            is_processing.set(false);
        });
    };

    let cancel_processing = move |_| {
        cancel_flag.set(true);
    };

    let download_all = move |_| {
        for (idx, item) in items.get_untracked().iter().enumerate() {
            if let Some(ref bytes) = item.audio_data {
                download_wav(bytes, &format!("batch_{}.wav", idx + 1));
            }
        }
    };

    let clear_items = move |_| {
        items.set(Vec::new());
        input_text.set(String::new());
    };

    view! {
        <div class="batch-processor">
            <h3 class="section-title">"Batch Processing"</h3>

            // Input textarea
            <div class="batch-input">
                <textarea
                    rows="6"
                    placeholder="Enter multiple lines of text (one per line)..."
                    prop:value=move || input_text.get()
                    disabled=move || is_processing.get()
                    on:input=move |ev| {
                        let target = ev.target().unwrap();
                        let ta: web_sys::HtmlTextAreaElement = target.unchecked_into();
                        input_text.set(ta.value());
                    }
                />
            </div>

            // Parse and process buttons
            <div class="batch-actions">
                <button
                    class="batch-parse-btn"
                    on:click=parse_items
                    disabled=move || line_count.get() == 0 || is_processing.get()
                >
                    {move || format!("Parse ({} items)", line_count.get())}
                </button>
                <button
                    class="batch-process-btn"
                    on:click=process_all
                    disabled=move || items.get().is_empty() || is_processing.get() || voice.get().is_empty()
                >
                    "Process All"
                </button>
                <Show when=move || is_processing.get()>
                    <button class="batch-cancel-btn" on:click=cancel_processing>
                        "Cancel"
                    </button>
                </Show>
            </div>

            // Progress bar during processing
            <Show when=move || is_processing.get()>
                <div class="batch-progress">
                    <div class="batch-progress-bar">
                        <div
                            class="batch-progress-fill"
                            style:width=move || {
                                let total = items.get().len();
                                let done = completed_count.get();
                                if total > 0 {
                                    format!("{}%", (done * 100) / total)
                                } else {
                                    "0%".to_string()
                                }
                            }
                        />
                    </div>
                    <span class="batch-progress-text">
                        {move || format!("{} / {}", completed_count.get(), items.get().len())}
                    </span>
                </div>
            </Show>

            // Items list
            <Show when=move || !items.get().is_empty()>
                <div class="batch-items">
                    {move || items.get().into_iter().enumerate().map(|(idx, item)| {
                        let status_class = match &item.status {
                            BatchItemStatus::Pending => "pending",
                            BatchItemStatus::Processing => "processing",
                            BatchItemStatus::Completed => "completed",
                            BatchItemStatus::Failed(_) => "failed",
                        };
                        let status_text = match &item.status {
                            BatchItemStatus::Pending => "Pending".to_string(),
                            BatchItemStatus::Processing => "Processing...".to_string(),
                            BatchItemStatus::Completed => "Completed".to_string(),
                            BatchItemStatus::Failed(e) => format!("Failed: {}", e),
                        };
                        let display_text: String = item.text.chars().take(40).collect();
                        let display_text = if item.text.len() > 40 {
                            format!("{}...", display_text)
                        } else {
                            display_text
                        };
                        let has_audio = item.audio_data.is_some();
                        let item_bytes = item.audio_data.clone();

                        view! {
                            <div class=format!("batch-item {}", status_class)>
                                <span class="batch-item-num">{idx + 1}"."</span>
                                <span class="batch-item-text" title=item.text.clone()>{display_text}</span>
                                <span class=format!("batch-item-status {}", status_class)>
                                    {status_text}
                                </span>
                                <Show when=move || has_audio>
                                    <button
                                        class="batch-item-download"
                                        on:click={
                                            let bytes = item_bytes.clone();
                                            move |_| {
                                                if let Some(ref b) = bytes {
                                                    download_wav(b, &format!("batch_{}.wav", idx + 1));
                                                }
                                            }
                                        }
                                        title="Download"
                                    >
                                        "Download"
                                    </button>
                                </Show>
                            </div>
                        }
                    }).collect_view()}
                </div>
            </Show>

            // Download all / Clear buttons
            <Show when=move || { completed_count.get() > 0 }>
                <div class="batch-final-actions">
                    <button class="batch-download-all" on:click=download_all>
                        {move || format!("Download All ({} files)", completed_count.get())}
                    </button>
                    <button class="batch-clear" on:click=clear_items>
                        "Clear"
                    </button>
                </div>
            </Show>
        </div>
    }
}

fn download_wav(bytes: &[u8], filename: &str) {
    let array = js_sys::Uint8Array::new_with_length(bytes.len() as u32);
    array.copy_from(bytes);

    let blob_parts = js_sys::Array::new();
    blob_parts.push(&array.buffer());

    let mut options = web_sys::BlobPropertyBag::new();
    options.type_("audio/wav");

    if let Ok(blob) = web_sys::Blob::new_with_u8_array_sequence_and_options(&blob_parts, &options) {
        if let Ok(url) = web_sys::Url::create_object_url_with_blob(&blob) {
            let document = web_sys::window().unwrap().document().unwrap();
            let a: web_sys::HtmlAnchorElement = document
                .create_element("a")
                .unwrap()
                .unchecked_into();
            a.set_href(&url);
            a.set_download(filename);
            a.click();
            let _ = web_sys::Url::revoke_object_url(&url);
        }
    }
}
