use base64::Engine;
use gloo_storage::{LocalStorage, Storage};
use leptos::prelude::*;
use leptos::task::spawn_local;

use crate::api::client::{fetch_voices, synthesize_json};
use crate::tauri;
use crate::storage;
use crate::storage::history::HistoryEntry;
use crate::storage::templates::TextTemplate;
use crate::components::audio_history::AudioHistory;
use crate::components::audio_player::AudioPlayer;
use crate::components::batch_processor::BatchProcessor;
use crate::components::model_selector::{Model, ModelSelector};
use crate::components::progress::Progress;
use crate::components::server_config::ServerConfig;
use crate::components::server_setup::ServerSetup;
use crate::components::sidebar::Sidebar;
use crate::components::synth_button::SynthButton;
use crate::components::text_input::TextInput;
use crate::components::text_templates::TextTemplates;
use crate::components::toast::{show_toast, ToastContainer, ToastMessage, ToastType};
use crate::components::voice_selector::VoiceSelector;
use crate::sse::stream::{start_streaming, SseEvent, StreamingState};

#[component]
pub fn App() -> impl IntoView {
    // Check if we have a saved server from previous session
    let saved_server: Option<String> = LocalStorage::get(storage::STORAGE_LAST_SERVER).ok();

    // Load persisted state from localStorage
    let initial_model: Model = LocalStorage::get::<String>(storage::STORAGE_MODEL)
        .ok()
        .and_then(|s| Model::from_str(&s))
        .unwrap_or_default();
    let initial_streaming: bool = LocalStorage::get(storage::STORAGE_STREAMING).unwrap_or(true);

    // Setup wizard state - show if no saved server
    let show_setup = RwSignal::new(saved_server.is_none());
    let is_connecting = RwSignal::new(saved_server.is_some()); // Show connecting state if we have a saved server

    // State signals
    let server_url = RwSignal::new(saved_server.clone().unwrap_or_default());
    let model = RwSignal::new(initial_model);
    let text = RwSignal::new(String::new());
    let selected_voice = RwSignal::new(String::new());
    let use_streaming = RwSignal::new(initial_streaming);
    let is_loading = RwSignal::new(false);
    let progress = RwSignal::new(None::<(usize, Option<usize>)>);
    let audio_data = RwSignal::new(None::<Vec<u8>>);
    let error_message = RwSignal::new(None::<String>);
    let status_message = RwSignal::new(None::<String>);

    // Audio history
    let audio_history = RwSignal::new(storage::history::load());

    // Text templates
    let templates = RwSignal::new(storage::templates::load());

    // UI state
    let sidebar_open = RwSignal::new(true);
    let toasts = RwSignal::new(Vec::<ToastMessage>::new());
    let batch_mode = RwSignal::new(false);

    // Voice lists
    let voices = RwSignal::new(Vec::<String>::new());
    let samples = RwSignal::new(Vec::<String>::new());

    // Fetch voices helper
    let fetch_voices_action = move |url: String| {
        spawn_local(async move {
            match fetch_voices(&url).await {
                Ok(response) => {
                    // Auto-select first voice if none selected
                    if selected_voice.get_untracked().is_empty() {
                        let current_model = model.get_untracked();
                        let first_voice = if current_model.uses_voices() {
                            response.voices.first().cloned()
                        } else {
                            response.samples.first().cloned()
                        };
                        if let Some(v) = first_voice {
                            selected_voice.set(v);
                        }
                    }
                    voices.set(response.voices);
                    samples.set(response.samples);
                }
                Err(e) => {
                    show_toast(toasts, &format!("Failed to fetch voices: {}", e), ToastType::Error);
                }
            }
        });
    };

    // On mount: try to auto-connect to saved server
    if let Some(saved_url) = saved_server {
        spawn_local(async move {
            // Try to connect to the saved server
            match fetch_voices(&saved_url).await {
                Ok(response) => {
                    // Success - we're connected
                    server_url.set(saved_url.clone());
                    let _ = LocalStorage::set(storage::STORAGE_SERVER_URL, &saved_url);

                    // Auto-select first voice
                    let current_model = model.get_untracked();
                    let first_voice = if current_model.uses_voices() {
                        response.voices.first().cloned()
                    } else {
                        response.samples.first().cloned()
                    };
                    if let Some(v) = first_voice {
                        selected_voice.set(v);
                    }
                    voices.set(response.voices);
                    samples.set(response.samples);

                    is_connecting.set(false);
                    show_setup.set(false);
                }
                Err(_) => {
                    // Failed to connect - show setup wizard
                    is_connecting.set(false);
                    show_setup.set(true);
                }
            }
        });
    }

    // Callback when server setup completes successfully
    let on_server_connected = Callback::new(move |url: String| {
        // Save as last server for next launch
        let _ = LocalStorage::set(storage::STORAGE_LAST_SERVER, &url);
        let _ = LocalStorage::set(storage::STORAGE_SERVER_URL, &url);
        server_url.set(url.clone());

        // Fetch voices and close setup
        fetch_voices_action(url);
        show_setup.set(false);
    });

    // Callbacks
    let on_server_change = Callback::new(move |url: String| {
        let _ = LocalStorage::set(storage::STORAGE_SERVER_URL, &url);
        fetch_voices_action(url);
    });

    let on_model_change = Callback::new(move |m: Model| {
        let _ = LocalStorage::set(storage::STORAGE_MODEL, m.as_str());
        // Auto-select first voice/sample for the new model
        let first_voice = if m.uses_voices() {
            voices.get_untracked().first().cloned()
        } else {
            samples.get_untracked().first().cloned()
        };
        selected_voice.set(first_voice.unwrap_or_default());
    });

    let on_streaming_change = Callback::new(move |streaming: bool| {
        let _ = LocalStorage::set(storage::STORAGE_STREAMING, streaming);
    });

    let on_synthesize = Callback::new(move |_: ()| {
        let url = server_url.get_untracked();
        let txt = text.get_untracked();
        let voice = selected_voice.get_untracked();
        let m = model.get_untracked();
        let streaming = use_streaming.get_untracked();

        if txt.trim().is_empty() {
            show_toast(toasts, "Please enter some text to synthesize", ToastType::Error);
            return;
        }

        if voice.is_empty() {
            show_toast(toasts, "Please select a voice", ToastType::Error);
            return;
        }

        is_loading.set(true);
        audio_data.set(None);
        progress.set(None);
        status_message.set(Some(format!(
            "Starting synthesis with {} model (first request may take longer while model loads)...",
            m.display_name()
        )));

        if streaming {
            spawn_local(async move {
                let mut state = StreamingState::default();

                let result = start_streaming(&url, &txt, &voice, Some(m.as_str()), |event| {
                    match &event {
                        SseEvent::Header(_) => {
                            status_message.set(Some("Generating audio...".to_string()));
                        }
                        SseEvent::Chunk(c) => {
                            status_message.set(None);
                            progress.set(Some((c.step, state.total_steps)));
                        }
                        SseEvent::Complete(c) => {
                            progress.set(Some((c.total_steps, Some(c.total_steps))));
                        }
                        _ => {}
                    }
                    state.apply_event(event);
                }).await;

                is_loading.set(false);
                progress.set(None);
                status_message.set(None);

                match result {
                    Ok(()) => {
                        if let Some(err) = state.error {
                            show_toast(toasts, &err, ToastType::Error);
                        } else if let Some(wav) = state.to_wav_bytes() {
                            audio_data.set(Some(wav.clone()));

                            // Save to history
                            let entry = HistoryEntry {
                                id: storage::generate_id(),
                                created_at: storage::now_millis(),
                                text: txt.clone(),
                                voice: voice.clone(),
                                model: m.as_str().to_string(),
                                audio_b64: base64::engine::general_purpose::STANDARD.encode(&wav),
                                server_url: url.clone(),
                            };
                            audio_history.update(|h| {
                                storage::history::add(h, entry);
                                storage::history::save(h);
                            });
                        } else {
                            show_toast(toasts, "No audio data received", ToastType::Error);
                        }
                    }
                    Err(e) => {
                        show_toast(toasts, &e, ToastType::Error);
                    }
                }
            });
        } else {
            spawn_local(async move {
                match synthesize_json(&url, &txt, &voice, Some(m.as_str())).await {
                    Ok(response) => {
                        match base64::Engine::decode(
                            &base64::engine::general_purpose::STANDARD,
                            &response.audio_base64,
                        ) {
                            Ok(wav_bytes) => {
                                audio_data.set(Some(wav_bytes.clone()));

                                // Save to history
                                let entry = HistoryEntry {
                                    id: storage::generate_id(),
                                    created_at: storage::now_millis(),
                                    text: txt.clone(),
                                    voice: voice.clone(),
                                    model: m.as_str().to_string(),
                                    audio_b64: base64::engine::general_purpose::STANDARD.encode(&wav_bytes),
                                    server_url: url.clone(),
                                };
                                audio_history.update(|h| {
                                    storage::history::add(h, entry);
                                    storage::history::save(h);
                                });
                            }
                            Err(e) => {
                                show_toast(toasts, &format!("Failed to decode audio: {}", e), ToastType::Error);
                            }
                        }
                    }
                    Err(e) => {
                        show_toast(toasts, &e.message, ToastType::Error);
                    }
                }
                is_loading.set(false);
                status_message.set(None);
            });
        }
    });

    view! {
        <div class="app-container">
            // Show connecting state while auto-connecting to saved server
            <Show when=move || is_connecting.get()>
                <div class="connecting-screen">
                    <h1>"VibeVoice"</h1>
                    <p>"Connecting to server..."</p>
                </div>
            </Show>

            // Show setup wizard if no server connected
            <Show when=move || show_setup.get() && !is_connecting.get()>
                <ServerSetup on_connected=on_server_connected />
            </Show>

            // Show main app when connected
            <Show when=move || !show_setup.get() && !is_connecting.get()>
                <header class="app-header">
                    <div class="logo-section">
                        <div class="app-logo" role="img" aria-label="VibeVoice logo"></div>
                        <div>
                            <h1 class="app-title">"VibeVoice-RS"</h1>
                            <p class="tagline">"Text-to-Speech Synthesis"</p>
                        </div>
                    </div>
                </header>

                <div class="main-layout">
                    // Sidebar with history and batch mode toggle
                    <Sidebar is_open=sidebar_open>

                        <button
                            class="mode-toggle-btn"
                            on:click=move |_| batch_mode.update(|b| *b = !*b)
                        >
                            {move || if batch_mode.get() { "Single Mode" } else { "Batch Mode" }}
                        </button>

                        <AudioHistory
                            history=audio_history.into()
                            current_server_url=server_url.into()
                            on_play=Callback::new(move |entry: HistoryEntry| {
                                if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(&entry.audio_b64) {
                                    audio_data.set(Some(bytes));
                                }
                            })
                            on_delete=Callback::new(move |id: String| {
                                audio_history.update(|h| {
                                    storage::history::remove(h, &id);
                                    storage::history::save(h);
                                });
                            })
                            on_reuse=Callback::new(move |entry: HistoryEntry| {
                                text.set(entry.text);
                                selected_voice.set(entry.voice);
                            })
                        />
                    </Sidebar>

                    // Main content area
                    <main class="main-content">
                        <div class="panel config-panel">
                            <ServerConfig server_url=server_url on_change=on_server_change />
                            <ModelSelector model=model on_change=on_model_change />
                            <VoiceSelector
                                model=model.into()
                                voices=voices.into()
                                samples=samples.into()
                                selected_voice=selected_voice
                                server_url=server_url.into()
                            />
                        </div>

                        // Conditional: Batch mode or Single mode
                        {move || if batch_mode.get() {
                            view! {
                                <div class="panel input-panel">
                                    <BatchProcessor
                                        server_url=server_url.into()
                                        voice=selected_voice.into()
                                        model=model.into()
                                    />
                                </div>
                            }.into_any()
                        } else {
                            view! {
                                <div class="panel input-panel">
                                    <TextTemplates
                                        templates=templates.into()
                                        current_text=text.into()
                                        on_select=Callback::new(move |t: TextTemplate| {
                                            text.set(t.text);
                                        })
                                        on_save=Callback::new(move |name: String| {
                                            let t = TextTemplate {
                                                id: storage::generate_id(),
                                                name,
                                                text: text.get_untracked(),
                                                created_at: storage::now_millis(),
                                            };
                                            templates.update(|ts| {
                                                ts.push(t);
                                                storage::templates::save(ts);
                                            });
                                        })
                                        on_delete=Callback::new(move |id: String| {
                                            templates.update(|ts| {
                                                ts.retain(|t| t.id != id);
                                                storage::templates::save(ts);
                                            });
                                        })
                                        on_edit=Callback::new(move |(id, new_text): (String, String)| {
                                            templates.update(|ts| {
                                                if let Some(t) = ts.iter_mut().find(|t| t.id == id) {
                                                    t.text = new_text;
                                                }
                                                storage::templates::save(ts);
                                            });
                                        })
                                    />
                                    <TextInput text=text />
                                    <SynthButton
                                        is_loading=is_loading.into()
                                        use_streaming=use_streaming
                                        on_synthesize=on_synthesize
                                        on_streaming_change=on_streaming_change
                                    />
                                </div>

                                <Progress progress=progress.into() />

                                {move || status_message.get().map(|msg| view! {
                                    <div class="status-message">{msg}</div>
                                })}

                                <AudioPlayer audio_data=audio_data.into() />
                            }.into_any()
                        }}
                    </main>
                </div>

                // Toast notifications
                <ToastContainer
                    toasts=toasts.into()
                    on_dismiss=Callback::new(move |id: usize| {
                        toasts.update(|t| t.retain(|m| m.id != id));
                    })
                />

                <footer class="app-footer">
                    <p>"VibeVoice-RS © Daniel Clough "<a href="https://github.com/danielclough/vibevoice-rs/blob/main/LICENSE">"MIT LICENSE"</a></p>
                </footer>
            </Show>
        </div>
    }
}
