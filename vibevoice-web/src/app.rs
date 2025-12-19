use gloo_storage::{LocalStorage, Storage};
use leptos::prelude::*;
use leptos::task::spawn_local;

use crate::api::{fetch_voices, synthesize_json, VoicesResponse};
use crate::components::*;
use crate::sse::{start_streaming, SseEvent, StreamingState};

const STORAGE_SERVER_URL: &str = "vibevoice.server_url";
const STORAGE_MODEL: &str = "vibevoice.model";
const STORAGE_STREAMING: &str = "vibevoice.use_streaming";
const DEFAULT_SERVER_URL: &str = "http://localhost:3000";

#[component]
pub fn App() -> impl IntoView {
    // Load persisted state from localStorage
    let initial_server_url: String = LocalStorage::get(STORAGE_SERVER_URL).unwrap_or_else(|_| DEFAULT_SERVER_URL.to_string());
    let initial_model: Model = LocalStorage::get::<String>(STORAGE_MODEL)
        .ok()
        .and_then(|s| Model::from_str(&s))
        .unwrap_or_default();
    let initial_streaming: bool = LocalStorage::get(STORAGE_STREAMING).unwrap_or(true);

    // State signals
    let server_url = RwSignal::new(initial_server_url);
    let model = RwSignal::new(initial_model);
    let text = RwSignal::new(String::new());
    let selected_voice = RwSignal::new(String::new());
    let use_streaming = RwSignal::new(initial_streaming);
    let is_loading = RwSignal::new(false);
    let progress = RwSignal::new(None::<(usize, Option<usize>)>);
    let audio_data = RwSignal::new(None::<Vec<u8>>);
    let error_message = RwSignal::new(None::<String>);
    let status_message = RwSignal::new(None::<String>);

    // Voice lists
    let voices = RwSignal::new(Vec::<String>::new());
    let samples = RwSignal::new(Vec::<String>::new());

    // Fetch voices on mount and when server URL changes
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
                    error_message.set(None);
                }
                Err(e) => {
                    error_message.set(Some(format!("Failed to fetch voices: {}", e)));
                }
            }
        });
    };

    // Initial fetch
    {
        let url = server_url.get_untracked();
        fetch_voices_action(url);
    }

    // Callbacks
    let on_server_change = Callback::new(move |url: String| {
        let _ = LocalStorage::set(STORAGE_SERVER_URL, &url);
        fetch_voices_action(url);
    });

    let on_model_change = Callback::new(move |m: Model| {
        let _ = LocalStorage::set(STORAGE_MODEL, m.as_str());
        // Auto-select first voice/sample for the new model
        let first_voice = if m.uses_voices() {
            voices.get_untracked().first().cloned()
        } else {
            samples.get_untracked().first().cloned()
        };
        selected_voice.set(first_voice.unwrap_or_default());
    });

    let on_streaming_change = Callback::new(move |streaming: bool| {
        let _ = LocalStorage::set(STORAGE_STREAMING, streaming);
    });

    let on_synthesize = Callback::new(move |_: ()| {
        let url = server_url.get_untracked();
        let txt = text.get_untracked();
        let voice = selected_voice.get_untracked();
        let m = model.get_untracked();
        let streaming = use_streaming.get_untracked();

        if txt.trim().is_empty() {
            error_message.set(Some("Please enter some text to synthesize".to_string()));
            return;
        }

        if voice.is_empty() {
            error_message.set(Some("Please select a voice".to_string()));
            return;
        }

        is_loading.set(true);
        error_message.set(None);
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
                            error_message.set(Some(err));
                        } else if let Some(wav) = state.to_wav_bytes() {
                            audio_data.set(Some(wav));
                        } else {
                            error_message.set(Some("No audio data received".to_string()));
                        }
                    }
                    Err(e) => {
                        error_message.set(Some(e));
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
                                audio_data.set(Some(wav_bytes));
                            }
                            Err(e) => {
                                error_message.set(Some(format!("Failed to decode audio: {}", e)));
                            }
                        }
                    }
                    Err(e) => {
                        error_message.set(Some(e.message));
                    }
                }
                is_loading.set(false);
                status_message.set(None);
            });
        }
    });

    view! {
        <div class="app">
            <header>
                <h1>"VibeVoice TTS"</h1>
            </header>

            <main>
                <div class="panel config-panel">
                    <ServerConfig server_url=server_url on_change=on_server_change />
                    <ModelSelector model=model on_change=on_model_change />
                    <VoiceSelector
                        model=model.into()
                        voices=voices.into()
                        samples=samples.into()
                        selected_voice=selected_voice
                    />
                </div>

                <div class="panel input-panel">
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

                {move || error_message.get().map(|msg| view! {
                    <div class="error-message">{msg}</div>
                })}

                <AudioPlayer audio_data=audio_data.into() />
            </main>

            <footer style="background-color:black;">
                <p>"VibeVoice-RS © Daniel Clough "<a href="https://github.com/danielclough/vibevoice-rs/blob/main/LICENSE">"MIT LICENSE"</a></p>
            </footer>
        </div>
    }
}
