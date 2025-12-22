//! Settings panel component for server configuration.

use leptos::prelude::*;
use leptos::task::spawn_local;
use wasm_bindgen::JsCast;

use crate::components::modal::Modal;
use crate::tauri::{self, TauriConfig};

/// Settings panel that allows users to configure server settings.
#[component]
pub fn Settings(
    is_open: RwSignal<bool>,
    server_url: RwSignal<String>,
    #[prop(into)] on_server_change: Callback<String>,
    /// When true, acts as part of initial setup flow
    #[prop(default = false)]
    setup_mode: bool,
    /// Called with server URL when setup completes (only used when setup_mode=true)
    #[prop(optional)]
    on_setup_complete: Option<Callback<String>>,
) -> impl IntoView {
    let is_tauri = tauri::is_tauri();

    // Local state for the form - mirrors TauriConfig
    let use_embedded = RwSignal::new(true);
    let server_port = RwSignal::new(3908u16);
    let custom_url = RwSignal::new(String::new());
    let safetensors_dir = RwSignal::new(None::<String>);
    let wav_dir = RwSignal::new(None::<String>);
    let output_dir = RwSignal::new(None::<String>);
    let default_model = RwSignal::new("realtime".to_string());

    let is_saving = RwSignal::new(false);
    let error_message = RwSignal::new(None::<String>);

    // Load current config when modal opens
    Effect::new(move || {
        if is_open.get() && is_tauri {
            spawn_local(async move {
                if let Some(config) = tauri::get_config().await {
                    let desktop = config.desktop_settings();
                    use_embedded.set(desktop.embedded_server);
                    server_port.set(config.server_port());
                    if let Some(url) = desktop.remote_server_url {
                        custom_url.set(url);
                    }
                    safetensors_dir.set(config.safetensors_dir);
                    wav_dir.set(config.wav_dir);
                    output_dir.set(config.output_dir);
                    default_model.set(desktop.default_model);
                }
            });
        }
    });

    let on_close = Callback::new(move |_: ()| {
        is_open.set(false);
        error_message.set(None);
    });

    // Directory picker handlers
    let pick_safetensors_dir = move |_| {
        spawn_local(async move {
            if let Some(path) = tauri::pick_directory().await {
                safetensors_dir.set(Some(path));
            }
        });
    };

    let pick_wav_dir = move |_| {
        spawn_local(async move {
            if let Some(path) = tauri::pick_directory().await {
                wav_dir.set(Some(path));
            }
        });
    };

    let pick_output_dir = move |_| {
        spawn_local(async move {
            if let Some(path) = tauri::pick_directory().await {
                output_dir.set(Some(path));
            }
        });
    };

    let on_save = {
        let on_server_change = on_server_change.clone();
        let on_setup_complete = on_setup_complete.clone();
        move |_| {
            is_saving.set(true);
            error_message.set(None);

            let embedded = use_embedded.get_untracked();
            let remote_url = if embedded {
                None
            } else {
                Some(custom_url.get_untracked())
            };

            let config = TauriConfig {
                host: None,
                port: Some(server_port.get_untracked()),
                safetensors_dir: safetensors_dir.get_untracked(),
                wav_dir: wav_dir.get_untracked(),
                output_dir: output_dir.get_untracked(),
                web_dir: None,
                cors_origins: vec![],
                desktop: Some(tauri::DesktopSettings {
                    embedded_server: embedded,
                    remote_server_url: remote_url.clone(),
                    default_model: default_model.get_untracked(),
                    hotkey_show: None,
                    start_minimized: false,
                    show_notifications: true,
                }),
            };

            let on_server_change = on_server_change.clone();
            let on_setup_complete = on_setup_complete.clone();

            spawn_local(async move {
                // Save config first
                if let Err(e) = tauri::save_config(&config).await {
                    error_message.set(Some(format!("Failed to save config: {}", e)));
                    is_saving.set(false);
                    return;
                }

                if embedded {
                    // Restart embedded server with new config
                    match tauri::start_embedded_server().await {
                        Ok(server_url_value) => {
                            server_url.set(server_url_value.clone());
                            // In setup mode, call on_setup_complete instead of on_server_change
                            if setup_mode {
                                if let Some(callback) = on_setup_complete {
                                    callback.run(server_url_value);
                                }
                            } else {
                                on_server_change.run(server_url_value);
                            }
                            is_open.set(false);
                        }
                        Err(e) => {
                            error_message.set(Some(e));
                        }
                    }
                } else {
                    // Use custom URL
                    let url = remote_url.unwrap_or_default();
                    if url.is_empty() {
                        error_message.set(Some("Please enter a server URL".to_string()));
                    } else {
                        server_url.set(url.clone());
                        // In setup mode, call on_setup_complete instead of on_server_change
                        if setup_mode {
                            if let Some(callback) = on_setup_complete {
                                callback.run(url);
                            }
                        } else {
                            on_server_change.run(url);
                        }
                        is_open.set(false);
                    }
                }
                is_saving.set(false);
            });
        }
    };

    view! {
        <Modal is_open=is_open.into() on_close=on_close title="Settings">
            <div class="settings-content">
                // Error message
                {move || error_message.get().map(|msg| view! {
                    <div class="settings-error">{msg}</div>
                })}

                // Server mode selection (Tauri only)
                <Show when=move || is_tauri>
                    <div class="settings-section">
                        <label class="settings-label">"Server Mode"</label>
                        <div class="settings-radio-group">
                            <label class="settings-radio">
                                <input
                                    type="radio"
                                    name="server-mode"
                                    checked=move || use_embedded.get()
                                    on:change=move |_| use_embedded.set(true)
                                />
                                <span>"Use Local Server"</span>
                            </label>
                            <label class="settings-radio">
                                <input
                                    type="radio"
                                    name="server-mode"
                                    checked=move || !use_embedded.get()
                                    on:change=move |_| use_embedded.set(false)
                                />
                                <span>"Use Remote Server"</span>
                            </label>
                        </div>
                    </div>

                    // Local server settings (shown when using embedded)
                    <Show when=move || use_embedded.get()>
                        // Safetensors Directory
                        <div class="settings-section">
                            <label class="settings-label">"Safetensors Directory"</label>
                            <div class="settings-path-picker">
                                <input
                                    type="text"
                                    class="settings-input"
                                    readonly=true
                                    placeholder="Select safetensors directory..."
                                    prop:value=move || safetensors_dir.get().unwrap_or_default()
                                />
                                <button class="settings-browse-btn" on:click=pick_safetensors_dir>
                                    "Browse"
                                </button>
                            </div>
                            <span class="settings-hint">"Directory containing .safetensors voice files (for realtime model)"</span>
                        </div>

                        // WAV Directory
                        <div class="settings-section">
                            <label class="settings-label">"WAV Directory"</label>
                            <div class="settings-path-picker">
                                <input
                                    type="text"
                                    class="settings-input"
                                    readonly=true
                                    placeholder="Select WAV samples directory..."
                                    prop:value=move || wav_dir.get().unwrap_or_default()
                                />
                                <button class="settings-browse-btn" on:click=pick_wav_dir>
                                    "Browse"
                                </button>
                            </div>
                            <span class="settings-hint">"Directory containing .wav samples (for batch models)"</span>
                        </div>

                        // Output Directory
                        <div class="settings-section">
                            <label class="settings-label">"Output Directory"</label>
                            <div class="settings-path-picker">
                                <input
                                    type="text"
                                    class="settings-input"
                                    readonly=true
                                    placeholder="Select output directory..."
                                    prop:value=move || output_dir.get().unwrap_or_default()
                                />
                                <button class="settings-browse-btn" on:click=pick_output_dir>
                                    "Browse"
                                </button>
                            </div>
                            <span class="settings-hint">"Directory to save generated audio files"</span>
                        </div>

                        // Default Model
                        <div class="settings-section">
                            <label class="settings-label">"Default Model"</label>
                            <select
                                class="settings-select"
                                prop:value=move || default_model.get()
                                on:change=move |ev| {
                                    let target = ev.target().unwrap();
                                    let select: web_sys::HtmlSelectElement = target.unchecked_into();
                                    default_model.set(select.value());
                                }
                            >
                                <option value="realtime">"Realtime (fastest)"</option>
                                <option value="1.5B">"1.5B (balanced)"</option>
                                <option value="7B">"7B (highest quality)"</option>
                            </select>
                        </div>

                        // Server Port
                        <div class="settings-section">
                            <label class="settings-label">"Server Port"</label>
                            <input
                                type="number"
                                class="settings-input settings-input-small"
                                min="1024"
                                max="65535"
                                prop:value=move || server_port.get().to_string()
                                on:input=move |ev| {
                                    let target = ev.target().unwrap();
                                    let input: web_sys::HtmlInputElement = target.unchecked_into();
                                    if let Ok(port) = input.value().parse::<u16>() {
                                        server_port.set(port);
                                    }
                                }
                            />
                        </div>
                    </Show>
                </Show>

                // Custom URL input (shown when not using embedded, or always in browser)
                <Show when=move || !use_embedded.get() || !is_tauri>
                    <div class="settings-section">
                        <label class="settings-label" for="custom-server-url">"Server URL"</label>
                        <input
                            type="text"
                            id="custom-server-url"
                            class="settings-input"
                            placeholder="http://localhost:3908"
                            prop:value=move || custom_url.get()
                            on:input=move |ev| {
                                let target = ev.target().unwrap();
                                let input: web_sys::HtmlInputElement = target.unchecked_into();
                                custom_url.set(input.value());
                            }
                        />
                    </div>
                </Show>

                // Save button
                <div class="settings-actions">
                    <button
                        class="settings-save-btn"
                        disabled=move || is_saving.get()
                        on:click=on_save
                    >
                        {move || if is_saving.get() { "Saving..." } else { "Save & Apply" }}
                    </button>
                </div>
            </div>
        </Modal>
    }
}

/// Settings button to open the settings modal.
#[component]
pub fn SettingsButton(
    #[prop(into)] on_click: Callback<()>,
) -> impl IntoView {
    view! {
        <button
            class="settings-btn"
            title="Settings"
            on:click=move |_| on_click.run(())
        >
            "Settings"
        </button>
    }
}
