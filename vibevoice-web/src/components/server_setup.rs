//! Server setup wizard component for first-run experience.

use leptos::prelude::*;
use leptos::task::spawn_local;
use wasm_bindgen::JsCast;

use crate::api::client::fetch_voices;
use crate::tauri;

/// A hosted server option
#[derive(Debug, Clone)]
pub struct HostedServer {
    pub name: &'static str,
    pub url: Option<&'static str>, // None = coming soon
    pub description: &'static str,
}

/// Connection state for the setup wizard
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionState {
    Idle,
    Connecting,
    Connected,
    Failed(String),
    StartingLocal,
}

/// Known hosted servers (placeholder entries for now)
const HOSTED_SERVERS: &[HostedServer] = &[HostedServer {
    name: "VibeVoice Cloud",
    url: None, // Coming soon
    description: "Hosted TTS service (coming soon)",
}];

#[component]
pub fn ServerSetup(
    /// Called when successfully connected to a server
    #[prop(into)]
    on_connected: Callback<String>,
) -> impl IntoView {
    let connection_state = RwSignal::new(ConnectionState::Idle);
    let custom_url = RwSignal::new(String::new());
    let is_tauri = RwSignal::new(false);

    // Check if running in Tauri on mount
    spawn_local(async move {
        is_tauri.set(tauri::is_tauri());
    });

    // Try to connect to a server URL
    let try_connect = {
        let on_connected = on_connected.clone();
        move |url: String| {
            connection_state.set(ConnectionState::Connecting);
            let on_connected = on_connected.clone();

            spawn_local(async move {
                match fetch_voices(&url).await {
                    Ok(_) => {
                        connection_state.set(ConnectionState::Connected);
                        on_connected.run(url);
                    }
                    Err(e) => {
                        connection_state.set(ConnectionState::Failed(e.message));
                    }
                }
            });
        }
    };

    // Handle custom URL connect button
    let try_connect_clone = try_connect.clone();
    let on_custom_connect = move |_| {
        let url = custom_url.get_untracked().trim().to_string();
        if url.is_empty() {
            connection_state.set(ConnectionState::Failed(
                "Please enter a server URL".to_string(),
            ));
            return;
        }
        try_connect_clone(url);
    };

    // Handle starting local server (Tauri only)
    let on_start_local = {
        let on_connected = on_connected.clone();
        move |_| {
            connection_state.set(ConnectionState::StartingLocal);
            let on_connected = on_connected.clone();

            spawn_local(async move {
                match tauri::start_embedded_server().await {
                    Some(url) => {
                        connection_state.set(ConnectionState::Connected);
                        on_connected.run(url);
                    }
                    None => {
                        connection_state.set(ConnectionState::Failed(
                            "Failed to start local server".to_string(),
                        ));
                    }
                }
            });
        }
    };

    let is_busy = Memo::new(move |_| {
        matches!(
            connection_state.get(),
            ConnectionState::Connecting | ConnectionState::StartingLocal
        )
    });

    let on_custom_connect_keydown = on_custom_connect.clone();

    view! {
        <div class="server-setup">
            <div class="setup-header">
                <h1>"VibeVoice"</h1>
                <p class="tagline">"Text-to-Speech Synthesis"</p>
            </div>

            <div class="setup-content">
                <h2>"Connect to a Server"</h2>

                // Error message
                {move || {
                    if let ConnectionState::Failed(msg) = connection_state.get() {
                        Some(view! { <div class="setup-error">{msg}</div> })
                    } else {
                        None
                    }
                }}

                // Hosted servers section
                <HostedServersSection try_connect=try_connect.clone() is_busy=is_busy />

                <div class="setup-divider">
                    <span>"or"</span>
                </div>

                // Custom URL section
                <div class="setup-section">
                    <h3>"Custom Server"</h3>
                    <CustomUrlForm
                        custom_url=custom_url
                        is_busy=is_busy
                        on_connect=on_custom_connect_keydown
                    />
                </div>

                // Local server section (Tauri only)
                <Show when=move || is_tauri.get()>
                    <div class="setup-divider">
                        <span>"or"</span>
                    </div>
                    <LocalServerSection
                        connection_state=connection_state.into()
                        is_busy=is_busy
                        on_start=on_start_local
                    />
                </Show>
            </div>
        </div>
    }
}

#[component]
fn HostedServersSection<F>(try_connect: F, is_busy: Memo<bool>) -> impl IntoView
where
    F: Fn(String) + Clone + 'static,
{
    view! {
        <div class="setup-section">
            <h3>"Hosted Servers"</h3>
            <div class="hosted-list">
                {HOSTED_SERVERS
                    .iter()
                    .map(|server| {
                        let is_available = server.url.is_some();
                        let url = server.url;
                        let try_connect = try_connect.clone();
                        view! {
                            <div class="hosted-item" class:disabled=!is_available>
                                <div class="hosted-info">
                                    <span class="hosted-name">{server.name}</span>
                                    <span class="hosted-desc">{server.description}</span>
                                </div>
                                {if let Some(url) = url {
                                    Some(
                                        view! {
                                            <button
                                                class="hosted-connect-btn"
                                                disabled=is_busy
                                                on:click=move |_| try_connect(url.to_string())
                                            >
                                                "Connect"
                                            </button>
                                        },
                                    )
                                } else {
                                    None
                                }}
                            </div>
                        }
                    })
                    .collect_view()}
            </div>
        </div>
    }
}

#[component]
fn CustomUrlForm<F>(custom_url: RwSignal<String>, is_busy: Memo<bool>, on_connect: F) -> impl IntoView
where
    F: Fn(()) + Clone + 'static,
{
    let on_connect_click = on_connect.clone();
    let on_connect_keydown = on_connect.clone();

    view! {
        <div class="custom-url-form">
            <input
                type="text"
                class="custom-url-input"
                placeholder="http://localhost:3908"
                disabled=is_busy
                prop:value=move || custom_url.get()
                on:input=move |ev| {
                    let target = ev.target().unwrap();
                    let input: web_sys::HtmlInputElement = target.unchecked_into();
                    custom_url.set(input.value());
                }
                on:keydown=move |ev| {
                    if ev.key() == "Enter" {
                        on_connect_keydown(());
                    }
                }
            />
            <button
                class="custom-connect-btn"
                disabled=is_busy
                on:click=move |_| on_connect_click(())
            >
                {move || {
                    if is_busy.get() {
                        "Connecting..."
                    } else {
                        "Connect"
                    }
                }}
            </button>
        </div>
    }
}

#[component]
fn LocalServerSection<F>(
    connection_state: Signal<ConnectionState>,
    is_busy: Memo<bool>,
    on_start: F,
) -> impl IntoView
where
    F: Fn(()) + Clone + 'static,
{
    view! {
        <div class="setup-section">
            <h3>"Local Server"</h3>
            <p class="section-desc">"Run TTS locally using your GPU"</p>
            <button class="start-local-btn" disabled=is_busy on:click=move |_| on_start(())>
                {move || {
                    if connection_state.get() == ConnectionState::StartingLocal {
                        "Starting..."
                    } else {
                        "Start Local Server"
                    }
                }}
            </button>
        </div>
    }
}
