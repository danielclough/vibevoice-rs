//! Toast notification system.

use leptos::prelude::*;
use leptos::task::spawn_local;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Toast notification type.
#[derive(Debug, Clone, PartialEq)]
pub enum ToastType {
    Success,
    Error,
    Info,
}

/// A single toast message.
#[derive(Debug, Clone)]
pub struct ToastMessage {
    pub id: usize,
    pub message: String,
    pub toast_type: ToastType,
}

/// Global counter for toast IDs.
static TOAST_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn next_toast_id() -> usize {
    TOAST_COUNTER.fetch_add(1, Ordering::SeqCst)
}

/// Show a toast notification that auto-dismisses after 5 seconds.
pub fn show_toast(toasts: RwSignal<Vec<ToastMessage>>, message: &str, toast_type: ToastType) {
    let id = next_toast_id();

    toasts.update(|t| {
        t.push(ToastMessage {
            id,
            message: message.to_string(),
            toast_type,
        });
    });

    // Auto-dismiss after 5 seconds
    spawn_local(async move {
        gloo_timers::future::TimeoutFuture::new(5000).await;
        toasts.update(|t| t.retain(|m| m.id != id));
    });
}

/// Container component for rendering toast notifications.
#[component]
pub fn ToastContainer(
    toasts: Signal<Vec<ToastMessage>>,
    #[prop(into)] on_dismiss: Callback<usize>,
) -> impl IntoView {
    view! {
        <div class="toast-container">
            {move || toasts.get().into_iter().map(|toast| {
                let id = toast.id;
                let class = match toast.toast_type {
                    ToastType::Success => "toast toast-success",
                    ToastType::Error => "toast toast-error",
                    ToastType::Info => "toast toast-info",
                };
                let icon = match toast.toast_type {
                    ToastType::Success => "OK",
                    ToastType::Error => "Error",
                    ToastType::Info => "Info",
                };

                view! {
                    <div class=class>
                        <span class="toast-icon">{icon}</span>
                        <span class="toast-message">{toast.message}</span>
                        <button
                            class="toast-dismiss"
                            on:click=move |_| on_dismiss.run(id)
                        >
                            "Dismiss"
                        </button>
                    </div>
                }
            }).collect_view()}
        </div>
    }
}
