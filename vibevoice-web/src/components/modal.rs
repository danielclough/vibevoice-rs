//! Reusable modal dialog component.

use leptos::prelude::*;

#[component]
pub fn Modal(
    is_open: Signal<bool>,
    #[prop(into)] on_close: Callback<()>,
    #[prop(into)] title: String,
    children: ChildrenFn,
) -> impl IntoView {
    let on_close_overlay = on_close.clone();
    let on_close_button = on_close.clone();

    view! {
        <Show when=move || is_open.get()>
            <div
                class="modal-overlay"
                on:click=move |_| on_close_overlay.run(())
            >
                <div
                    class="modal-content"
                    on:click=|e| e.stop_propagation()
                >
                    <div class="modal-header">
                        <h3 class="modal-title">{title.clone()}</h3>
                        <button
                            class="modal-close"
                            on:click=move |_| on_close_button.run(())
                        >
                            "Close"
                        </button>
                    </div>
                    <div class="modal-body">
                        {children()}
                    </div>
                </div>
            </div>
        </Show>
    }
}
