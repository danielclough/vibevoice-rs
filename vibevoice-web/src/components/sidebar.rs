//! Collapsible sidebar component.

use leptos::prelude::*;

#[component]
pub fn Sidebar(
    is_open: RwSignal<bool>,
    children: Children,
) -> impl IntoView {
    view! {
        <aside
            class="sidebar"
            class:sidebar-open=move || is_open.get()
            class:sidebar-collapsed=move || !is_open.get()
        >
            <button
                class="sidebar-toggle"
                on:click=move |_| is_open.update(|o| *o = !*o)
                title=move || if is_open.get() { "Collapse sidebar" } else { "Expand sidebar" }
            >
                {move || if is_open.get() { "Hide" } else { "Show" }}
            </button>
            <div class="sidebar-content">
                {children()}
            </div>
        </aside>
    }
}
