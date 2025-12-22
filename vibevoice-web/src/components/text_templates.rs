//! Text templates component for saving and reusing common text inputs.

use leptos::prelude::*;
use wasm_bindgen::JsCast;
use crate::storage::templates::TextTemplate;

#[component]
pub fn TextTemplates(
    templates: Signal<Vec<TextTemplate>>,
    current_text: Signal<String>,
    #[prop(into)] on_select: Callback<TextTemplate>,
    #[prop(into)] on_save: Callback<String>,
    #[prop(into)] on_delete: Callback<String>,
) -> impl IntoView {
    let show_save_input = RwSignal::new(false);
    let template_name = RwSignal::new(String::new());

    let save_template = move || {
        let name = template_name.get_untracked();
        if !name.trim().is_empty() {
            on_save.run(name.trim().to_string());
            template_name.set(String::new());
            show_save_input.set(false);
        }
    };

    let handle_keydown = move |ev: web_sys::KeyboardEvent| {
        if ev.key() == "Enter" {
            save_template();
        } else if ev.key() == "Escape" {
            show_save_input.set(false);
            template_name.set(String::new());
        }
    };

    view! {
        <div class="templates-section">
            <div class="templates-controls">
                <select
                    class="template-select"
                    on:change=move |ev| {
                        let target = ev.target().unwrap();
                        let select: web_sys::HtmlSelectElement = target.unchecked_into();
                        let id = select.value();
                        if !id.is_empty() {
                            if let Some(t) = templates.get().into_iter().find(|t| t.id == id) {
                                on_select.run(t);
                            }
                        }
                        select.set_value("");  // Reset to placeholder
                    }
                >
                    <option value="" disabled=true selected=true>"Load template..."</option>
                    {move || templates.get().into_iter().map(|t| {
                        let id = t.id.clone();
                        let name = t.name.clone();
                        view! {
                            <option value=id.clone()>
                                {name}
                            </option>
                        }
                    }).collect_view()}
                </select>
                <button
                    class="save-template-btn"
                    on:click=move |_| show_save_input.update(|v| *v = !*v)
                    disabled=move || current_text.get().trim().is_empty()
                    title="Save current text as template"
                >
                    "Save"
                </button>
            </div>
            <Show when=move || show_save_input.get()>
                <div class="save-template-input">
                    <input
                        type="text"
                        placeholder="Template name..."
                        prop:value=move || template_name.get()
                        on:input=move |ev| {
                            let target = ev.target().unwrap();
                            let input: web_sys::HtmlInputElement = target.unchecked_into();
                            template_name.set(input.value());
                        }
                        on:keydown=handle_keydown
                    />
                    <button class="confirm-btn" on:click=move |_| save_template()>"Save"</button>
                    <button
                        class="cancel-btn"
                        on:click=move |_| {
                            show_save_input.set(false);
                            template_name.set(String::new());
                        }
                    >
                        "Cancel"
                    </button>
                </div>
            </Show>
            // Template list with delete buttons (shown when there are templates)
            <Show when=move || !templates.get().is_empty()>
                <div class="templates-list">
                    {move || templates.get().into_iter().map(|t| {
                        let id_for_delete = t.id.clone();
                        let template_for_select = t.clone();
                        view! {
                            <div
                                class="template-item"
                                on:click=move |_| on_select.run(template_for_select.clone())
                            >
                                <span class="template-name">{t.name.clone()}</span>
                                <button
                                    class="template-delete-btn"
                                    on:click=move |ev| {
                                        ev.stop_propagation();
                                        on_delete.run(id_for_delete.clone());
                                    }
                                    title="Delete template"
                                >
                                    "Delete"
                                </button>
                            </div>
                        }
                    }).collect_view()}
                </div>
            </Show>
        </div>
    }
}
