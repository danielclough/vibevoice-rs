use leptos::prelude::*;
use wasm_bindgen::JsCast;

#[component]
pub fn TextInput(text: RwSignal<String>) -> impl IntoView {
    let on_input = move |ev: web_sys::Event| {
        let target = ev.target().unwrap();
        let textarea: web_sys::HtmlTextAreaElement = target.unchecked_into();
        text.set(textarea.value());
    };

    view! {
        <div class="text-section">
            <label for="tts-text">"Text to Synthesize"</label>
            <textarea
                id="tts-text"
                rows="4"
                placeholder="Enter text to convert to speech..."
                prop:value=move || text.get()
                on:input=on_input
            />
        </div>
    }
}
