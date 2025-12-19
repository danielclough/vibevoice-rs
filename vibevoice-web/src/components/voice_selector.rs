use leptos::prelude::*;
use wasm_bindgen::JsCast;

use super::Model;

#[component]
pub fn VoiceSelector(
    model: Signal<Model>,
    voices: Signal<Vec<String>>,
    samples: Signal<Vec<String>>,
    selected_voice: RwSignal<String>,
) -> impl IntoView {
    let options = move || {
        let m = model.get();
        if m.uses_voices() {
            voices.get()
        } else {
            samples.get()
        }
    };

    let on_change = move |ev: web_sys::Event| {
        let target = ev.target().unwrap();
        let select: web_sys::HtmlSelectElement = target.unchecked_into();
        selected_voice.set(select.value());
    };

    let label = move || {
        if model.get().uses_voices() {
            "Voice"
        } else {
            "Voice Sample"
        }
    };

    view! {
        <div class="config-section">
            <label for="voice-select">{label}</label>
            <select
                id="voice-select"
                prop:value=move || selected_voice.get()
                on:change=on_change
            >
                <option value="" disabled=true>"Select a voice..."</option>
                {move || options().into_iter().map(|v| {
                    let v_for_value = v.clone();
                    let v_for_selected = v.clone();
                    let v_for_text = v.clone();
                    view! {
                        <option value=v_for_value selected=move || selected_voice.get() == v_for_selected>
                            {v_for_text}
                        </option>
                    }
                }).collect::<Vec<_>>()}
            </select>
        </div>
    }
}
