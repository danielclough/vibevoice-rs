use leptos::prelude::*;
use wasm_bindgen::JsCast;

#[component]
pub fn SynthButton(
    is_loading: Signal<bool>,
    use_streaming: RwSignal<bool>,
    #[prop(into)] on_synthesize: Callback<()>,
    #[prop(into)] on_streaming_change: Callback<bool>,
) -> impl IntoView {
    let on_click = move |_| {
        on_synthesize.run(());
    };

    let on_toggle = move |ev: web_sys::Event| {
        let target = ev.target().unwrap();
        let input: web_sys::HtmlInputElement = target.unchecked_into();
        let checked = input.checked();
        use_streaming.set(checked);
        on_streaming_change.run(checked);
    };

    let button_text = move || {
        if is_loading.get() {
            "Synthesizing..."
        } else {
            "Synthesize"
        }
    };

    view! {
        <div class="synth-section">
            <div class="streaming-toggle">
                <label class="checkbox-label">
                    <input
                        type="checkbox"
                        checked=move || use_streaming.get()
                        on:change=on_toggle
                    />
                    "Enable streaming"
                </label>
            </div>
            <button
                class="synth-button"
                on:click=on_click
                disabled=move || is_loading.get()
            >
                {button_text}
            </button>
        </div>
    }
}
