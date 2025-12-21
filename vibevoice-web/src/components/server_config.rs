use leptos::prelude::*;
use wasm_bindgen::JsCast;

#[component]
pub fn ServerConfig(
    server_url: RwSignal<String>,
    #[prop(into)] on_change: Callback<String>,
) -> impl IntoView {
    let on_input = move |ev: web_sys::Event| {
        let target = ev.target().unwrap();
        let input: web_sys::HtmlInputElement = target.unchecked_into();
        let value = input.value();
        server_url.set(value.clone());
        on_change.run(value);
    };

    view! {
        <div class="config-section">
            <label for="server-url">"Server URL"</label>
            <input
                type="text"
                id="server-url"
                placeholder="http://localhost:3908"
                prop:value=move || server_url.get()
                on:input=on_input
            />
        </div>
    }
}
