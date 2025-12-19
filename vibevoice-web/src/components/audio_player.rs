use leptos::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{Blob, BlobPropertyBag, HtmlAnchorElement, Url};

#[component]
pub fn AudioPlayer(audio_data: Signal<Option<Vec<u8>>>) -> impl IntoView {
    let audio_url = Memo::new(move |_| {
        audio_data.get().and_then(|data| {
            let uint8_array = js_sys::Uint8Array::from(data.as_slice());
            let array = js_sys::Array::new();
            array.push(&uint8_array.buffer());

            let mut options = BlobPropertyBag::new();
            options.type_("audio/wav");

            Blob::new_with_u8_array_sequence_and_options(&array, &options)
                .ok()
                .and_then(|blob| Url::create_object_url_with_blob(&blob).ok())
        })
    });

    let has_audio = move || audio_data.get().is_some();

    let download = move |_| {
        if let Some(url) = audio_url.get() {
            let window = web_sys::window().unwrap();
            let document = window.document().unwrap();
            let a: HtmlAnchorElement = document.create_element("a").unwrap().unchecked_into();
            a.set_href(&url);
            a.set_download("synthesis.wav");
            a.click();
        }
    };

    view! {
        <div class="audio-section" class:hidden=move || !has_audio()>
            <label>"Audio Output"</label>
            <div class="audio-controls">
                {move || audio_url.get().map(|url| {
                    view! {
                        <audio controls=true src=url />
                    }
                })}
                <button class="download-button" on:click=download disabled=move || !has_audio()>
                    "Download WAV"
                </button>
            </div>
        </div>
    }
}
