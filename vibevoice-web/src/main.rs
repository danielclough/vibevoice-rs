mod api;
mod app;
mod components;
mod sse;
mod storage;
mod tauri;

use app::App;
use leptos::prelude::*;

fn main() {
    console_error_panic_hook::set_once();
    mount_to_body(App);
}
