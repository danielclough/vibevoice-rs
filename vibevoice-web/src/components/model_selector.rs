use leptos::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Model {
    #[default]
    Realtime,
    Batch1_5B,
    Batch7B,
}

impl Model {
    pub fn as_str(&self) -> &'static str {
        match self {
            Model::Realtime => "realtime",
            Model::Batch1_5B => "1.5B",
            Model::Batch7B => "7B",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "realtime" => Some(Model::Realtime),
            "1.5B" => Some(Model::Batch1_5B),
            "7B" => Some(Model::Batch7B),
            _ => None,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Model::Realtime => "Single/Fast (0.5B)",
            Model::Batch1_5B => "Multi/Script (1.5B)",
            Model::Batch7B => "Multi/Script (7B)",
        }
    }

    pub fn uses_voices(&self) -> bool {
        matches!(self, Model::Realtime)
    }
}

#[component]
pub fn ModelSelector(
    model: RwSignal<Model>,
    #[prop(into)] on_change: Callback<Model>,
) -> impl IntoView {
    let make_handler = move |m: Model| {
        move |_| {
            model.set(m);
            on_change.run(m);
        }
    };

    view! {
        <div class="config-section">
            <label>"Model"</label>
            <div class="radio-group">
                <label class="radio-label">
                    <input
                        type="radio"
                        name="model"
                        value="realtime"
                        checked=move || model.get() == Model::Realtime
                        on:change=make_handler(Model::Realtime)
                    />
                    {Model::Realtime.display_name()}
                </label>
                <label class="radio-label">
                    <input
                        type="radio"
                        name="model"
                        value="1.5B"
                        checked=move || model.get() == Model::Batch1_5B
                        on:change=make_handler(Model::Batch1_5B)
                    />
                    {Model::Batch1_5B.display_name()}
                </label>
                <label class="radio-label">
                    <input
                        type="radio"
                        name="model"
                        value="7B"
                        checked=move || model.get() == Model::Batch7B
                        on:change=make_handler(Model::Batch7B)
                    />
                    {Model::Batch7B.display_name()}
                </label>
            </div>
        </div>
    }
}
