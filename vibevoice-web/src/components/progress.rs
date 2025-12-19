use leptos::prelude::*;

#[component]
pub fn Progress(progress: Signal<Option<(usize, Option<usize>)>>) -> impl IntoView {
    let progress_text = move || {
        progress.get().map(|(current, total)| {
            match total {
                Some(t) => format!("Step {} / {}", current, t),
                None => format!("Step {}", current),
            }
        })
    };

    let progress_percent = move || {
        progress.get().and_then(|(current, total)| {
            total.map(|t| if t > 0 { (current * 100) / t } else { 0 })
        })
    };

    view! {
        <div class="progress-section" class:hidden=move || progress.get().is_none()>
            {move || progress_text().map(|text| {
                view! {
                    <div class="progress-info">
                        <span class="progress-text">{text}</span>
                        {move || progress_percent().map(|pct| {
                            view! {
                                <div class="progress-bar">
                                    <div
                                        class="progress-fill"
                                        style:width=format!("{}%", pct)
                                    />
                                </div>
                            }
                        })}
                    </div>
                }
            })}
        </div>
    }
}
