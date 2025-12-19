use base64::Engine;
use js_sys::{Object, Reflect, Uint8Array};
use serde::Deserialize;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{ReadableStreamDefaultReader, Request, RequestInit, RequestMode, Response};

use crate::api::SynthesizeRequest;

#[derive(Debug, Clone, Deserialize)]
pub struct HeaderEvent {
    pub wav_header: String,
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChunkEvent {
    pub step: usize,
    pub pcm_chunk: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompleteEvent {
    pub duration_secs: f32,
    pub total_steps: usize,
    pub total_pcm_bytes: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ErrorEvent {
    pub error: String,
}

#[derive(Debug, Clone)]
pub enum SseEvent {
    Header(HeaderEvent),
    Chunk(ChunkEvent),
    Complete(CompleteEvent),
    Error(ErrorEvent),
}

pub struct StreamingState {
    pub wav_header: Option<Vec<u8>>,
    pub pcm_chunks: Vec<Vec<u8>>,
    pub current_step: usize,
    pub total_steps: Option<usize>,
    pub is_complete: bool,
    pub error: Option<String>,
}

impl Default for StreamingState {
    fn default() -> Self {
        Self {
            wav_header: None,
            pcm_chunks: Vec::new(),
            current_step: 0,
            total_steps: None,
            is_complete: false,
            error: None,
        }
    }
}

impl StreamingState {
    pub fn apply_event(&mut self, event: SseEvent) {
        match event {
            SseEvent::Header(h) => {
                if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(&h.wav_header) {
                    self.wav_header = Some(bytes);
                }
            }
            SseEvent::Chunk(c) => {
                self.current_step = c.step;
                if let Some(pcm_b64) = c.pcm_chunk {
                    if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(&pcm_b64) {
                        self.pcm_chunks.push(bytes);
                    }
                }
            }
            SseEvent::Complete(c) => {
                self.total_steps = Some(c.total_steps);
                self.is_complete = true;
            }
            SseEvent::Error(e) => {
                self.error = Some(e.error);
                self.is_complete = true;
            }
        }
    }

    pub fn to_wav_bytes(&self) -> Option<Vec<u8>> {
        let header = self.wav_header.as_ref()?;
        let total_pcm: usize = self.pcm_chunks.iter().map(|c| c.len()).sum();
        let mut wav = Vec::with_capacity(header.len() + total_pcm);
        wav.extend_from_slice(header);
        for chunk in &self.pcm_chunks {
            wav.extend_from_slice(chunk);
        }
        // Fix the data size in the WAV header
        if wav.len() >= 44 {
            let data_size = (wav.len() - 44) as u32;
            wav[40..44].copy_from_slice(&data_size.to_le_bytes());
            let file_size = (wav.len() - 8) as u32;
            wav[4..8].copy_from_slice(&file_size.to_le_bytes());
        }
        Some(wav)
    }
}

fn parse_sse_event(event_type: &str, data: &str) -> Option<SseEvent> {
    match event_type {
        "header" => serde_json::from_str::<HeaderEvent>(data)
            .ok()
            .map(SseEvent::Header),
        "chunk" => serde_json::from_str::<ChunkEvent>(data)
            .ok()
            .map(SseEvent::Chunk),
        "complete" => serde_json::from_str::<CompleteEvent>(data)
            .ok()
            .map(SseEvent::Complete),
        "error" => serde_json::from_str::<ErrorEvent>(data)
            .ok()
            .map(SseEvent::Error),
        _ => None,
    }
}

fn parse_sse_buffer(buffer: &str) -> (Vec<SseEvent>, String) {
    let mut events = Vec::new();
    let mut remaining = String::new();
    let mut current_event_type = String::new();
    let mut current_data = String::new();

    for line in buffer.split('\n') {
        if line.starts_with("event:") {
            current_event_type = line[6..].trim().to_string();
        } else if line.starts_with("data:") {
            current_data = line[5..].trim().to_string();
        } else if line.is_empty() && !current_event_type.is_empty() {
            if let Some(event) = parse_sse_event(&current_event_type, &current_data) {
                events.push(event);
            }
            current_event_type.clear();
            current_data.clear();
        }
    }

    // Keep unparsed data for next iteration
    if !current_event_type.is_empty() || !current_data.is_empty() {
        if !current_event_type.is_empty() {
            remaining.push_str("event:");
            remaining.push_str(&current_event_type);
            remaining.push('\n');
        }
        if !current_data.is_empty() {
            remaining.push_str("data:");
            remaining.push_str(&current_data);
            remaining.push('\n');
        }
    }

    (events, remaining)
}

pub async fn start_streaming<F>(
    server_url: &str,
    text: &str,
    voice: &str,
    model: Option<&str>,
    mut on_event: F,
) -> Result<(), String>
where
    F: FnMut(SseEvent),
{
    let url = format!("{}/synthesize/stream", server_url.trim_end_matches('/'));
    let request_body = SynthesizeRequest {
        text: text.to_string(),
        voice: voice.to_string(),
        model: model.map(|s| s.to_string()),
    };
    let body_json =
        serde_json::to_string(&request_body).map_err(|e| format!("Serialize error: {}", e))?;

    let mut opts = RequestInit::new();
    opts.method("POST");
    opts.mode(RequestMode::Cors);
    opts.body(Some(&JsValue::from_str(&body_json)));

    let request =
        Request::new_with_str_and_init(&url, &opts).map_err(|e| format!("Request error: {:?}", e))?;
    request
        .headers()
        .set("Content-Type", "application/json")
        .map_err(|e| format!("Header error: {:?}", e))?;

    let window = web_sys::window().ok_or("No window")?;
    let response: Response = JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(|e| format!("Fetch error: {:?}", e))?
        .dyn_into()
        .map_err(|_| "Response cast error")?;

    if !response.ok() {
        return Err(format!("Server error: {}", response.status()));
    }

    let body = response.body().ok_or("No response body")?;
    let reader: ReadableStreamDefaultReader = body
        .get_reader()
        .dyn_into()
        .map_err(|_| "Reader cast error")?;

    let decoder = web_sys::TextDecoder::new().map_err(|e| format!("TextDecoder error: {:?}", e))?;

    let mut buffer = String::new();

    loop {
        let result = JsFuture::from(reader.read())
            .await
            .map_err(|e| format!("Read error: {:?}", e))?;

        let done = Reflect::get(&result, &JsValue::from_str("done"))
            .map_err(|_| "Get done error")?
            .as_bool()
            .unwrap_or(true);

        if done {
            break;
        }

        let value = Reflect::get(&result, &JsValue::from_str("value"))
            .map_err(|_| "Get value error")?;

        if !value.is_undefined() {
            let uint8_array: Uint8Array = value.dyn_into().map_err(|_| "Uint8Array cast error")?;
            let text = decoder
                .decode_with_buffer_source(&uint8_array)
                .map_err(|e| format!("Decode error: {:?}", e))?;

            buffer.push_str(&text);

            let (events, remaining) = parse_sse_buffer(&buffer);
            buffer = remaining;

            for event in events {
                on_event(event);
            }
        }
    }

    // Parse any remaining buffer
    if !buffer.is_empty() {
        let (events, _) = parse_sse_buffer(&buffer);
        for event in events {
            on_event(event);
        }
    }

    Ok(())
}

pub fn pcm_to_float32(pcm_bytes: &[u8]) -> Vec<f32> {
    pcm_bytes
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect()
}
