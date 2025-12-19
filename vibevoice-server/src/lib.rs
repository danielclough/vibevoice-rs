//! VibeVoice HTTP Server with SSE streaming support.

use axum::{
    Json, Router, extract::State, http::{StatusCode, header}, response::{
        IntoResponse, Response, sse::{Event, Sse}
    }, routing::{get, post}
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use clap::{Parser, ValueEnum};
use futures::stream::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{mpsc as std_mpsc, Arc};
use std::thread;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tower_http::services::ServeDir;
use tracing::{error, info};
use vibevoice::{AudioData, Device, ModelVariant, Progress, VibeVoice};

// =============================================================================
// Configuration
// =============================================================================

/// Server configuration loaded from YAML file.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Host/domain to bind to (default: 0.0.0.0)
    #[serde(default)]
    pub host: Option<String>,

    /// Port to listen on (default: 3000)
    #[serde(default)]
    pub port: Option<u16>,

    /// Directory containing voice safetensors files (for realtime model)
    #[serde(default)]
    pub voices_dir: Option<PathBuf>,

    /// Directory containing WAV samples for voice cloning (for batch models)
    #[serde(default)]
    pub samples_dir: Option<PathBuf>,

    /// Optional directory to save output WAV files
    #[serde(default)]
    pub output_dir: Option<PathBuf>,

    /// Directory containing the built vibevoice-web frontend to serve at /
    #[serde(default)]
    pub web_dir: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            host: None,
            port: None,
            voices_dir: None,
            samples_dir: None,
            output_dir: None,
            web_dir: None,
        }
    }
}

impl Config {
    /// Load config from YAML file.
    pub fn from_file(path: &PathBuf) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Config = serde_yaml::from_str(&contents)?;
        Ok(config)
    }

    /// Resolve a voice name to a full path based on model type.
    ///
    /// If the voice is already an absolute path, returns it as-is.
    /// For Realtime model: looks in voices_dir for .safetensors files.
    /// For Batch models: looks in samples_dir for .wav files.
    pub fn resolve_voice(&self, voice: &str, model: ModelVariant) -> String {
        let path = PathBuf::from(voice);

        // If it's already an absolute path, use it directly
        if path.is_absolute() {
            return voice.to_string();
        }

        match model {
            ModelVariant::Realtime => {
                // Try voices_dir for safetensors
                if let Some(ref voices_dir) = self.voices_dir {
                    let with_ext = if voice.ends_with(".safetensors") {
                        voices_dir.join(voice)
                    } else {
                        voices_dir.join(format!("{}.safetensors", voice))
                    };
                    if with_ext.exists() {
                        return with_ext.to_string_lossy().to_string();
                    }
                    let exact = voices_dir.join(voice);
                    if exact.exists() {
                        return exact.to_string_lossy().to_string();
                    }
                }
            }
            ModelVariant::Batch1_5B | ModelVariant::Batch7B => {
                // Try samples_dir for wav files
                if let Some(ref samples_dir) = self.samples_dir {
                    let with_ext = if voice.ends_with(".wav") {
                        samples_dir.join(voice)
                    } else {
                        samples_dir.join(format!("{}.wav", voice))
                    };
                    if with_ext.exists() {
                        return with_ext.to_string_lossy().to_string();
                    }
                    let exact = samples_dir.join(voice);
                    if exact.exists() {
                        return exact.to_string_lossy().to_string();
                    }
                }
            }
        }

        // Fall back to original (let the model handle the error)
        voice.to_string()
    }
}

// =============================================================================
// CLI Arguments
// =============================================================================

#[derive(Parser, Debug)]
#[command(name = "vibevoice-server")]
#[command(about = "VibeVoice TTS HTTP Server with SSE streaming")]
pub struct Args {
    /// Path to YAML config file
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    /// Port to listen on (overrides config file)
    #[arg(short, long)]
    pub port: Option<u16>,

    /// Model variant to use
    #[arg(short, long, value_enum, default_value = "realtime")]
    pub model: ModelArg,

    /// Enable CORS for all origins (enabled by default, use --no-cors to disable)
    #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
    pub cors: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum, Deserialize)]
pub enum ModelArg {
    #[value(name = "1.5B")]
    #[serde(rename = "1.5B")]
    Batch1_5B,
    #[value(name = "7B")]
    #[serde(rename = "7B")]
    Batch7B,
    #[value(name = "realtime")]
    #[serde(rename = "realtime")]
    Realtime,
}

impl From<ModelArg> for ModelVariant {
    fn from(arg: ModelArg) -> Self {
        match arg {
            ModelArg::Batch1_5B => ModelVariant::Batch1_5B,
            ModelArg::Batch7B => ModelVariant::Batch7B,
            ModelArg::Realtime => ModelVariant::Realtime,
        }
    }
}

// =============================================================================
// Request/Response Types
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct SynthesizeRequest {
    pub text: String,
    pub voice: String,
    /// Optional model variant for this request (defaults to server's --model flag)
    #[serde(default)]
    pub model: Option<ModelArg>,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
}

#[derive(Debug, Serialize)]
pub struct VoicesResponse {
    /// Available voice safetensors (realtime model)
    pub voices: Vec<String>,
    /// Available WAV samples (batch model)
    pub samples: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct SynthesizeJsonResponse {
    pub audio_base64: String,
    pub duration_secs: f32,
    pub sample_rate: u32,
}

/// Sent first in SSE stream - contains WAV header (44 bytes)
#[derive(Debug, Serialize)]
pub struct HeaderEvent {
    /// Base64-encoded WAV header (44 bytes)
    pub wav_header: String,
    pub sample_rate: u32,
}

/// Sent for each audio chunk - contains raw PCM (no header)
#[derive(Debug, Serialize)]
pub struct ChunkEvent {
    pub step: usize,
    /// Base64-encoded raw PCM bytes (16-bit signed int, little-endian)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pcm_chunk: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CompleteEvent {
    pub duration_secs: f32,
    pub total_steps: usize,
    /// Total PCM bytes sent (for client to fix WAV header sizes)
    pub total_pcm_bytes: usize,
}

#[derive(Debug, Serialize)]
pub struct ErrorEvent {
    pub error: String,
}

// =============================================================================
// Worker Thread Communication
// =============================================================================

pub enum WorkerRequest {
    /// Basic synthesis - returns complete audio
    Synthesize {
        text: String,
        voice: String,
        model: ModelVariant,
        response_tx: oneshot::Sender<WorkerResponse>,
    },
    /// Streaming synthesis - sends progress via channel
    SynthesizeStream {
        text: String,
        voice: String,
        model: ModelVariant,
        progress_tx: mpsc::Sender<StreamEvent>,
    },
}

pub enum WorkerResponse {
    Success {
        audio_bytes: Vec<u8>,
        duration_secs: f32,
        sample_rate: u32,
    },
    Error(String),
}

pub enum StreamEvent {
    /// WAV header - sent first
    Header { wav_header: Vec<u8>, sample_rate: u32 },
    /// Raw PCM chunk - sent for each step
    Chunk { step: usize, pcm_bytes: Option<Vec<u8>> },
    /// Completion - sent last
    Complete { duration_secs: f32, total_steps: usize, total_pcm_bytes: usize },
    /// Error
    Error(String),
}

// =============================================================================
// Application State
// =============================================================================

#[derive(Clone)]
pub struct AppState {
    pub worker_tx: std_mpsc::Sender<WorkerRequest>,
    pub config: Arc<Config>,
    pub default_model: ModelVariant,
}

// =============================================================================
// Worker Thread
// =============================================================================

pub fn spawn_worker_thread(
    default_variant: ModelVariant,
    worker_rx: std_mpsc::Receiver<WorkerRequest>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        use std::collections::HashMap;

        info!("Worker thread starting, loading default model {:?}...", default_variant);

        // Cache of loaded models - keyed by variant
        let mut models: HashMap<ModelVariant, VibeVoice> = HashMap::new();

        // Load default model at startup
        match VibeVoice::new(default_variant, Device::auto()) {
            Ok(vv) => {
                models.insert(default_variant, vv);
                info!("Default model {:?} loaded, worker ready for requests", default_variant);
            }
            Err(e) => {
                error!("Failed to initialize default VibeVoice model: {}", e);
                return;
            }
        };

        while let Ok(request) = worker_rx.recv() {
            match request {
                WorkerRequest::Synthesize { text, voice, model, response_tx } => {
                    info!("Processing synthesize request: {} chars, model {:?}", text.len(), model);

                    // Get or load the requested model
                    if !models.contains_key(&model) {
                        info!("Loading model {:?} on demand...", model);
                        match VibeVoice::new(model, Device::auto()) {
                            Ok(vv) => {
                                models.insert(model, vv);
                                info!("Model {:?} loaded successfully", model);
                            }
                            Err(e) => {
                                let _ = response_tx.send(WorkerResponse::Error(
                                    format!("Failed to load model {:?}: {}", model, e)
                                ));
                                continue;
                            }
                        }
                    }

                    let vv = models.get_mut(&model).unwrap();
                    let result = vv.synthesize(&text, Some(&voice));

                    let response = match result {
                        Ok(audio) => {
                            match audio.to_wav_bytes() {
                                Ok(bytes) => WorkerResponse::Success {
                                    audio_bytes: bytes,
                                    duration_secs: audio.duration_secs(),
                                    sample_rate: audio.sample_rate(),
                                },
                                Err(e) => WorkerResponse::Error(format!("WAV encoding failed: {}", e)),
                            }
                        }
                        Err(e) => WorkerResponse::Error(format!("Synthesis failed: {}", e)),
                    };

                    let _ = response_tx.send(response);
                }

                WorkerRequest::SynthesizeStream { text, voice, model, progress_tx } => {
                    info!("Processing streaming synthesize request: {} chars, model {:?}", text.len(), model);

                    // Get or load the requested model
                    if !models.contains_key(&model) {
                        info!("Loading model {:?} on demand...", model);
                        match VibeVoice::new(model, Device::auto()) {
                            Ok(vv) => {
                                models.insert(model, vv);
                                info!("Model {:?} loaded successfully", model);
                            }
                            Err(e) => {
                                let _ = progress_tx.blocking_send(StreamEvent::Error(
                                    format!("Failed to load model {:?}: {}", model, e)
                                ));
                                continue;
                            }
                        }
                    }

                    let vv = models.get_mut(&model).unwrap();

                    // Send WAV header first (before any audio chunks)
                    let sample_rate = 24000u32;
                    let wav_header = AudioData::wav_header_for_streaming(sample_rate, None);
                    let _ = progress_tx.blocking_send(StreamEvent::Header {
                        wav_header,
                        sample_rate,
                    });

                    let progress_tx_clone = progress_tx.clone();
                    let mut total_steps = 0;
                    let total_pcm_bytes = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
                    let total_pcm_bytes_clone = total_pcm_bytes.clone();

                    let result = vv.synthesize_with_callback(
                        &text,
                        Some(&voice),
                        Some(Box::new(move |progress: Progress| {
                            total_steps = progress.step;

                            // Convert to raw PCM (no WAV header)
                            let pcm_bytes = match progress.audio_chunk {
                                Some(chunk) => {
                                    let bytes = chunk.to_pcm_bytes();
                                    total_pcm_bytes_clone.fetch_add(bytes.len(), std::sync::atomic::Ordering::Relaxed);
                                    Some(bytes)
                                }
                                None => {
                                    tracing::warn!("Chunk {} has no audio data", progress.step);
                                    None
                                }
                            };

                            let event = StreamEvent::Chunk {
                                step: progress.step,
                                pcm_bytes,
                            };

                            // Use blocking send since we're in a sync callback
                            if let Err(e) = progress_tx_clone.blocking_send(event) {
                                tracing::error!("Failed to send chunk {}: {}", progress.step, e);
                            }
                        })),
                    );

                    match result {
                        Ok(audio) => {
                            let _ = progress_tx.blocking_send(StreamEvent::Complete {
                                duration_secs: audio.duration_secs(),
                                total_steps,
                                total_pcm_bytes: total_pcm_bytes.load(std::sync::atomic::Ordering::Relaxed),
                            });
                        }
                        Err(e) => {
                            let _ = progress_tx.blocking_send(StreamEvent::Error(
                                format!("Synthesis failed: {}", e)
                            ));
                        }
                    }
                }
            }
        }

        info!("Worker thread shutting down");
    })
}

// =============================================================================
// HTTP Handlers
// =============================================================================

pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

pub async fn list_voices(State(state): State<AppState>) -> Json<VoicesResponse> {
    let mut voices = Vec::new();
    let mut samples = Vec::new();

    // List voices from voices_dir
    if let Some(ref dir) = state.config.voices_dir {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "safetensors") {
                    if let Some(stem) = path.file_stem() {
                        voices.push(stem.to_string_lossy().to_string());
                    }
                }
            }
        }
    }

    // List samples from samples_dir
    if let Some(ref dir) = state.config.samples_dir {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "wav") {
                    if let Some(stem) = path.file_stem() {
                        samples.push(stem.to_string_lossy().to_string());
                    }
                }
            }
        }
    }

    voices.sort();
    samples.sort();

    Json(VoicesResponse { voices, samples })
}

pub async fn synthesize(
    State(state): State<AppState>,
    Json(req): Json<SynthesizeRequest>,
) -> Response {
    let (response_tx, response_rx) = oneshot::channel();

    // Use request model if specified, otherwise use server default
    let model = req.model.map(Into::into).unwrap_or(state.default_model);

    // Resolve voice path from config (model-aware)
    let voice = state.config.resolve_voice(&req.voice, model);

    let request = WorkerRequest::Synthesize {
        text: req.text,
        voice,
        model,
        response_tx,
    };

    if state.worker_tx.send(request).is_err() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorEvent { error: "Worker unavailable".to_string() }),
        ).into_response();
    }

    match response_rx.await {
        Ok(WorkerResponse::Success { audio_bytes, .. }) => {
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "audio/wav")],
                audio_bytes,
            ).into_response()
        }
        Ok(WorkerResponse::Error(e)) => {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorEvent { error: e }),
            ).into_response()
        }
        Err(_) => {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorEvent { error: "Worker communication failed".to_string() }),
            ).into_response()
        }
    }
}

pub async fn synthesize_json(
    State(state): State<AppState>,
    Json(req): Json<SynthesizeRequest>,
) -> Response {
    let (response_tx, response_rx) = oneshot::channel();

    // Use request model if specified, otherwise use server default
    let model = req.model.map(Into::into).unwrap_or(state.default_model);

    // Resolve voice path from config (model-aware)
    let voice = state.config.resolve_voice(&req.voice, model);

    let request = WorkerRequest::Synthesize {
        text: req.text,
        voice,
        model,
        response_tx,
    };

    if state.worker_tx.send(request).is_err() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorEvent { error: "Worker unavailable".to_string() }),
        ).into_response();
    }

    match response_rx.await {
        Ok(WorkerResponse::Success { audio_bytes, duration_secs, sample_rate }) => {
            Json(SynthesizeJsonResponse {
                audio_base64: BASE64.encode(&audio_bytes),
                duration_secs,
                sample_rate,
            }).into_response()
        }
        Ok(WorkerResponse::Error(e)) => {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorEvent { error: e }),
            ).into_response()
        }
        Err(_) => {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorEvent { error: "Worker communication failed".to_string() }),
            ).into_response()
        }
    }
}

pub async fn synthesize_stream(
    State(state): State<AppState>,
    Json(req): Json<SynthesizeRequest>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (progress_tx, progress_rx) = mpsc::channel::<StreamEvent>(32);

    // Use request model if specified, otherwise use server default
    let model = req.model.map(Into::into).unwrap_or(state.default_model);

    // Resolve voice path from config (model-aware)
    let voice = state.config.resolve_voice(&req.voice, model);

    let request = WorkerRequest::SynthesizeStream {
        text: req.text,
        voice,
        model,
        progress_tx,
    };

    // Send request to worker (fire and forget, errors handled via stream)
    let _ = state.worker_tx.send(request);

    let stream = ReceiverStream::new(progress_rx).map(|event| {
        let sse_event = match event {
            StreamEvent::Header { wav_header, sample_rate } => {
                let data = HeaderEvent {
                    wav_header: BASE64.encode(&wav_header),
                    sample_rate,
                };
                Event::default()
                    .event("header")
                    .json_data(data)
                    .unwrap_or_else(|_| Event::default().event("error").data("JSON encoding failed"))
            }
            StreamEvent::Chunk { step, pcm_bytes } => {
                let data = ChunkEvent {
                    step,
                    pcm_chunk: pcm_bytes.map(|bytes| BASE64.encode(&bytes)),
                };
                Event::default()
                    .event("chunk")
                    .json_data(data)
                    .unwrap_or_else(|_| Event::default().event("error").data("JSON encoding failed"))
            }
            StreamEvent::Complete { duration_secs, total_steps, total_pcm_bytes } => {
                let data = CompleteEvent { duration_secs, total_steps, total_pcm_bytes };
                Event::default()
                    .event("complete")
                    .json_data(data)
                    .unwrap_or_else(|_| Event::default().event("error").data("JSON encoding failed"))
            }
            StreamEvent::Error(msg) => {
                let data = ErrorEvent { error: msg };
                Event::default()
                    .event("error")
                    .json_data(data)
                    .unwrap_or_else(|_| Event::default().event("error").data("JSON encoding failed"))
            }
        };
        Ok(sse_event)
    });

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("keep-alive"),
    )
}

pub fn router(state: AppState) -> Router {
    let mut router = Router::new()
        .route("/health", get(health))
        .route("/voices", get(list_voices))
        .route("/synthesize", post(synthesize))
        .route("/synthesize/json", post(synthesize_json))
        .route("/synthesize/stream", post(synthesize_stream));

    // Serve static files from web_dir if configured
    if let Some(ref web_dir) = state.config.web_dir {
        info!("Serving static files from: {}", web_dir.display());
        router = router.fallback_service(ServeDir::new(web_dir));
    }

    router.with_state(state)
}