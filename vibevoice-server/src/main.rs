use clap::Parser;
use std::sync::{Arc, mpsc as std_mpsc};
use tower_http::cors::{Any, CorsLayer};
use tracing::{info, warn};
use vibevoice::ModelVariant;

use vibevoice_server::{self, AppState, Args, Config, WorkerRequest, router, spawn_worker_thread};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    // Load config from file or use defaults
    let config = if let Some(ref config_path) = args.config {
        info!("Loading config from: {}", config_path.display());
        Config::from_file(config_path)?
    } else {
        warn!("No config file specified, using defaults");
        Config::default()
    };

    if let Some(ref dir) = config.voices_dir {
        info!("Voices directory: {}", dir.display());
    }
    if let Some(ref dir) = config.samples_dir {
        info!("Samples directory: {}", dir.display());
    }
    if let Some(ref dir) = config.output_dir {
        info!("Output directory: {}", dir.display());
    }
    if let Some(ref dir) = config.web_dir {
        info!("Web directory: {}", dir.display());
    }

    // Resolve host/port: CLI args override config, which overrides defaults
    let host = args.host.clone();
    let port = args.port.unwrap_or_else(|| config.port.unwrap_or(3908));

    info!("Starting VibeVoice server on {}:{}", host, port);
    info!("Model: {:?}", args.model);

    // Create channel for worker communication
    let (worker_tx, worker_rx) = std_mpsc::channel::<WorkerRequest>();

    // Spawn worker thread (owns VibeVoice, not Send/Sync)
    let _worker_handle = spawn_worker_thread(args.model.into(), worker_rx);

    // Create app state
    let default_model: ModelVariant = args.model.into();
    let config = Arc::new(config);
    let has_web_dir = config.web_dir.is_some();
    let state = AppState {
        worker_tx,
        config,
        default_model,
    };

    // Build router
    let mut app = router(state);

    // Add CORS if requested
    if args.cors {
        info!("CORS enabled for all origins");
        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any);
        app = app.layer(cors);
    }

    // Start server
    let addr: std::net::SocketAddr = format!("{}:{}", host, port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;

    info!("Server listening on http://{}", addr);
    info!("Endpoints:");
    info!("  GET  /health            - Health check");
    info!("  GET  /voices            - List available voices");
    info!("  POST /synthesize        - Returns WAV audio");
    info!("  POST /synthesize/json   - Returns JSON with base64 audio");
    info!("  POST /synthesize/stream - SSE streaming with progress");
    if has_web_dir {
        info!("  GET  /*                 - Static files (frontend)");
    }

    axum::serve(listener, app).await?;

    Ok(())
}
