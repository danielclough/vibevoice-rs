//! Embedded Axum server management.

use std::net::SocketAddr;
use std::sync::{mpsc as std_mpsc, Arc};
use std::thread::{self, JoinHandle};
use tokio::sync::oneshot;
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info};
use vibevoice::ModelVariant;
use vibevoice_server::{router, spawn_worker_thread, AppState, Config};

/// Handle to the embedded server, managing lifecycle.
pub struct EmbeddedServer {
    /// Thread handle for the tokio runtime running the server
    server_thread: Option<JoinHandle<()>>,
    /// Thread handle for the TTS worker
    worker_thread: Option<JoinHandle<()>>,
    /// Channel to signal server shutdown
    shutdown_tx: Option<oneshot::Sender<()>>,
    /// Server port
    port: u16,
}

impl EmbeddedServer {
    /// Start the embedded server with the given configuration.
    ///
    /// This spawns:
    /// 1. A worker thread for TTS inference (owns non-Send VibeVoice model)
    /// 2. A tokio runtime thread for the Axum HTTP server
    ///
    /// Returns once the server is ready to accept connections.
    pub fn start(config: Config, model_variant: ModelVariant) -> anyhow::Result<Self> {
        let port = config.port.unwrap_or(3000);
        let host = config.host.clone().unwrap_or_else(|| "127.0.0.1".to_string());
        let addr: SocketAddr = format!("{}:{}", host, port).parse()?;

        // Channel for worker communication
        let (worker_tx, worker_rx) = std_mpsc::channel();

        // Spawn worker thread (owns VibeVoice model)
        info!(
            "Starting TTS worker thread with model {:?}...",
            model_variant
        );
        let worker_thread = spawn_worker_thread(model_variant, worker_rx);

        // Create app state
        let config = Arc::new(config);
        let state = AppState {
            worker_tx,
            config,
            default_model: model_variant,
        };

        // Channel to signal server is ready
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();

        // Channel for shutdown signal
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

        // Spawn server thread with its own tokio runtime
        let server_thread = thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("Failed to create tokio runtime");

            rt.block_on(async move {
                // Build router with CORS for local development
                let cors = CorsLayer::new()
                    .allow_origin(Any)
                    .allow_methods(Any)
                    .allow_headers(Any);

                let app = router(state).layer(cors);

                // Bind listener
                let listener = match tokio::net::TcpListener::bind(addr).await {
                    Ok(l) => l,
                    Err(e) => {
                        error!("Failed to bind server to {}: {}", addr, e);
                        let _ = ready_tx.send(Err(e.to_string()));
                        return;
                    }
                };

                info!("Embedded server listening on http://{}", addr);
                let _ = ready_tx.send(Ok(()));

                // Serve with graceful shutdown
                axum::serve(listener, app)
                    .with_graceful_shutdown(async move {
                        let _ = shutdown_rx.await;
                        info!("Server received shutdown signal");
                    })
                    .await
                    .unwrap_or_else(|e| error!("Server error: {}", e));

                info!("Server shut down");
            });
        });

        // Wait for server to be ready
        match ready_rx.recv() {
            Ok(Ok(())) => {
                info!("Embedded server ready on port {}", port);
            }
            Ok(Err(e)) => {
                return Err(anyhow::anyhow!("Server failed to start: {}", e));
            }
            Err(_) => {
                return Err(anyhow::anyhow!(
                    "Server thread died before becoming ready"
                ));
            }
        }

        Ok(Self {
            server_thread: Some(server_thread),
            worker_thread: Some(worker_thread),
            shutdown_tx: Some(shutdown_tx),
            port,
        })
    }

    /// Get the server URL.
    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Initiate graceful shutdown.
    pub fn shutdown(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            info!("Initiating server shutdown...");
            let _ = tx.send(());
        }
    }
}

impl Drop for EmbeddedServer {
    fn drop(&mut self) {
        self.shutdown();

        // Wait for threads to finish
        if let Some(handle) = self.server_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.worker_thread.take() {
            // Worker thread will exit when channel is dropped
            let _ = handle.join();
        }

        info!("Embedded server stopped");
    }
}
