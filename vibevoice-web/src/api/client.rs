use gloo_net::http::Request;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoicesResponse {
    pub voices: Vec<String>,
    pub samples: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesizeJsonResponse {
    pub audio_base64: String,
    pub duration_secs: f32,
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct SynthesizeRequest {
    pub text: String,
    pub voice: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ApiError {
    pub message: String,
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl From<gloo_net::Error> for ApiError {
    fn from(err: gloo_net::Error) -> Self {
        ApiError {
            message: err.to_string(),
        }
    }
}

pub async fn fetch_voices(server_url: &str) -> Result<VoicesResponse, ApiError> {
    let url = format!("{}/voices", server_url.trim_end_matches('/'));
    let response = Request::get(&url)
        .send()
        .await?;

    if !response.ok() {
        return Err(ApiError {
            message: format!("Failed to fetch voices: {}", response.status()),
        });
    }

    response.json().await.map_err(|e| ApiError {
        message: format!("Failed to parse voices response: {}", e),
    })
}

pub async fn synthesize_json(
    server_url: &str,
    text: &str,
    voice: &str,
    model: Option<&str>,
) -> Result<SynthesizeJsonResponse, ApiError> {
    let url = format!("{}/synthesize/json", server_url.trim_end_matches('/'));
    let request = SynthesizeRequest {
        text: text.to_string(),
        voice: voice.to_string(),
        model: model.map(|s| s.to_string()),
    };

    let response = Request::post(&url)
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(&request).map_err(|e| ApiError {
            message: format!("Failed to serialize request: {}", e),
        })?)?
        .send()
        .await?;

    if !response.ok() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(ApiError {
            message: format!("Synthesis failed ({}): {}", status, body),
        });
    }

    response.json().await.map_err(|e| ApiError {
        message: format!("Failed to parse synthesis response: {}", e),
    })
}

pub async fn check_health(server_url: &str) -> Result<bool, ApiError> {
    let url = format!("{}/health", server_url.trim_end_matches('/'));
    let response = Request::get(&url)
        .send()
        .await?;

    Ok(response.ok())
}
