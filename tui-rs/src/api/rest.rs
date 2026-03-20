use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub size: i64,
    pub weight: f64,
    #[serde(default = "default_backend")]
    pub backend: String,
}

fn default_backend() -> String {
    "ollama".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsResponse {
    pub total_runs: Option<i64>,
    pub avg_confidence: Option<f64>,
    pub avg_latency_ms: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Personality {
    pub name: String,
    pub system_prompt: String,
}

pub struct ApiClient {
    base_url: String,
    client: reqwest::Client,
}

impl ApiClient {
    pub fn new(base_url: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("failed to build HTTP client");
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client,
        }
    }

    pub async fn health_check(&self) -> Result<bool, reqwest::Error> {
        let resp = self.client.get(format!("{}/health", self.base_url)).send().await?;
        Ok(resp.status().is_success())
    }

    pub async fn list_models(&self) -> Result<Vec<ModelInfo>, reqwest::Error> {
        self.client
            .get(format!("{}/api/models", self.base_url))
            .send()
            .await?
            .json()
            .await
    }

    pub async fn get_stats(&self) -> Result<StatsResponse, reqwest::Error> {
        self.client
            .get(format!("{}/api/enhanced/stats", self.base_url))
            .send()
            .await?
            .json()
            .await
    }

    pub async fn list_personality_sets(&self) -> Result<Vec<String>, reqwest::Error> {
        self.client
            .get(format!("{}/api/personality-lists", self.base_url))
            .send()
            .await?
            .json()
            .await
    }

    pub async fn get_personality_list(&self, name: &str) -> Result<Vec<Personality>, reqwest::Error> {
        self.client
            .get(format!("{}/api/personality-list/{}", self.base_url, name))
            .send()
            .await?
            .json()
            .await
    }
}
