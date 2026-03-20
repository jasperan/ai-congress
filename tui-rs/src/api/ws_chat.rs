use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

#[derive(Debug, Serialize)]
pub struct WsChatRequest {
    pub prompt: String,
    pub models: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voting_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub personalities: Option<Vec<PersonalityRef>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub history: Option<Vec<HistoryEntry>>,
}

#[derive(Debug, Serialize)]
pub struct PersonalityRef {
    pub name: String,
    pub system_prompt: String,
}

#[derive(Debug, Serialize)]
pub struct HistoryEntry {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct WsChatEvent {
    #[serde(rename = "type", default)]
    pub event_type: String,
    pub message: Option<String>,
    pub name: Option<String>,
    pub model: Option<String>,
    pub content: Option<String>,
    pub status: Option<String>,
    pub response: Option<String>,
    pub confidence: Option<f64>,
    pub semantic_confidence: Option<f64>,
    pub vote_breakdown: Option<serde_json::Value>,
    pub semantic_vote: Option<serde_json::Value>,
    pub personalities: Option<Vec<serde_json::Value>>,
    pub data: Option<serde_json::Value>,
}

pub struct WsChatClient {
    write: SplitSink<WsStream, Message>,
    read: SplitStream<WsStream>,
}

impl WsChatClient {
    pub async fn connect(base_url: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let ws_url = base_url
            .replace("http://", "ws://")
            .replace("https://", "wss://");
        let url = format!("{}/ws/chat", ws_url.trim_end_matches('/'));
        let (stream, _) = connect_async(&url).await?;
        let (write, read) = stream.split();
        Ok(Self { write, read })
    }

    pub async fn send_chat(&mut self, req: WsChatRequest) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string(&req)?;
        self.write.send(Message::Text(json.into())).await?;
        Ok(())
    }

    pub async fn next_event(&mut self) -> Option<WsChatEvent> {
        loop {
            match self.read.next().await {
                Some(Ok(Message::Text(text))) => {
                    match serde_json::from_str(&text) {
                        Ok(event) => return Some(event),
                        Err(_) => continue,
                    }
                }
                Some(Ok(Message::Ping(_) | Message::Pong(_) | Message::Binary(_))) => continue,
                Some(Ok(Message::Close(_))) | Some(Err(_)) | None => return None,
                _ => continue,
            }
        }
    }

    pub async fn close(mut self) {
        let _ = self.write.send(Message::Close(None)).await;
    }
}
