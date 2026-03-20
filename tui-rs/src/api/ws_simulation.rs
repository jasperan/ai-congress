use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use serde::Serialize;
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

#[derive(Debug, Serialize)]
struct SimConfig {
    topic: String,
    num_agents: u32,
    num_ticks: u32,
    model: String,
}

pub struct WsSimulationClient {
    write: SplitSink<WsStream, Message>,
    read: SplitStream<WsStream>,
}

impl WsSimulationClient {
    pub async fn connect(url: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let (stream, _) = connect_async(url).await?;
        let (write, read) = stream.split();
        Ok(Self { write, read })
    }

    pub async fn send_config(
        &mut self,
        topic: &str,
        agents: u32,
        ticks: u32,
        model: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = SimConfig {
            topic: topic.to_string(),
            num_agents: agents,
            num_ticks: ticks,
            model: model.to_string(),
        };
        let json = serde_json::to_string(&config)?;
        self.write.send(Message::Text(json.into())).await?;
        Ok(())
    }

    pub async fn next_event(&mut self) -> Option<serde_json::Value> {
        loop {
            match self.read.next().await {
                Some(Ok(Message::Text(text))) => {
                    match serde_json::from_str(&text) {
                        Ok(val) => return Some(val),
                        Err(_) => continue,
                    }
                }
                Some(Ok(Message::Ping(_) | Message::Pong(_) | Message::Binary(_))) => continue,
                Some(Ok(Message::Close(_))) | Some(Err(_)) | None => return None,
                _ => continue,
            }
        }
    }
}
