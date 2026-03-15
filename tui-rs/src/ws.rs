use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

pub struct WsClient {
    write: SplitSink<WsStream, Message>,
    read: SplitStream<WsStream>,
}

impl WsClient {
    /// Connect to the WebSocket server at the given URL.
    pub async fn connect(url: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let (ws_stream, _response) = connect_async(url).await?;
        let (write, read) = ws_stream.split();
        Ok(Self { write, read })
    }

    /// Send the simulation configuration to the server.
    pub async fn send_config(
        &mut self,
        topic: &str,
        agents: u32,
        ticks: u32,
        model: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = serde_json::json!({
            "topic": topic,
            "num_agents": agents,
            "num_ticks": ticks,
            "model": model,
        });
        let msg = Message::Text(config.to_string().into());
        self.write.send(msg).await?;
        Ok(())
    }

    /// Read the next JSON event from the WebSocket.
    /// Returns None if the connection is closed.
    pub async fn next_event(&mut self) -> Option<serde_json::Value> {
        loop {
            match self.read.next().await {
                Some(Ok(Message::Text(text))) => {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&text) {
                        return Some(val);
                    }
                    // Skip non-JSON text messages
                }
                Some(Ok(Message::Ping(_))) | Some(Ok(Message::Pong(_))) => {
                    // Ignore ping/pong
                    continue;
                }
                Some(Ok(Message::Close(_))) | None => {
                    return None;
                }
                Some(Ok(_)) => {
                    // Binary or other message types
                    continue;
                }
                Some(Err(_)) => {
                    return None;
                }
            }
        }
    }
}
