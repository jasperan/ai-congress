/// Session tab management (future feature).
/// Each session holds independent screen state and WebSocket connections.
#[allow(dead_code)]
pub enum Session {
    Chat { label: String },
    Simulation { label: String },
}
