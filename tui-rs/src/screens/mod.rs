pub mod chat_dashboard;
pub mod mode_select;
pub mod models;
pub mod results;
pub mod simulation;
pub mod splash;

use ratatui::layout::Rect;
use ratatui::Frame;

/// Action returned by a screen's update method to tell the router what to do.
#[derive(Debug)]
pub enum Action {
    None,
    SwitchScreen(ScreenId),
    Quit,
    Retry,
}

#[derive(Debug, Clone)]
pub enum ScreenId {
    Splash,
    Models,
    ModeSelect { selected_models: Vec<String> },
    ChatDashboard(ChatLaunchConfig),
    Simulation(SimLaunchConfig),
    Results(SessionResult),
}

#[derive(Debug, Clone)]
pub struct ChatLaunchConfig {
    pub prompt: String,
    pub models: Vec<String>,
    pub mode: String,
    pub temperature: f64,
    pub voting_mode: String,
    pub inference_backend: String,
}

#[derive(Debug, Clone)]
pub struct SimLaunchConfig {
    pub topic: String,
    pub agents: u32,
    pub ticks: u32,
    pub model: String,
}

#[derive(Debug, Clone)]
pub enum SessionResult {
    Chat {
        final_answer: String,
        confidence: f64,
        vote_breakdown: serde_json::Value,
        responses: Vec<(String, String)>,
    },
    Simulation {
        result: String,
        yea: u32,
        nay: u32,
        abstain: u32,
        amendments: Vec<serde_json::Value>,
        historical_accuracy: Option<f64>,
    },
}
