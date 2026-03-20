pub mod agent_pane;
pub mod amendment_tracker;
pub mod feed;
pub mod help_overlay;
pub mod persuasion_graph;
pub mod progress_bar;
pub mod sparkline_widget;
pub mod vote_tracker;

use ratatui::style::Color;

/// Data needed to render a single agent pane.
#[derive(Debug, Clone)]
pub struct AgentPaneData {
    pub name: String,
    pub subtitle: Option<String>,
    pub tokens: String,
    pub last_response: String,
    pub active: bool,
    pub selected: bool,
    pub latency_ms: u32,
    pub sentiment: Option<f64>,
    pub sentiment_history: Option<Vec<f64>>,
    pub vote: Option<String>,
}

/// A single entry in the message/discussion feed.
#[derive(Debug, Clone)]
pub struct FeedEntryData {
    pub tick_or_index: u32,
    pub agent_name: String,
    pub party: Option<String>,
    pub content: String,
    pub entry_type: FeedEntryType,
    pub timestamp: String,
}

#[derive(Debug, Clone)]
pub enum FeedEntryType {
    Speech,
    Vote,
    System,
    ModelResponse,
    FinalAnswer,
    Lobby,
    Filibuster,
    Amendment,
    DirectAddress,
}

/// Data for the vote tracker panel.
#[derive(Debug, Clone)]
pub struct VoteTrackerData {
    pub yea: u32,
    pub nay: u32,
    pub abstain: u32,
    pub total_agents: usize,
    pub per_agent_votes: Vec<(String, String, String)>,
    pub persuasion_edges: Vec<(String, String, f64)>,
    pub result: Option<String>,
    pub confidence: Option<f64>,
    pub vote_breakdown: Option<Vec<(String, f64)>>,
}

/// Data for an amendment entry.
#[derive(Debug, Clone)]
pub struct AmendmentData {
    pub id: u32,
    pub proposer: String,
    pub text: String,
    pub status: String,
    pub yea: u32,
    pub nay: u32,
}
