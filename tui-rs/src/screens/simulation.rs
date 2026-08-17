use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::theme;

use super::sim_phases::centered_rect;
use super::sim_sentiment::{make_progress_bar, progress_color, render_sparkline, truncate_str};

// ── Supporting Types (copied from app.rs) ───────────────────────────────────

#[derive(Clone, Debug)]
pub struct AgentInfo {
    pub name: String,
    pub party: String,
    pub state: String,
    pub chamber: String,
}

#[derive(Clone, Debug)]
pub struct AgentStream {
    pub name: String,
    pub tokens: String,
    pub active: bool,
    pub latency_ms: u32,
    pub last_response: String,
}

#[derive(Clone, Debug)]
pub struct FeedEntry {
    pub tick: u32,
    pub agent_name: String,
    pub party: String,
    pub content: String,
    pub entry_type: FeedEntryType,
    pub timestamp: String,
}

#[derive(Clone, Debug)]
pub enum FeedEntryType {
    Speech,
    Vote,
    System,
    Lobby,
    Filibuster,
    Amendment,
    DirectAddress,
}

#[derive(Clone, Debug)]
pub struct Vote {
    pub agent_name: String,
    pub party: String,
    pub vote: String,
    pub rationale: String,
}

#[derive(Clone, Debug)]
pub struct AmendmentInfo {
    pub id: u32,
    pub proposer: String,
    pub text: String,
    pub status: String,
    pub yea: u32,
    pub nay: u32,
}

#[derive(Clone, Debug)]
pub struct FilibusterInfo {
    pub agent_name: String,
    pub start_tick: u32,
    pub active: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LayoutMode {
    Focus,
    Grid,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct WsEvent {
    #[serde(rename = "type")]
    pub event_type: Option<String>,
    pub event: Option<String>,
    pub tick: Option<u32>,
    pub agent: Option<String>,
    pub agent_name: Option<String>,
    pub party: Option<String>,
    pub state: Option<String>,
    pub chamber: Option<String>,
    pub content: Option<String>,
    pub token: Option<String>,
    pub tokens: Option<String>,
    pub vote: Option<String>,
    pub rationale: Option<String>,
    pub phase: Option<String>,
    pub phase_name: Option<String>,
    pub result: Option<String>,
    pub message: Option<String>,
    pub latency_ms: Option<u32>,
    pub tokens_per_sec: Option<f64>,
    pub total_tokens: Option<u64>,
    pub agents: Option<Vec<serde_json::Value>>,
    pub max_ticks: Option<u32>,
    pub participants: Option<Vec<String>>,
    pub full_response: Option<String>,
    pub yea: Option<u32>,
    pub nay: Option<u32>,
    pub abstain: Option<u32>,
    pub timestamp: Option<String>,
    pub model: Option<String>,
    // Advanced feature fields
    pub sentiment: Option<f64>,
    pub old_score: Option<f64>,
    pub new_score: Option<f64>,
    pub drift: Option<f64>,
    pub direction: Option<String>,
    pub speaker: Option<String>,
    pub target: Option<String>,
    pub snippet: Option<String>,
    pub amendment_id: Option<u32>,
    pub amendment_text: Option<String>,
    pub new_bill_text: Option<String>,
    pub affiliation: Option<String>,
    pub bias: Option<String>,
    pub duration: Option<u32>,
    pub reason: Option<String>,
    pub influencer: Option<String>,
    pub influenced: Option<String>,
    pub strength: Option<f64>,
    pub is_filibuster: Option<bool>,
    pub is_lobby: Option<bool>,
    pub lobby_agents: Option<Vec<serde_json::Value>>,
    pub historical_accuracy: Option<serde_json::Value>,
    pub final_sentiments: Option<serde_json::Value>,
    pub amendments: Option<Vec<serde_json::Value>>,
    pub top_persuaders: Option<Vec<serde_json::Value>>,
    pub final_sentiment: Option<f64>,
}

// ── SimulationScreen ─────────────────────────────────────────────────────────

pub struct SimulationScreen {
    pub topic: String,
    pub model: String,
    pub agents: Vec<AgentInfo>,
    pub current_tick: u32,
    pub max_ticks: u32,
    pub phase: String,
    pub phase_name: String,
    pub feed: Vec<FeedEntry>,
    pub agent_streams: HashMap<String, AgentStream>,
    pub votes: HashMap<String, Vote>,
    pub yea_count: u32,
    pub nay_count: u32,
    pub abstain_count: u32,
    pub tokens_per_sec: f64,
    pub total_tokens: u64,
    pub layout_mode: LayoutMode,
    pub selected_agent: usize,
    pub feed_scroll: u16,
    pub running: bool,
    pub simulation_result: Option<String>,
    pub throughput_buckets: VecDeque<u32>,
    pub last_bucket_time: Instant,
    pub token_count_this_second: u32,
    pub agent_sentiment: HashMap<String, f64>,
    pub amendments: Vec<AmendmentInfo>,
    pub filibuster: Option<FilibusterInfo>,
    pub lobby_agents: Vec<String>,
    pub persuasion_edges: Vec<(String, String, f64)>,
    pub historical_accuracy: Option<f64>,
    // New fields
    pub sentiment_history: HashMap<String, VecDeque<f64>>,
    pub bill_text: String,
    pub previous_bill_text: String,
    pub show_inspector: bool,
    pub inspector_agent: Option<String>,
    pub inspector_scroll: u16,
    pub show_persuasion_graph: bool,
}

impl SimulationScreen {
    pub fn new(topic: String, max_ticks: u32, model: String) -> Self {
        Self {
            topic,
            model,
            agents: Vec::new(),
            current_tick: 0,
            max_ticks,
            phase: String::from("WAITING"),
            phase_name: String::from("Waiting"),
            feed: Vec::new(),
            agent_streams: HashMap::new(),
            votes: HashMap::new(),
            yea_count: 0,
            nay_count: 0,
            abstain_count: 0,
            tokens_per_sec: 0.0,
            total_tokens: 0,
            layout_mode: LayoutMode::Focus,
            selected_agent: 0,
            feed_scroll: 0,
            running: true,
            simulation_result: None,
            throughput_buckets: VecDeque::from(vec![0; 60]),
            last_bucket_time: Instant::now(),
            token_count_this_second: 0,
            agent_sentiment: HashMap::new(),
            amendments: Vec::new(),
            filibuster: None,
            lobby_agents: Vec::new(),
            persuasion_edges: Vec::new(),
            historical_accuracy: None,
            sentiment_history: HashMap::new(),
            bill_text: String::new(),
            previous_bill_text: String::new(),
            show_inspector: false,
            inspector_agent: None,
            inspector_scroll: 0,
            show_persuasion_graph: false,
        }
    }

    /// Rotate the throughput ring buffer if a second has elapsed.
    pub fn tick_throughput(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_bucket_time);
        if elapsed.as_secs() >= 1 {
            let seconds = elapsed.as_secs().min(60) as usize;
            for _ in 0..seconds {
                self.throughput_buckets.pop_front();
                self.throughput_buckets.push_back(self.token_count_this_second);
                self.token_count_this_second = 0;
            }
            self.last_bucket_time = now;
        }
    }

    /// Process a WebSocket event and update screen state.
    pub fn handle_event(&mut self, event: WsEvent) {
        let event_type = event
            .event_type
            .as_deref()
            .or(event.event.as_deref())
            .unwrap_or("unknown");

        match event_type {
            "simulation_start" | "simulation_init" | "init" => {
                if let Some(agents_arr) = &event.agents {
                    self.agents.clear();
                    for a in agents_arr {
                        let name = a
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Unknown")
                            .to_string();
                        let party = a
                            .get("party")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let state = a
                            .get("state")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let chamber = a
                            .get("chamber")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Senate")
                            .to_string();
                        self.agent_streams.insert(
                            name.clone(),
                            AgentStream {
                                name: name.clone(),
                                tokens: String::new(),
                                active: false,
                                latency_ms: 0,
                                last_response: String::new(),
                            },
                        );
                        self.agent_sentiment.insert(name.clone(), 0.0);
                        self.agents.push(AgentInfo {
                            name,
                            party,
                            state,
                            chamber,
                        });
                    }
                }
                if let Some(lobby_arr) = &event.lobby_agents {
                    for la in lobby_arr {
                        if let Some(name) = la.get("name").and_then(|v| v.as_str()) {
                            self.lobby_agents.push(name.to_string());
                        }
                    }
                }
                if let Some(m) = &event.model {
                    self.model = m.clone();
                }
                if let Some(mt) = event.max_ticks {
                    self.max_ticks = mt;
                }
                if let Some(t) = &event.content {
                    self.topic = t.clone();
                }
                self.phase = String::from("INITIALIZED");
                self.phase_name = String::from("Initialized");
                let lobby_msg = if self.lobby_agents.is_empty() {
                    String::new()
                } else {
                    format!(", {} lobby witnesses", self.lobby_agents.len())
                };
                self.add_system_feed(format!(
                    "Simulation initialized: {} agents{}",
                    self.agents.len(),
                    lobby_msg,
                ));
            }
            "tick_start" | "tick" => {
                if let Some(t) = event.tick {
                    self.current_tick = t;
                }
                if let Some(p) = &event.phase {
                    self.phase = p.clone();
                }
                if let Some(pn) = &event.phase_name {
                    self.phase_name = pn.clone();
                }
            }
            "agent_speaking" => {
                let agent_name = event
                    .agent
                    .as_deref()
                    .or(event.agent_name.as_deref())
                    .unwrap_or("Unknown")
                    .to_string();

                if let Some(stream) = self.agent_streams.get_mut(&agent_name) {
                    stream.active = true;
                    stream.tokens.clear();
                }

                if let Some(idx) = self.agents.iter().position(|a| a.name == agent_name) {
                    self.selected_agent = idx;
                }
            }
            "group_discussion" => {
                let names = if let Some(ref parts) = event.participants {
                    parts.join(", ")
                } else {
                    event.content.clone().unwrap_or_default()
                };
                if !names.is_empty() {
                    self.add_system_feed(format!("Group discussion: {}", names));
                }
            }
            "token_stream" | "token" | "stream" => {
                let agent_name = event
                    .agent
                    .as_deref()
                    .or(event.agent_name.as_deref())
                    .unwrap_or("Unknown")
                    .to_string();
                let token_text = event
                    .token
                    .as_deref()
                    .or(event.tokens.as_deref())
                    .or(event.content.as_deref())
                    .unwrap_or("")
                    .to_string();

                if let Some(stream) = self.agent_streams.get_mut(&agent_name) {
                    stream.tokens.push_str(&token_text);
                    stream.active = true;
                    if let Some(lat) = event.latency_ms {
                        stream.latency_ms = lat;
                    }
                } else {
                    self.agent_streams.insert(
                        agent_name.clone(),
                        AgentStream {
                            name: agent_name.clone(),
                            tokens: token_text.clone(),
                            active: true,
                            latency_ms: event.latency_ms.unwrap_or(0),
                            last_response: String::new(),
                        },
                    );
                }
                self.token_count_this_second += 1;
                self.total_tokens += 1;
            }
            "agent_done" | "agent_complete" | "response_complete" => {
                let agent_name = event
                    .agent
                    .as_deref()
                    .or(event.agent_name.as_deref())
                    .unwrap_or("Unknown")
                    .to_string();

                let is_lobby = event.is_lobby.unwrap_or(false);
                let is_filibuster = event.is_filibuster.unwrap_or(false);

                let mut feed_content = None;
                if let Some(stream) = self.agent_streams.get_mut(&agent_name) {
                    stream.active = false;
                    stream.last_response = stream.tokens.clone();
                    if let Some(lat) = event.latency_ms {
                        stream.latency_ms = lat;
                    }
                    let content = event
                        .full_response
                        .as_ref()
                        .or(event.content.as_ref())
                        .cloned()
                        .unwrap_or_else(|| stream.tokens.clone());
                    stream.tokens.clear();
                    feed_content = Some(content);
                }

                if let Some(s) = event.sentiment {
                    self.agent_sentiment.insert(agent_name.clone(), s);
                }

                if let Some(content) = feed_content {
                    let party = self
                        .agents
                        .iter()
                        .find(|a| a.name == agent_name)
                        .map(|a| a.party.clone())
                        .unwrap_or_default();

                    let entry_type = if is_lobby {
                        FeedEntryType::Lobby
                    } else if is_filibuster {
                        FeedEntryType::Filibuster
                    } else {
                        FeedEntryType::Speech
                    };

                    self.add_feed(FeedEntry {
                        tick: self.current_tick,
                        agent_name: agent_name.clone(),
                        party,
                        content,
                        entry_type,
                        timestamp: event.timestamp.clone().unwrap_or_default(),
                    });
                }
            }
            "opinion_drift" => {
                let agent_name = event
                    .agent_name
                    .as_deref()
                    .unwrap_or("Unknown")
                    .to_string();
                if let Some(new_score) = event.new_score {
                    self.agent_sentiment.insert(agent_name.clone(), new_score);
                    // Track sentiment history (ring buffer, cap at 60)
                    let history = self.sentiment_history.entry(agent_name).or_default();
                    if history.len() >= 60 {
                        history.pop_front();
                    }
                    history.push_back(new_score);
                }
            }
            "direct_address" => {
                let speaker = event.speaker.clone().unwrap_or_default();
                let target = event.target.clone().unwrap_or_default();
                if !speaker.is_empty() && !target.is_empty() {
                    self.add_feed(FeedEntry {
                        tick: self.current_tick,
                        agent_name: speaker.clone(),
                        party: String::new(),
                        content: format!("Addressing {}", target),
                        entry_type: FeedEntryType::DirectAddress,
                        timestamp: String::new(),
                    });
                }
            }
            "amendment_proposed" => {
                let agent_name = event.agent_name.clone().unwrap_or_default();
                let id = event.amendment_id.unwrap_or(0);
                let text = event.amendment_text.clone().unwrap_or_default();

                self.amendments.push(AmendmentInfo {
                    id,
                    proposer: agent_name.clone(),
                    text: text.clone(),
                    status: String::from("pending"),
                    yea: 0,
                    nay: 0,
                });

                self.add_feed(FeedEntry {
                    tick: self.current_tick,
                    agent_name,
                    party: String::new(),
                    content: format!("Amendment #{}: {}", id, truncate_str(&text, 80)),
                    entry_type: FeedEntryType::Amendment,
                    timestamp: String::new(),
                });
            }
            "amendment_vote_result" | "amendment_voting" => {
                if let Some(id) = event.amendment_id {
                    let result = event.result.clone().unwrap_or_default();
                    let yea = event.yea.unwrap_or(0);
                    let nay = event.nay.unwrap_or(0);

                    if let Some(amend) = self.amendments.iter_mut().find(|a| a.id == id) {
                        amend.status = result.clone();
                        amend.yea = yea;
                        amend.nay = nay;
                    }

                    // Track bill text changes
                    if let Some(ref new_text) = event.new_bill_text {
                        self.previous_bill_text = self.bill_text.clone();
                        self.bill_text = new_text.clone();
                    }

                    if !result.is_empty() {
                        self.add_system_feed(format!(
                            "Amendment #{} {}: {}-{}",
                            id,
                            result.to_uppercase(),
                            yea,
                            nay,
                        ));
                    }
                }
            }
            "lobby_speaking" => {
                let agent_name = event.agent_name.clone().unwrap_or_default();
                let affiliation = event.affiliation.clone().unwrap_or_default();

                if !self.agent_streams.contains_key(&agent_name) {
                    self.agent_streams.insert(
                        agent_name.clone(),
                        AgentStream {
                            name: agent_name.clone(),
                            tokens: String::new(),
                            active: true,
                            latency_ms: 0,
                            last_response: String::new(),
                        },
                    );
                } else if let Some(stream) = self.agent_streams.get_mut(&agent_name) {
                    stream.active = true;
                    stream.tokens.clear();
                }

                self.add_system_feed(format!(
                    "Lobby testimony: {} ({})",
                    agent_name, affiliation,
                ));
            }
            "filibuster_start" => {
                let agent_name = event.agent_name.clone().unwrap_or_default();
                self.filibuster = Some(FilibusterInfo {
                    agent_name: agent_name.clone(),
                    start_tick: self.current_tick,
                    active: true,
                });
                self.add_system_feed(format!(
                    "FILIBUSTER: {} has taken the floor!",
                    agent_name,
                ));
            }
            "cloture_vote" => {
                let result = event.result.clone().unwrap_or_default();
                let yea = event.yea.unwrap_or(0);
                let nay = event.nay.unwrap_or(0);
                self.add_system_feed(format!(
                    "Cloture vote: {} ({}-{})",
                    result.to_uppercase(),
                    yea,
                    nay,
                ));
            }
            "filibuster_end" => {
                let reason = event.reason.clone().unwrap_or_default();
                if let Some(ref mut fb) = self.filibuster {
                    fb.active = false;
                }
                self.add_system_feed(format!("Filibuster ended ({})", reason));
            }
            "persuasion_update" => {
                let influencer = event.influencer.clone().unwrap_or_default();
                let influenced = event.influenced.clone().unwrap_or_default();
                let strength = event.strength.unwrap_or(0.0);

                if !influencer.is_empty() && !influenced.is_empty() {
                    if let Some(edge) = self
                        .persuasion_edges
                        .iter_mut()
                        .find(|(a, b, _)| a == &influencer && b == &influenced)
                    {
                        edge.2 = strength;
                    } else {
                        self.persuasion_edges.push((influencer, influenced, strength));
                    }
                }
            }
            "speech" | "statement" => {
                let agent_name = event
                    .agent
                    .as_deref()
                    .or(event.agent_name.as_deref())
                    .unwrap_or("Unknown")
                    .to_string();
                let content = event.content.clone().unwrap_or_default();
                let party = event.party.clone().unwrap_or_else(|| {
                    self.agents
                        .iter()
                        .find(|a| a.name == agent_name)
                        .map(|a| a.party.clone())
                        .unwrap_or_default()
                });

                self.add_feed(FeedEntry {
                    tick: event.tick.unwrap_or(self.current_tick),
                    agent_name,
                    party,
                    content,
                    entry_type: FeedEntryType::Speech,
                    timestamp: event.timestamp.clone().unwrap_or_default(),
                });
            }
            "vote" | "vote_cast" => {
                let agent_name = event
                    .agent
                    .as_deref()
                    .or(event.agent_name.as_deref())
                    .unwrap_or("Unknown")
                    .to_string();
                let party = event.party.clone().unwrap_or_else(|| {
                    self.agents
                        .iter()
                        .find(|a| a.name == agent_name)
                        .map(|a| a.party.clone())
                        .unwrap_or_default()
                });
                let vote_str = event.vote.clone().unwrap_or_default();
                let rationale = event.rationale.clone().unwrap_or_default();

                match vote_str.to_lowercase().as_str() {
                    "yea" | "yes" => self.yea_count += 1,
                    "nay" | "no" => self.nay_count += 1,
                    "abstain" => self.abstain_count += 1,
                    _ => {}
                }

                self.votes.insert(
                    agent_name.clone(),
                    Vote {
                        agent_name: agent_name.clone(),
                        party: party.clone(),
                        vote: vote_str.clone(),
                        rationale: rationale.clone(),
                    },
                );

                self.add_feed(FeedEntry {
                    tick: event.tick.unwrap_or(self.current_tick),
                    agent_name,
                    party,
                    content: format!("VOTE: {} - {}", vote_str.to_uppercase(), rationale),
                    entry_type: FeedEntryType::Vote,
                    timestamp: event.timestamp.clone().unwrap_or_default(),
                });
            }
            "vote_tally" | "tally" => {
                if let Some(y) = event.yea {
                    self.yea_count = y;
                }
                if let Some(n) = event.nay {
                    self.nay_count = n;
                }
                if let Some(a) = event.abstain {
                    self.abstain_count = a;
                }
            }
            "phase_change" | "phase" => {
                if let Some(p) = &event.phase {
                    self.phase = p.clone();
                }
                if let Some(pn) = &event.phase_name {
                    self.phase_name = pn.clone();
                }
                if let Some(msg) = &event.message {
                    self.add_system_feed(msg.clone());
                }
            }
            "metrics" | "stats" => {
                if let Some(tps) = event.tokens_per_sec {
                    self.tokens_per_sec = tps;
                }
                if let Some(tt) = event.total_tokens {
                    self.total_tokens = tt;
                }
            }
            "simulation_complete" | "complete" | "done" => {
                self.running = false;

                if let Some(ref hist) = event.historical_accuracy {
                    if let Some(acc) = hist.get("accuracy").and_then(|v| v.as_f64()) {
                        self.historical_accuracy = Some(acc);
                    }
                }

                self.simulation_result = event
                    .result
                    .clone()
                    .or_else(|| event.message.clone())
                    .or(Some("Simulation complete".to_string()));

                let mut result_msg = self
                    .simulation_result
                    .clone()
                    .unwrap_or_else(|| "Simulation complete".to_string());

                if let Some(acc) = self.historical_accuracy {
                    result_msg.push_str(&format!(" | Historical accuracy: {:.0}%", acc));
                }

                self.add_system_feed(result_msg);
            }
            "error" => {
                let msg = event
                    .message
                    .clone()
                    .unwrap_or_else(|| "Unknown error".to_string());
                self.add_system_feed(format!("ERROR: {}", msg));
            }
            _ => {
                // Unknown event type; ignore silently
            }
        }

        if let Some(tps) = event.tokens_per_sec {
            self.tokens_per_sec = tps;
        }
    }

    fn add_feed(&mut self, entry: FeedEntry) {
        self.feed.push(entry);
        if self.feed.len() > 500 {
            self.feed.remove(0);
        }
    }

    fn add_system_feed(&mut self, message: String) {
        self.add_feed(FeedEntry {
            tick: self.current_tick,
            agent_name: String::from("SYSTEM"),
            party: String::new(),
            content: message,
            entry_type: FeedEntryType::System,
            timestamp: chrono::Local::now().format("%H:%M:%S").to_string(),
        });
    }

    pub fn toggle_layout(&mut self) {
        self.layout_mode = match self.layout_mode {
            LayoutMode::Focus => LayoutMode::Grid,
            LayoutMode::Grid => LayoutMode::Focus,
        };
    }

    pub fn next_agent(&mut self) {
        if !self.agents.is_empty() {
            self.selected_agent = (self.selected_agent + 1) % self.agents.len();
        }
    }

    pub fn prev_agent(&mut self) {
        if !self.agents.is_empty() {
            if self.selected_agent == 0 {
                self.selected_agent = self.agents.len() - 1;
            } else {
                self.selected_agent -= 1;
            }
        }
    }

    pub fn scroll_feed_up(&mut self) {
        if self.feed_scroll > 0 {
            self.feed_scroll -= 1;
        }
    }

    pub fn scroll_feed_down(&mut self) {
        let max = self.feed.len() as u16;
        if self.feed_scroll < max.saturating_sub(1) {
            self.feed_scroll += 1;
        }
    }

    pub fn is_running(&self) -> bool {
        self.running
    }

    pub fn force_complete(&mut self, msg: String) {
        self.running = false;
        if self.simulation_result.is_none() {
            self.simulation_result = Some(msg);
        }
    }

    // ── Key Handling ─────────────────────────────────────────────────────────

    pub fn handle_key(&mut self, key: crossterm::event::KeyEvent) -> super::Action {
        use crossterm::event::KeyCode;

        // If inspector is open, handle inspector keys first
        if self.show_inspector {
            match key.code {
                KeyCode::Esc => {
                    self.show_inspector = false;
                    return super::Action::None;
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.inspector_scroll = self.inspector_scroll.saturating_sub(1);
                    return super::Action::None;
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.inspector_scroll += 1;
                    return super::Action::None;
                }
                _ => return super::Action::None,
            }
        }

        // If persuasion graph is showing, Esc closes it
        if self.show_persuasion_graph {
            match key.code {
                KeyCode::Esc => {
                    self.show_persuasion_graph = false;
                    return super::Action::None;
                }
                _ => {} // fall through to normal keys
            }
        }

        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => {
                if !self.running {
                    super::Action::SwitchScreen(super::ScreenId::Models)
                } else {
                    super::Action::Quit
                }
            }
            KeyCode::Tab => {
                self.toggle_layout();
                super::Action::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                match self.layout_mode {
                    LayoutMode::Focus => self.next_agent(),
                    LayoutMode::Grid => self.scroll_feed_down(),
                }
                super::Action::None
            }
            KeyCode::Up | KeyCode::Char('k') => {
                match self.layout_mode {
                    LayoutMode::Focus => self.prev_agent(),
                    LayoutMode::Grid => self.scroll_feed_up(),
                }
                super::Action::None
            }
            KeyCode::PageDown => {
                for _ in 0..10 {
                    self.scroll_feed_down();
                }
                super::Action::None
            }
            KeyCode::PageUp => {
                for _ in 0..10 {
                    self.scroll_feed_up();
                }
                super::Action::None
            }
            KeyCode::Char('g') => {
                self.show_persuasion_graph = !self.show_persuasion_graph;
                super::Action::None
            }
            KeyCode::Enter => {
                // Open inspector for selected agent
                if let Some(agent) = self.agents.get(self.selected_agent) {
                    self.show_inspector = true;
                    self.inspector_agent = Some(agent.name.clone());
                    self.inspector_scroll = 0;
                }
                super::Action::None
            }
            KeyCode::Home => {
                self.feed_scroll = 0;
                super::Action::None
            }
            KeyCode::End => {
                self.feed_scroll = self.feed.len().saturating_sub(1) as u16;
                super::Action::None
            }
            _ => super::Action::None,
        }
    }

    // ── Draw ─────────────────────────────────────────────────────────────────

    pub fn draw(&self, f: &mut Frame, area: Rect) {
        // Top-level layout: header (3) + body (flex) + status bar (1)
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(10),
                Constraint::Length(1),
            ])
            .split(area);

        self.draw_header(f, chunks[0]);
        self.draw_status_bar(f, chunks[2]);

        match self.layout_mode {
            LayoutMode::Focus => self.draw_focus(f, chunks[1]),
            LayoutMode::Grid => self.draw_grid(f, chunks[1]),
        }

        // Overlays (drawn last, on top)
        if self.show_persuasion_graph {
            let nodes: Vec<crate::widgets::persuasion_graph::GraphNode> = self.agents.iter()
                .map(|a| crate::widgets::persuasion_graph::GraphNode {
                    name: a.name.clone(),
                    party: a.party.clone(),
                })
                .collect();
            let edges: Vec<crate::widgets::persuasion_graph::GraphEdge> = self.persuasion_edges.iter()
                .filter_map(|(from, to, strength)| {
                    let from_idx = self.agents.iter().position(|a| &a.name == from)?;
                    let to_idx = self.agents.iter().position(|a| &a.name == to)?;
                    Some(crate::widgets::persuasion_graph::GraphEdge {
                        from: from_idx,
                        to: to_idx,
                        strength: *strength,
                    })
                })
                .collect();
            let overlay = centered_rect(70, 70, area);
            crate::widgets::persuasion_graph::draw_persuasion_graph(f, overlay, &nodes, &edges);
        }

        if self.show_inspector {
            if let Some(ref agent_name) = self.inspector_agent {
                self.draw_inspector(f, area, agent_name);
            }
        }
    }

    // ── Header ───────────────────────────────────────────────────────────────

    fn draw_header(&self, f: &mut Frame, area: Rect) {
        let progress = if self.max_ticks > 0 {
            self.current_tick as f64 / self.max_ticks as f64
        } else {
            0.0
        };

        let tps = self.compute_tps();

        let mut title_spans = vec![
            Span::styled(
                " AI CONGRESS ",
                Style::default()
                    .fg(theme::INFO)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" | ", Style::default().fg(theme::DIM)),
            Span::styled(
                format!("Tick {}/{}", self.current_tick, self.max_ticks),
                Style::default().fg(theme::INFO),
            ),
            Span::styled(" | ", Style::default().fg(theme::DIM)),
            Span::styled(
                &self.phase_name,
                Style::default()
                    .fg(theme::WARNING)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" | ", Style::default().fg(theme::DIM)),
            Span::styled(&self.model, Style::default().fg(theme::SECONDARY)),
            Span::styled(" | ", Style::default().fg(theme::DIM)),
            Span::styled(
                format!("{:.1} tok/s", tps),
                Style::default().fg(theme::SUCCESS),
            ),
        ];

        if let Some(ref fb) = self.filibuster {
            if fb.active {
                title_spans.push(Span::styled(" | ", Style::default().fg(theme::DIM)));
                title_spans.push(Span::styled(
                    format!(" FILIBUSTER: {} ", fb.agent_name),
                    Style::default()
                        .fg(theme::BG)
                        .bg(theme::ERROR)
                        .add_modifier(Modifier::BOLD),
                ));
            }
        }

        let title_line = Line::from(title_spans);

        let topic_line = Line::from(vec![
            Span::styled(" Topic: ", Style::default().fg(theme::MUTED)),
            Span::styled(
                truncate(&self.topic, area.width as usize - 10),
                Style::default().fg(theme::INFO),
            ),
        ]);

        let progress_line = Line::from(vec![
            Span::raw(" "),
            Span::styled(
                make_progress_bar(progress, (area.width as usize).saturating_sub(4)),
                Style::default().fg(progress_color(progress)),
            ),
            Span::raw(" "),
        ]);

        let block = Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(theme::DIM));

        let text = vec![title_line, topic_line, progress_line];
        let paragraph = Paragraph::new(text).block(block);
        f.render_widget(paragraph, area);
    }


















    // ── Status Bar ───────────────────────────────────────────────────────────

    fn draw_status_bar(&self, f: &mut Frame, area: Rect) {
        let sparkline = render_sparkline(&self.throughput_buckets, 20);
        let tps = self.compute_tps();

        let status = if self.running { "LIVE" } else { "DONE" };
        let status_color = if self.running {
            theme::SUCCESS
        } else {
            theme::MUTED
        };

        let mut spans = vec![
            Span::styled(
                format!(" {} ", status),
                Style::default()
                    .fg(theme::BG)
                    .bg(status_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(sparkline, Style::default().fg(theme::INFO)),
            Span::styled(
                format!(" {:.1} tok/s", tps),
                Style::default().fg(theme::INFO),
            ),
            Span::styled(
                format!("  {} total", self.total_tokens),
                Style::default().fg(theme::MUTED),
            ),
        ];

        if let Some(acc) = self.historical_accuracy {
            spans.push(Span::styled(
                format!("  Accuracy: {:.0}%", acc),
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            ));
        }

        spans.extend([
            Span::raw("  "),
            Span::styled(
                "Tab",
                Style::default()
                    .fg(theme::INFO)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(":layout ", Style::default().fg(theme::SUBTEXT)),
            Span::styled(
                "j/k",
                Style::default()
                    .fg(theme::INFO)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(":agent ", Style::default().fg(theme::SUBTEXT)),
            Span::styled(
                "g",
                Style::default()
                    .fg(theme::INFO)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(":graph ", Style::default().fg(theme::SUBTEXT)),
            Span::styled(
                "Enter",
                Style::default()
                    .fg(theme::INFO)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(":inspect ", Style::default().fg(theme::SUBTEXT)),
            Span::styled(
                "q",
                Style::default()
                    .fg(theme::INFO)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(":quit", Style::default().fg(theme::SUBTEXT)),
        ]);

        let line = Line::from(spans);
        let paragraph = Paragraph::new(line);
        f.render_widget(paragraph, area);
    }

    fn draw_inspector(&self, f: &mut Frame, area: Rect, agent_name: &str) {
        let overlay = centered_rect(80, 80, area);

        // Clear background
        f.render_widget(ratatui::widgets::Clear, overlay);
        let bg = Block::default().style(Style::default().bg(theme::BG));
        f.render_widget(bg, overlay);

        let block = Block::default()
            .title(format!(" Agent: {} ", agent_name))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::INFO));
        let inner = block.inner(overlay);
        f.render_widget(block, overlay);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),   // sentiment sparkline
                Constraint::Length(5),   // influence map
                Constraint::Min(5),      // response history
            ])
            .split(inner);

        // 1. Sentiment timeline
        if let Some(history) = self.sentiment_history.get(agent_name) {
            let history_vec: Vec<f64> = history.iter().copied().collect();
            if !history_vec.is_empty() {
                let sparkline = crate::widgets::agent_pane::render_sentiment_sparkline(
                    &history_vec,
                    chunks[0].width.saturating_sub(12) as usize,
                );
                let current = history_vec.last().copied().unwrap_or(0.0);
                let label = format!("Sentiment: {:.2} ", current);
                let lines = vec![
                    Line::from(vec![
                        Span::styled(label, Style::default().fg(theme::MUTED)),
                    ]),
                    sparkline,
                ];
                let block = Block::default()
                    .title(" Sentiment Timeline ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(theme::DIM));
                let para = Paragraph::new(lines).block(block);
                f.render_widget(para, chunks[0]);
            }
        }

        // 2. Influence map
        {
            let incoming: Vec<_> = self.persuasion_edges.iter()
                .filter(|(_, target, _)| target == agent_name)
                .collect();
            let outgoing: Vec<_> = self.persuasion_edges.iter()
                .filter(|(source, _, _)| source == agent_name)
                .collect();

            let mut lines = Vec::new();
            if !incoming.is_empty() {
                lines.push(Line::from(vec![
                    Span::styled("← Influenced by: ", Style::default().fg(theme::MUTED)),
                ]));
                let spans: Vec<Span> = incoming.iter().take(5).map(|(from, _, strength)| {
                    let last = from.split_whitespace().last().unwrap_or(from.as_str());
                    Span::styled(format!("{} ({:.2})  ", last, strength), Style::default().fg(theme::INFO))
                }).collect();
                lines.push(Line::from(spans));
            }
            if !outgoing.is_empty() {
                lines.push(Line::from(vec![
                    Span::styled("→ Influenced: ", Style::default().fg(theme::MUTED)),
                ]));
                let spans: Vec<Span> = outgoing.iter().take(5).map(|(_, to, strength)| {
                    let last = to.split_whitespace().last().unwrap_or(to.as_str());
                    Span::styled(format!("{} ({:.2})  ", last, strength), Style::default().fg(theme::SECONDARY))
                }).collect();
                lines.push(Line::from(spans));
            }
            if lines.is_empty() {
                lines.push(Line::from(Span::styled(
                    "No influence data yet",
                    Style::default().fg(theme::MUTED),
                )));
            }

            let block = Block::default()
                .title(" Influence Map ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::DIM));
            let para = Paragraph::new(lines).block(block);
            f.render_widget(para, chunks[1]);
        }

        // 3. Response history
        {
            let speeches: Vec<&FeedEntry> = self.feed.iter()
                .filter(|e| e.agent_name == agent_name)
                .collect();

            let visible_height = chunks[2].height.saturating_sub(2) as usize;
            let start = if speeches.len() <= visible_height {
                0
            } else {
                speeches.len()
                    .saturating_sub(visible_height)
                    .min(self.inspector_scroll as usize)
            };

            let lines: Vec<Line> = speeches.iter()
                .skip(start)
                .take(visible_height)
                .map(|entry| {
                    let icon = match entry.entry_type {
                        FeedEntryType::Speech => ">>",
                        FeedEntryType::Vote => "##",
                        FeedEntryType::Lobby => "$$",
                        FeedEntryType::Filibuster => "!!",
                        FeedEntryType::Amendment => "&&",
                        FeedEntryType::DirectAddress => "->",
                        FeedEntryType::System => "**",
                    };
                    let max_width = chunks[2].width.saturating_sub(15) as usize;
                    let content = if entry.content.len() > max_width {
                        format!("{}...", &entry.content[..max_width.saturating_sub(3)])
                    } else {
                        entry.content.clone()
                    };
                    Line::from(vec![
                        Span::styled(format!("[T{:>3}] ", entry.tick), Style::default().fg(theme::MUTED)),
                        Span::styled(format!("{} ", icon), Style::default().fg(theme::INFO)),
                        Span::styled(content, Style::default().fg(theme::SUBTEXT)),
                    ])
                })
                .collect();

            let block = Block::default()
                .title(format!(" History ({} entries) ", speeches.len()))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::DIM));
            let para = Paragraph::new(lines).block(block);
            f.render_widget(para, chunks[2]);
        }
    }


}

// ── Free helper functions (kept) ──────────────────────────────────

pub(crate) fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len > 3 {
        format!("{}...", &s[..max_len - 3])
    } else {
        s[..max_len].to_string()
    }
}

pub(crate) fn party_abbrev(party: &str) -> &str {
    match party.to_uppercase().as_str() {
        "REPUBLICAN" => "R",
        "DEMOCRATIC" | "DEMOCRAT" => "D",
        "INDEPENDENT" => "I",
        _ => "?",
    }
}

pub(crate) fn state_abbrev(state: &str) -> String {
    match state {
        "Alabama" => "AL",
        "Alaska" => "AK",
        "Arizona" => "AZ",
        "Arkansas" => "AR",
        "California" => "CA",
        "Colorado" => "CO",
        "Connecticut" => "CT",
        "Delaware" => "DE",
        "Florida" => "FL",
        "Georgia" => "GA",
        "Hawaii" => "HI",
        "Idaho" => "ID",
        "Illinois" => "IL",
        "Indiana" => "IN",
        "Iowa" => "IA",
        "Kansas" => "KS",
        "Kentucky" => "KY",
        "Louisiana" => "LA",
        "Maine" => "ME",
        "Maryland" => "MD",
        "Massachusetts" => "MA",
        "Michigan" => "MI",
        "Minnesota" => "MN",
        "Mississippi" => "MS",
        "Missouri" => "MO",
        "Montana" => "MT",
        "Nebraska" => "NE",
        "Nevada" => "NV",
        "New Hampshire" => "NH",
        "New Jersey" => "NJ",
        "New Mexico" => "NM",
        "New York" => "NY",
        "North Carolina" => "NC",
        "North Dakota" => "ND",
        "Ohio" => "OH",
        "Oklahoma" => "OK",
        "Oregon" => "OR",
        "Pennsylvania" => "PA",
        "Rhode Island" => "RI",
        "South Carolina" => "SC",
        "South Dakota" => "SD",
        "Tennessee" => "TN",
        "Texas" => "TX",
        "Utah" => "UT",
        "Vermont" => "VT",
        "Virginia" => "VA",
        "Washington" => "WA",
        "West Virginia" => "WV",
        "Wisconsin" => "WI",
        "Wyoming" => "WY",
        _ => return state[..2.min(state.len())].to_uppercase(),
    }
    .to_string()
}
