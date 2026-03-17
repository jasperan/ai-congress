use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// The main application state.
pub struct App {
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
    // New: opinion drift
    pub agent_sentiment: HashMap<String, f64>,
    // New: amendments
    pub amendments: Vec<AmendmentInfo>,
    // New: filibuster
    pub filibuster: Option<FilibusterInfo>,
    // New: lobby agents
    pub lobby_agents: Vec<String>,
    // New: persuasion edges (influencer, influenced, strength)
    pub persuasion_edges: Vec<(String, String, f64)>,
    // New: historical accuracy
    pub historical_accuracy: Option<f64>,
    // Shell feedback
    pub toast: Option<ToastState>,
}

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

#[derive(Clone, Debug)]
pub enum ToastLevel {
    Info,
    Success,
    Warning,
    Error,
}

#[derive(Clone, Debug)]
pub struct ToastState {
    pub level: ToastLevel,
    pub message: String,
    pub expires_at: Instant,
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
    // New fields for advanced features
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

impl App {
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
            toast: None,
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
                self.throughput_buckets
                    .push_back(self.token_count_this_second);
                self.token_count_this_second = 0;
            }
            self.last_bucket_time = now;
        }
    }

    pub fn show_toast(&mut self, level: ToastLevel, message: impl Into<String>) {
        self.toast = Some(ToastState {
            level,
            message: message.into(),
            expires_at: Instant::now() + Duration::from_secs(4),
        });
    }

    pub fn expire_toast(&mut self) {
        if self
            .toast
            .as_ref()
            .is_some_and(|toast| Instant::now() >= toast.expires_at)
        {
            self.toast = None;
        }
    }

    /// Process a WebSocket event and update app state.
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
                // Load lobby agent names
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

                // Update sentiment from event
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
            // ── New: Opinion Drift ──────────────────────────────────
            "opinion_drift" => {
                let agent_name = event.agent_name.as_deref().unwrap_or("Unknown").to_string();
                if let Some(new_score) = event.new_score {
                    self.agent_sentiment.insert(agent_name, new_score);
                }
            }
            // ── New: Direct Address ─────────────────────────────────
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
            // ── New: Amendment Events ───────────────────────────────
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
            // ── New: Lobby Events ───────────────────────────────────
            "lobby_speaking" => {
                let agent_name = event.agent_name.clone().unwrap_or_default();
                let affiliation = event.affiliation.clone().unwrap_or_default();

                // Create stream for lobby agent if not exists
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

                self.add_system_feed(format!("Lobby testimony: {} ({})", agent_name, affiliation,));
            }
            // ── New: Filibuster Events ──────────────────────────────
            "filibuster_start" => {
                let agent_name = event.agent_name.clone().unwrap_or_default();
                self.filibuster = Some(FilibusterInfo {
                    agent_name: agent_name.clone(),
                    start_tick: self.current_tick,
                    active: true,
                });
                self.add_system_feed(format!("FILIBUSTER: {} has taken the floor!", agent_name,));
                self.show_toast(
                    ToastLevel::Warning,
                    format!("Filibuster started: {}", agent_name),
                );
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
                self.show_toast(ToastLevel::Info, format!("Filibuster ended ({})", reason));
            }
            // ── New: Persuasion Events ──────────────────────────────
            "persuasion_update" => {
                let influencer = event.influencer.clone().unwrap_or_default();
                let influenced = event.influenced.clone().unwrap_or_default();
                let strength = event.strength.unwrap_or(0.0);

                if !influencer.is_empty() && !influenced.is_empty() {
                    // Update or add edge
                    if let Some(edge) = self
                        .persuasion_edges
                        .iter_mut()
                        .find(|(a, b, _)| a == &influencer && b == &influenced)
                    {
                        edge.2 = strength;
                    } else {
                        self.persuasion_edges
                            .push((influencer, influenced, strength));
                    }
                }
            }
            // ── Existing handlers ───────────────────────────────────
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

                // Extract historical accuracy
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

                self.add_system_feed(result_msg.clone());
                self.show_toast(ToastLevel::Success, result_msg);
            }
            "error" => {
                let msg = event
                    .message
                    .clone()
                    .unwrap_or_else(|| "Unknown error".to_string());
                self.add_system_feed(format!("ERROR: {}", msg));
                self.show_toast(ToastLevel::Error, format!("Error: {}", msg));
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
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len > 3 {
        format!("{}...", &s[..max_len - 3])
    } else {
        s[..max_len].to_string()
    }
}
