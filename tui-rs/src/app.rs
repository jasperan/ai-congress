use std::collections::{HashMap, VecDeque};
use std::time::Instant;

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
}

#[derive(Clone, Debug)]
pub struct Vote {
    pub agent_name: String,
    pub party: String,
    pub vote: String,
    pub rationale: String,
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
                        self.agents.push(AgentInfo {
                            name,
                            party,
                            state,
                            chamber,
                        });
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
                self.add_system_feed(format!("Simulation initialized: {} agents", self.agents.len()));
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

                // Mark agent active and clear token buffer for new generation
                if let Some(stream) = self.agent_streams.get_mut(&agent_name) {
                    stream.active = true;
                    stream.tokens.clear();
                }

                // Auto-select this agent in focus mode
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
                // Count tokens for throughput
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

                // Extract data from stream before calling add_feed (avoids double borrow)
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

                if let Some(content) = feed_content {
                    let party = self
                        .agents
                        .iter()
                        .find(|a| a.name == agent_name)
                        .map(|a| a.party.clone())
                        .unwrap_or_default();

                    self.add_feed(FeedEntry {
                        tick: self.current_tick,
                        agent_name: agent_name.clone(),
                        party,
                        content,
                        entry_type: FeedEntryType::Speech,
                        timestamp: event.timestamp.clone().unwrap_or_default(),
                    });
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
                self.simulation_result = event
                    .result
                    .clone()
                    .or_else(|| event.message.clone())
                    .or(Some("Simulation complete".to_string()));
                self.add_system_feed(
                    self.simulation_result
                        .clone()
                        .unwrap_or_else(|| "Simulation complete".to_string()),
                );
            }
            "error" => {
                let msg = event.message.clone().unwrap_or_else(|| "Unknown error".to_string());
                self.add_system_feed(format!("ERROR: {}", msg));
            }
            _ => {
                // Unknown event type; ignore silently
            }
        }

        // Update tokens_per_sec from event if provided
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
