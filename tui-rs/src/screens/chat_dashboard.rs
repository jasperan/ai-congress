use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};
use tui_input::backend::crossterm::EventHandler;
use tui_input::Input;

use crate::api::ws_chat::WsChatEvent;
use crate::theme;
use crate::widgets::sparkline_widget::{compute_tps, render_throughput_sparkline};
use crate::widgets::{
    agent_pane, feed, vote_tracker, AgentPaneData, FeedEntryData, FeedEntryType, VoteTrackerData,
};
use super::{Action, ChatLaunchConfig, ScreenId, SessionResult};

// ── Local enums ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum LayoutMode {
    Focus,
    Grid,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChatStatus {
    Waiting,
    Streaming,
    Complete,
    Error,
}

// ── ModelStream ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ModelStream {
    pub name: String,
    pub tokens: String,
    pub last_response: String,
    pub active: bool,
    pub complete: bool,
}

impl ModelStream {
    fn new(name: String) -> Self {
        Self {
            name,
            tokens: String::new(),
            last_response: String::new(),
            active: false,
            complete: false,
        }
    }
}

// ── ChatDashboardScreen ────────────────────────────────────────────────────────

const THROUGHPUT_BUCKETS: usize = 60;

pub struct ChatDashboardScreen {
    pub config: ChatLaunchConfig,
    pub model_streams: HashMap<String, ModelStream>,
    pub model_order: Vec<String>,
    pub feed: Vec<FeedEntryData>,
    pub feed_scroll: u16,
    pub final_answer: Option<String>,
    pub confidence: Option<f64>,
    pub vote_breakdown: Option<serde_json::Value>,
    pub layout_mode: LayoutMode,
    pub selected_model: usize,
    // throughput tracking
    pub throughput_buckets: VecDeque<u32>,
    pub last_bucket_time: Instant,
    pub token_count_this_second: u32,
    pub total_tokens: u64,
    // status
    pub status: ChatStatus,
    pub error_msg: Option<String>,
    // follow-up prompt
    pub prompt_input: Input,
    pub prompt_active: bool,
    // inspector overlay
    pub show_inspector: bool,
    pub inspector_model: Option<String>,
    pub inspector_scroll: u16,
}

impl ChatDashboardScreen {
    pub fn new(config: ChatLaunchConfig) -> Self {
        let mut buckets = VecDeque::new();
        buckets.resize(THROUGHPUT_BUCKETS, 0u32);
        Self {
            config,
            model_streams: HashMap::new(),
            model_order: Vec::new(),
            feed: Vec::new(),
            feed_scroll: 0,
            final_answer: None,
            confidence: None,
            vote_breakdown: None,
            layout_mode: LayoutMode::Focus,
            selected_model: 0,
            throughput_buckets: buckets,
            last_bucket_time: Instant::now(),
            token_count_this_second: 0,
            total_tokens: 0,
            status: ChatStatus::Waiting,
            error_msg: None,
            prompt_input: Input::default(),
            prompt_active: false,
            show_inspector: false,
            inspector_model: None,
            inspector_scroll: 0,
        }
    }

    // ── WebSocket event handler ──────────────────────────────────────────────

    pub fn handle_ws_event(&mut self, event: WsChatEvent) {
        let idx = self.feed.len() as u32;
        let now = chrono::Local::now().format("%H:%M:%S").to_string();

        match event.event_type.as_str() {
            "start" => {
                self.status = ChatStatus::Streaming;
                self.feed.push(FeedEntryData {
                    tick_or_index: idx,
                    agent_name: "System".to_string(),
                    party: None,
                    content: "Session started".to_string(),
                    entry_type: FeedEntryType::System,
                    timestamp: now,
                });
            }

            "status_update" => {
                if let Some(name) = &event.name {
                    let name = name.clone();
                    if !self.model_order.contains(&name) {
                        self.model_order.push(name.clone());
                    }
                    let stream = self
                        .model_streams
                        .entry(name.clone())
                        .or_insert_with(|| ModelStream::new(name.clone()));
                    stream.active = true;
                    if let Some(status) = &event.status {
                        stream.tokens = status.clone();
                    }
                }
            }

            "chunk" => {
                if let Some(content) = &event.content {
                    let key = event
                        .name
                        .clone()
                        .or_else(|| event.model.clone())
                        .unwrap_or_else(|| "model".to_string());

                    if !self.model_order.contains(&key) {
                        self.model_order.push(key.clone());
                    }
                    let stream = self
                        .model_streams
                        .entry(key.clone())
                        .or_insert_with(|| ModelStream::new(key.clone()));
                    stream.active = true;
                    stream.tokens.push_str(content);

                    let token_count = content.split_whitespace().count().max(1) as u32;
                    self.token_count_this_second += token_count;
                    self.total_tokens += token_count as u64;
                }
            }

            "model_response" => {
                let model_name = event
                    .model
                    .clone()
                    .or_else(|| {
                        event
                            .data
                            .as_ref()
                            .and_then(|d| d.get("model"))
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    })
                    .unwrap_or_else(|| "unknown".to_string());

                let response_text = event
                    .content
                    .clone()
                    .or_else(|| {
                        event
                            .data
                            .as_ref()
                            .and_then(|d| d.get("content"))
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    })
                    .unwrap_or_default();

                if !self.model_order.contains(&model_name) {
                    self.model_order.push(model_name.clone());
                }
                let stream = self
                    .model_streams
                    .entry(model_name.clone())
                    .or_insert_with(|| ModelStream::new(model_name.clone()));
                stream.active = false;
                stream.complete = true;
                stream.last_response = response_text.clone();

                self.feed.push(FeedEntryData {
                    tick_or_index: idx,
                    agent_name: model_name,
                    party: None,
                    content: response_text,
                    entry_type: FeedEntryType::ModelResponse,
                    timestamp: now,
                });
            }

            "final_answer" => {
                self.final_answer = event
                    .response
                    .clone()
                    .or_else(|| event.content.clone())
                    .or_else(|| event.message.clone());
                self.confidence = event.confidence.or(event.semantic_confidence);
                self.vote_breakdown = event
                    .vote_breakdown
                    .clone()
                    .or_else(|| event.semantic_vote.clone());

                let summary = self
                    .final_answer
                    .clone()
                    .unwrap_or_else(|| "Final answer received".to_string());

                self.feed.push(FeedEntryData {
                    tick_or_index: idx,
                    agent_name: "Congress".to_string(),
                    party: None,
                    content: summary,
                    entry_type: FeedEntryType::FinalAnswer,
                    timestamp: now,
                });
            }

            "end" => {
                self.status = ChatStatus::Complete;
                for stream in self.model_streams.values_mut() {
                    stream.active = false;
                }
            }

            "error" => {
                self.status = ChatStatus::Error;
                self.error_msg = event
                    .message
                    .clone()
                    .or_else(|| event.content.clone())
                    .or_else(|| {
                        event
                            .data
                            .as_ref()
                            .and_then(|d| d.get("error"))
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    });
            }

            _ => {}
        }
    }

    // ── Throughput tick ──────────────────────────────────────────────────────

    pub fn tick_throughput(&mut self) {
        if self.last_bucket_time.elapsed().as_secs_f64() >= 1.0 {
            if self.throughput_buckets.len() >= THROUGHPUT_BUCKETS {
                self.throughput_buckets.pop_front();
            }
            self.throughput_buckets.push_back(self.token_count_this_second);
            self.token_count_this_second = 0;
            self.last_bucket_time = Instant::now();
        }
    }

    // ── Key handler ──────────────────────────────────────────────────────────

    pub fn handle_key(&mut self, key: KeyEvent) -> Action {
        // Inspector overlay takes all input
        if self.show_inspector {
            match key.code {
                KeyCode::Esc | KeyCode::Char('q') => {
                    self.show_inspector = false;
                    self.inspector_scroll = 0;
                }
                KeyCode::Char('j') | KeyCode::Down => {
                    self.inspector_scroll = self.inspector_scroll.saturating_add(1);
                }
                KeyCode::Char('k') | KeyCode::Up => {
                    self.inspector_scroll = self.inspector_scroll.saturating_sub(1);
                }
                _ => {}
            }
            return Action::None;
        }

        // Follow-up prompt input
        if self.prompt_active {
            match key.code {
                KeyCode::Esc => {
                    self.prompt_active = false;
                }
                KeyCode::Enter => {
                    let prompt = self.prompt_input.value().to_string();
                    if !prompt.is_empty() {
                        self.prompt_input.reset();
                        self.prompt_active = false;
                        let mut new_config = self.config.clone();
                        new_config.prompt = prompt;
                        return Action::SwitchScreen(ScreenId::ChatDashboard(new_config));
                    }
                }
                _ => {
                    self.prompt_input
                        .handle_event(&crossterm::event::Event::Key(key));
                }
            }
            return Action::None;
        }

        match key.code {
            KeyCode::Tab => {
                self.layout_mode = if self.layout_mode == LayoutMode::Focus {
                    LayoutMode::Grid
                } else {
                    LayoutMode::Focus
                };
            }

            KeyCode::Char('j') | KeyCode::Down => {
                if self.layout_mode == LayoutMode::Focus {
                    let n = self.model_order.len();
                    if n > 0 {
                        self.selected_model = (self.selected_model + 1) % n;
                    }
                } else {
                    self.feed_scroll = self.feed_scroll.saturating_add(1);
                }
            }

            KeyCode::Char('k') | KeyCode::Up => {
                if self.layout_mode == LayoutMode::Focus {
                    let n = self.model_order.len();
                    if n > 0 {
                        self.selected_model =
                            (self.selected_model + n.saturating_sub(1)) % n.max(1);
                    }
                } else {
                    self.feed_scroll = self.feed_scroll.saturating_sub(1);
                }
            }

            KeyCode::Char(c) if c.is_ascii_digit() && c != '0' => {
                let idx = (c as usize) - ('1' as usize);
                if idx < self.model_order.len() {
                    self.selected_model = idx;
                }
            }

            KeyCode::Enter => {
                if let Some(name) = self.model_order.get(self.selected_model) {
                    self.inspector_model = Some(name.clone());
                    self.show_inspector = true;
                    self.inspector_scroll = 0;
                }
            }

            KeyCode::Char('p') if self.status == ChatStatus::Complete => {
                self.prompt_active = true;
            }

            KeyCode::Char('q') | KeyCode::Esc => {
                if self.status == ChatStatus::Complete {
                    let result = self.build_results();
                    return Action::SwitchScreen(ScreenId::Results(result));
                } else {
                    return Action::Quit;
                }
            }

            _ => {}
        }

        Action::None
    }

    // ── Build SessionResult ──────────────────────────────────────────────────

    pub fn build_results(&self) -> SessionResult {
        let responses: Vec<(String, String)> = self
            .model_order
            .iter()
            .filter_map(|name| {
                self.model_streams.get(name).map(|s| {
                    let text = if !s.last_response.is_empty() {
                        s.last_response.clone()
                    } else {
                        s.tokens.clone()
                    };
                    (name.clone(), text)
                })
            })
            .collect();

        SessionResult::Chat {
            final_answer: self.final_answer.clone().unwrap_or_default(),
            confidence: self.confidence.unwrap_or(0.0),
            vote_breakdown: self
                .vote_breakdown
                .clone()
                .unwrap_or(serde_json::Value::Null),
            responses,
        }
    }

    // ── Draw ─────────────────────────────────────────────────────────────────

    pub fn draw(&self, f: &mut Frame, area: Rect) {
        if self.show_inspector {
            self.draw_inspector(f, area);
            return;
        }
        match self.layout_mode {
            LayoutMode::Focus => self.draw_focus(f, area),
            LayoutMode::Grid => self.draw_grid(f, area),
        }
    }

    fn draw_focus(&self, f: &mut Frame, area: Rect) {
        let buckets_slice: Vec<u32> = self.throughput_buckets.iter().copied().collect();
        let tps = compute_tps(&buckets_slice, 5);

        let prompt_bar = if self.prompt_active { 3u16 } else { 0 };
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(0),
                Constraint::Length(1),
                Constraint::Length(prompt_bar),
            ])
            .split(area);

        // Header
        self.draw_header(f, rows[0], tps);

        // Body: left 55% / right 45%
        let body_cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
            .split(rows[1]);

        // Left: up to 4 stacked agent panes
        let visible: Vec<&str> = self.model_order.iter().map(|s| s.as_str()).take(4).collect();
        if !visible.is_empty() {
            let per_pane = body_cols[0].height / visible.len() as u16;
            let constraints: Vec<Constraint> =
                visible.iter().map(|_| Constraint::Length(per_pane)).collect();
            let pane_areas = Layout::default()
                .direction(Direction::Vertical)
                .constraints(constraints)
                .split(body_cols[0]);

            for (i, name) in visible.iter().enumerate() {
                let stream = self.model_streams.get(*name);
                let data = AgentPaneData {
                    name: (*name).to_string(),
                    subtitle: None,
                    tokens: stream.map(|s| s.tokens.clone()).unwrap_or_default(),
                    last_response: stream
                        .map(|s| s.last_response.clone())
                        .unwrap_or_default(),
                    active: stream.map(|s| s.active).unwrap_or(false),
                    selected: i == self.selected_model,
                    latency_ms: 0,
                    sentiment: None,
                    sentiment_history: None,
                    vote: None,
                };
                agent_pane::draw_agent_pane(f, pane_areas[i], &data);
            }
        }

        // Right: feed (65%) + vote tracker (35%)
        let right_rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
            .split(body_cols[1]);

        feed::draw_feed(f, right_rows[0], &self.feed, self.feed_scroll);

        let vtdata = self.build_vote_tracker_data();
        vote_tracker::draw_vote_tracker(f, right_rows[1], &vtdata);

        // Status bar
        let sparkline_str =
            render_throughput_sparkline(&buckets_slice, area.width.saturating_sub(50) as usize);
        let hints = if self.status == ChatStatus::Complete {
            "Tab:layout  j/k:nav  Enter:inspect  p:follow-up  q:results"
        } else {
            "Tab:layout  j/k:nav  Enter:inspect  q:quit"
        };
        let status_line = Line::from(vec![
            Span::styled(sparkline_str, Style::default().fg(theme::BLUE)),
            Span::styled(format!(" {:.1}t/s  ", tps), Style::default().fg(theme::ACCENT)),
            Span::styled(hints, Style::default().fg(theme::DIM_GRAY)),
        ]);
        f.render_widget(Paragraph::new(vec![status_line]), rows[2]);

        // Prompt bar
        if self.prompt_active && prompt_bar > 0 {
            let prompt_block = Block::default()
                .title(" Follow-up Prompt (Enter to send, Esc to cancel) ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::CYAN));
            let prompt_inner = prompt_block.inner(rows[3]);
            f.render_widget(prompt_block, rows[3]);
            let prompt_para = Paragraph::new(self.prompt_input.value())
                .style(Style::default().fg(theme::GRAY));
            f.render_widget(prompt_para, prompt_inner);
        }
    }

    fn draw_grid(&self, f: &mut Frame, area: Rect) {
        let buckets_slice: Vec<u32> = self.throughput_buckets.iter().copied().collect();
        let tps = compute_tps(&buckets_slice, 5);

        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),
                Constraint::Min(0),
                Constraint::Length(1),
            ])
            .split(area);

        // Simple header line
        let status_str = self.status_str();
        let header = Line::from(vec![
            Span::styled(
                " AI CONGRESS CHAT [GRID] ",
                Style::default().fg(theme::CYAN).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(
                    "| {} models | {:.1} tok/s | {} ",
                    self.model_order.len(),
                    tps,
                    status_str
                ),
                Style::default().fg(theme::DIM_GRAY),
            ),
        ]);
        f.render_widget(Paragraph::new(vec![header]), rows[0]);

        // 5x2 grid of model cells
        let all_models: Vec<&str> = self.model_order.iter().map(|s| s.as_str()).collect();
        let cols = 5usize;
        let total = all_models.len().min(10);
        if total > 0 {
            let row_count = (total + cols - 1) / cols;
            let row_pct = (100 / row_count) as u16;
            let row_constraints: Vec<Constraint> =
                (0..row_count).map(|_| Constraint::Percentage(row_pct)).collect();
            let grid_rows = Layout::default()
                .direction(Direction::Vertical)
                .constraints(row_constraints)
                .split(rows[1]);

            for row_i in 0..row_count {
                let start = row_i * cols;
                let end = (start + cols).min(total);
                let col_count = end - start;
                if col_count == 0 {
                    break;
                }
                let col_pct = (100 / col_count) as u16;
                let col_constraints: Vec<Constraint> =
                    (0..col_count).map(|_| Constraint::Percentage(col_pct)).collect();
                let cell_areas = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints(col_constraints)
                    .split(grid_rows[row_i]);

                for (col_i, model_name) in all_models[start..end].iter().enumerate() {
                    let global_idx = start + col_i;
                    let stream = self.model_streams.get(*model_name);
                    let data = AgentPaneData {
                        name: (*model_name).to_string(),
                        subtitle: None,
                        tokens: stream.map(|s| s.tokens.clone()).unwrap_or_default(),
                        last_response: stream
                            .map(|s| s.last_response.clone())
                            .unwrap_or_default(),
                        active: stream.map(|s| s.active).unwrap_or(false),
                        selected: global_idx == self.selected_model,
                        latency_ms: 0,
                        sentiment: None,
                        sentiment_history: None,
                        vote: None,
                    };
                    agent_pane::draw_agent_pane(f, cell_areas[col_i], &data);
                }
            }
        }

        // Status bar
        let sparkline_str = render_throughput_sparkline(&buckets_slice, 40);
        let status_line = Line::from(vec![
            Span::styled(sparkline_str, Style::default().fg(theme::BLUE)),
            Span::styled(
                format!(" {:.1}t/s  Tab:focus  j/k:feed  q:quit/results", tps),
                Style::default().fg(theme::DIM_GRAY),
            ),
        ]);
        f.render_widget(Paragraph::new(vec![status_line]), rows[2]);
    }

    fn draw_inspector(&self, f: &mut Frame, area: Rect) {
        let model_name = match &self.inspector_model {
            Some(n) => n.clone(),
            None => return,
        };
        let stream = match self.model_streams.get(&model_name) {
            Some(s) => s,
            None => return,
        };

        let block = Block::default()
            .title(format!(
                " Inspector: {} (Esc to close, j/k scroll) ",
                model_name
            ))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::CYAN));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let full_text = if !stream.last_response.is_empty() {
            stream.last_response.clone()
        } else {
            stream.tokens.clone()
        };

        let para = Paragraph::new(full_text)
            .style(Style::default().fg(theme::GRAY))
            .wrap(Wrap { trim: false })
            .scroll((self.inspector_scroll, 0));
        f.render_widget(para, inner);
    }

    fn draw_header(&self, f: &mut Frame, area: Rect, tps: f64) {
        let status_str = self.status_str();
        let status_color = self.status_color();

        let header_text = vec![
            Line::from(vec![
                Span::styled(
                    " AI CONGRESS CHAT ",
                    Style::default()
                        .fg(theme::CYAN)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("| {} models ", self.model_order.len()),
                    Style::default().fg(theme::DIM_GRAY),
                ),
                Span::styled(
                    format!("| {} ", self.config.mode),
                    Style::default().fg(theme::ACCENT),
                ),
                Span::styled(
                    format!("| {:.1} tok/s ", tps),
                    Style::default().fg(theme::BLUE),
                ),
                Span::styled(
                    format!("| {} ", status_str),
                    Style::default()
                        .fg(status_color)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![Span::styled(
                format!(" Prompt: {}", &self.config.prompt),
                Style::default().fg(theme::DIM_GRAY),
            )]),
        ];

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::DARK_GRAY));
        let inner = block.inner(area);
        f.render_widget(block, area);
        f.render_widget(Paragraph::new(header_text), inner);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn status_str(&self) -> &'static str {
        match self.status {
            ChatStatus::Waiting => "WAITING",
            ChatStatus::Streaming => "STREAMING",
            ChatStatus::Complete => "COMPLETE",
            ChatStatus::Error => "ERROR",
        }
    }

    fn status_color(&self) -> ratatui::style::Color {
        match self.status {
            ChatStatus::Waiting => theme::YELLOW,
            ChatStatus::Streaming => theme::GREEN,
            ChatStatus::Complete => theme::CYAN,
            ChatStatus::Error => theme::RED,
        }
    }

    fn build_vote_tracker_data(&self) -> VoteTrackerData {
        let breakdown: Option<Vec<(String, f64)>> =
            self.vote_breakdown.as_ref().and_then(|v| {
                v.as_object().map(|obj| {
                    obj.iter()
                        .filter_map(|(k, val)| val.as_f64().map(|f| (k.clone(), f)))
                        .collect()
                })
            });

        VoteTrackerData {
            yea: 0,
            nay: 0,
            abstain: 0,
            total_agents: self.model_order.len(),
            per_agent_votes: Vec::new(),
            persuasion_edges: Vec::new(),
            result: self.final_answer.as_ref().map(|a| {
                let short = &a[..a.len().min(40)];
                format!("{}{}", short, if a.len() > 40 { "..." } else { "" })
            }),
            confidence: self.confidence.map(|c| c * 100.0),
            vote_breakdown: breakdown,
        }
    }
}
