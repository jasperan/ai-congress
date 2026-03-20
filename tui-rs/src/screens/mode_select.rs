use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;
use tui_input::backend::crossterm::EventHandler;
use tui_input::Input;

use crate::theme;

const SWARM_MODES: &[&str] = &["multi_model", "multi_request", "hybrid", "personality", "streaming"];
const VOTING_MODES: &[&str] = &["classic", "semantic"];
const BACKENDS: &[&str] = &["ollama", "openai"];

#[derive(Debug, Clone, PartialEq)]
pub enum SessionMode {
    Chat,
    Simulation,
}

pub struct ModeSelectScreen {
    pub selected_models: Vec<String>,
    mode: SessionMode,
    // Chat fields
    prompt: Input,
    temperature: Input,
    swarm_mode_idx: usize,
    voting_mode_idx: usize,
    backend_idx: usize,
    // Simulation fields
    sim_topic: Input,
    sim_agents: u32,
    sim_ticks: u32,
    // UI state
    focus_idx: usize,
    error_msg: Option<String>,
}

impl ModeSelectScreen {
    pub fn new(selected_models: Vec<String>) -> Self {
        Self {
            selected_models,
            mode: SessionMode::Chat,
            prompt: Input::default(),
            temperature: Input::new("0.7".to_string()),
            swarm_mode_idx: 0,
            voting_mode_idx: 0,
            backend_idx: 0,
            sim_topic: Input::new("Should AI systems be regulated by federal law?".to_string()),
            sim_agents: 10,
            sim_ticks: 100,
            focus_idx: 0,
            error_msg: None,
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> super::Action {
        // Mode toggle with Left/Right when focus is on mode cards (idx 0)
        if self.focus_idx == 0 {
            match key.code {
                KeyCode::Left | KeyCode::Right => {
                    self.mode = match self.mode {
                        SessionMode::Chat => SessionMode::Simulation,
                        SessionMode::Simulation => SessionMode::Chat,
                    };
                    return super::Action::None;
                }
                _ => {}
            }
        }

        // Ctrl shortcuts (chat mode only)
        if self.mode == SessionMode::Chat {
            if key.modifiers.contains(KeyModifiers::CONTROL) {
                match key.code {
                    KeyCode::Char('s') => {
                        self.swarm_mode_idx = (self.swarm_mode_idx + 1) % SWARM_MODES.len();
                        return super::Action::None;
                    }
                    KeyCode::Char('d') => {
                        self.voting_mode_idx = (self.voting_mode_idx + 1) % VOTING_MODES.len();
                        return super::Action::None;
                    }
                    KeyCode::Char('b') => {
                        self.backend_idx = (self.backend_idx + 1) % BACKENDS.len();
                        return super::Action::None;
                    }
                    _ => {}
                }
            }
        }

        match key.code {
            KeyCode::Tab => {
                let max_focus = if self.mode == SessionMode::Chat { 2 } else { 3 };
                self.focus_idx = (self.focus_idx + 1) % (max_focus + 1);
                super::Action::None
            }
            KeyCode::BackTab => {
                let max_focus = if self.mode == SessionMode::Chat { 2 } else { 3 };
                self.focus_idx = if self.focus_idx == 0 { max_focus } else { self.focus_idx - 1 };
                super::Action::None
            }
            KeyCode::Enter => self.try_launch(),
            KeyCode::Esc => super::Action::SwitchScreen(super::ScreenId::Models),
            _ => {
                // Route text input to the focused field
                match self.mode {
                    SessionMode::Chat => match self.focus_idx {
                        1 => {
                            self.prompt.handle_event(&crossterm::event::Event::Key(key));
                        }
                        2 => {
                            self.temperature.handle_event(&crossterm::event::Event::Key(key));
                        }
                        _ => {}
                    },
                    SessionMode::Simulation => match self.focus_idx {
                        1 => {
                            self.sim_topic.handle_event(&crossterm::event::Event::Key(key));
                        }
                        2 => match key.code {
                            KeyCode::Up => self.sim_agents = (self.sim_agents + 1).min(10),
                            KeyCode::Down => self.sim_agents = self.sim_agents.saturating_sub(1).max(1),
                            _ => {}
                        },
                        3 => match key.code {
                            KeyCode::Up => self.sim_ticks += 10,
                            KeyCode::Down => self.sim_ticks = self.sim_ticks.saturating_sub(10).max(10),
                            _ => {}
                        },
                        _ => {}
                    },
                }
                super::Action::None
            }
        }
    }

    fn try_launch(&mut self) -> super::Action {
        match self.mode {
            SessionMode::Chat => {
                let prompt = self.prompt.value().trim().to_string();
                if prompt.is_empty() {
                    self.error_msg = Some("Prompt cannot be empty".to_string());
                    return super::Action::None;
                }
                let temp: f64 = self
                    .temperature
                    .value()
                    .trim()
                    .parse()
                    .unwrap_or(0.7);

                super::Action::SwitchScreen(super::ScreenId::ChatDashboard(
                    super::ChatLaunchConfig {
                        prompt,
                        models: self.selected_models.clone(),
                        mode: SWARM_MODES[self.swarm_mode_idx].to_string(),
                        temperature: temp,
                        voting_mode: VOTING_MODES[self.voting_mode_idx].to_string(),
                        inference_backend: BACKENDS[self.backend_idx].to_string(),
                    },
                ))
            }
            SessionMode::Simulation => {
                let topic = self.sim_topic.value().trim().to_string();
                if topic.is_empty() {
                    self.error_msg = Some("Topic cannot be empty".to_string());
                    return super::Action::None;
                }
                let model = self
                    .selected_models
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "qwen3.5:9b".to_string());

                super::Action::SwitchScreen(super::ScreenId::Simulation(
                    super::SimLaunchConfig {
                        topic,
                        agents: self.sim_agents,
                        ticks: self.sim_ticks,
                        model,
                    },
                ))
            }
        }
    }

    pub fn draw(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),  // title
                Constraint::Length(1),  // models subtitle
                Constraint::Length(5),  // mode cards
                Constraint::Min(10),    // config fields
                Constraint::Length(1),  // error / hints
            ])
            .split(area);

        // Title
        let title = Paragraph::new(Line::from(Span::styled(
            "Launch Session",
            Style::default().fg(theme::CYAN).add_modifier(Modifier::BOLD),
        )))
        .alignment(Alignment::Center);
        f.render_widget(title, chunks[0]);

        // Models subtitle
        let models_str = self.selected_models.join(", ");
        let subtitle = Paragraph::new(Line::from(Span::styled(
            format!("Models: {}", models_str),
            Style::default().fg(theme::DIM_GRAY),
        )))
        .alignment(Alignment::Center);
        f.render_widget(subtitle, chunks[1]);

        // Mode cards
        let card_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(10),
                Constraint::Percentage(35),
                Constraint::Percentage(10),
                Constraint::Percentage(35),
                Constraint::Percentage(10),
            ])
            .split(chunks[2]);

        let chat_border = if self.mode == SessionMode::Chat {
            Style::default().fg(theme::CYAN)
        } else {
            Style::default().fg(theme::DARK_GRAY)
        };
        let sim_border = if self.mode == SessionMode::Simulation {
            Style::default().fg(theme::CYAN)
        } else {
            Style::default().fg(theme::DARK_GRAY)
        };

        let chat_card = Paragraph::new(vec![
            Line::from(""),
            Line::from(Span::styled(
                "Chat / Swarm",
                if self.mode == SessionMode::Chat {
                    Style::default().fg(theme::CYAN).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(theme::DIM_GRAY)
                },
            )),
        ])
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(chat_border),
        );

        let sim_card = Paragraph::new(vec![
            Line::from(""),
            Line::from(Span::styled(
                "Congressional Simulation",
                if self.mode == SessionMode::Simulation {
                    Style::default().fg(theme::CYAN).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(theme::DIM_GRAY)
                },
            )),
        ])
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(sim_border),
        );

        f.render_widget(chat_card, card_layout[1]);
        f.render_widget(sim_card, card_layout[3]);

        // Config fields
        let config_block = Block::default()
            .title(" Configuration ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::DARK_GRAY));
        let config_inner = config_block.inner(chunks[3]);
        f.render_widget(config_block, chunks[3]);

        match self.mode {
            SessionMode::Chat => self.draw_chat_config(f, config_inner),
            SessionMode::Simulation => self.draw_sim_config(f, config_inner),
        }

        // Error / hints
        let hint_line = if let Some(ref err) = self.error_msg {
            Line::from(Span::styled(err, Style::default().fg(theme::RED)))
        } else {
            let hint = match self.mode {
                SessionMode::Chat => {
                    "Tab: cycle fields  Ctrl+S: swarm mode  Ctrl+D: voting  Ctrl+B: backend  Enter: launch  Esc: back"
                }
                SessionMode::Simulation => {
                    "Tab: cycle fields  ←/→: switch mode  Enter: launch  Esc: back"
                }
            };
            Line::from(Span::styled(hint, Style::default().fg(theme::DIM_GRAY)))
        };
        let hints = Paragraph::new(hint_line).alignment(Alignment::Center);
        f.render_widget(hints, chunks[4]);
    }

    fn draw_chat_config(&self, f: &mut Frame, area: Rect) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Min(0),
            ])
            .split(area);

        // Prompt
        let prompt_style = if self.focus_idx == 1 {
            Style::default().fg(theme::CYAN)
        } else {
            Style::default().fg(theme::GRAY)
        };
        let prompt_val = self.prompt.value();
        let prompt_display = if prompt_val.is_empty() && self.focus_idx != 1 {
            "Ask anything..."
        } else {
            prompt_val
        };
        let prompt_line = Line::from(vec![
            Span::styled("  Prompt: ", Style::default().fg(theme::DIM_GRAY)),
            Span::styled(prompt_display, prompt_style),
            if self.focus_idx == 1 {
                Span::styled("▌", Style::default().fg(theme::CYAN))
            } else {
                Span::raw("")
            },
        ]);
        f.render_widget(Paragraph::new(prompt_line), rows[0]);

        // Temperature
        let temp_style = if self.focus_idx == 2 {
            Style::default().fg(theme::CYAN)
        } else {
            Style::default().fg(theme::GRAY)
        };
        let temp_line = Line::from(vec![
            Span::styled("  Temperature: ", Style::default().fg(theme::DIM_GRAY)),
            Span::styled(self.temperature.value(), temp_style),
            if self.focus_idx == 2 {
                Span::styled("▌", Style::default().fg(theme::CYAN))
            } else {
                Span::raw("")
            },
        ]);
        f.render_widget(Paragraph::new(temp_line), rows[1]);

        // Swarm mode
        let mode_line = Line::from(vec![
            Span::styled("  Mode: ", Style::default().fg(theme::DIM_GRAY)),
            Span::styled(
                SWARM_MODES[self.swarm_mode_idx],
                Style::default().fg(theme::ACCENT),
            ),
            Span::styled("  (Ctrl+S to cycle)", Style::default().fg(theme::DIM_GRAY)),
        ]);
        f.render_widget(Paragraph::new(mode_line), rows[2]);

        // Voting mode
        let voting_line = Line::from(vec![
            Span::styled("  Voting: ", Style::default().fg(theme::DIM_GRAY)),
            Span::styled(
                VOTING_MODES[self.voting_mode_idx],
                Style::default().fg(theme::ACCENT),
            ),
            Span::styled("  (Ctrl+D to cycle)", Style::default().fg(theme::DIM_GRAY)),
        ]);
        f.render_widget(Paragraph::new(voting_line), rows[3]);

        // Backend
        let backend_line = Line::from(vec![
            Span::styled("  Backend: ", Style::default().fg(theme::DIM_GRAY)),
            Span::styled(
                BACKENDS[self.backend_idx],
                Style::default().fg(theme::ACCENT),
            ),
            Span::styled("  (Ctrl+B to cycle)", Style::default().fg(theme::DIM_GRAY)),
        ]);
        f.render_widget(Paragraph::new(backend_line), rows[4]);
    }

    fn draw_sim_config(&self, f: &mut Frame, area: Rect) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Min(0),
            ])
            .split(area);

        // Topic
        let topic_style = if self.focus_idx == 1 {
            Style::default().fg(theme::CYAN)
        } else {
            Style::default().fg(theme::GRAY)
        };
        let topic_line = Line::from(vec![
            Span::styled("  Topic: ", Style::default().fg(theme::DIM_GRAY)),
            Span::styled(self.sim_topic.value(), topic_style),
            if self.focus_idx == 1 {
                Span::styled("▌", Style::default().fg(theme::CYAN))
            } else {
                Span::raw("")
            },
        ]);
        f.render_widget(Paragraph::new(topic_line), rows[0]);

        // Agents
        let agents_style = if self.focus_idx == 2 {
            Style::default().fg(theme::CYAN)
        } else {
            Style::default().fg(theme::GRAY)
        };
        let agents_line = Line::from(vec![
            Span::styled("  Agents: ", Style::default().fg(theme::DIM_GRAY)),
            Span::styled(format!("{}", self.sim_agents), agents_style),
            Span::styled("  (↑/↓ to adjust, 1-10)", Style::default().fg(theme::DIM_GRAY)),
        ]);
        f.render_widget(Paragraph::new(agents_line), rows[1]);

        // Ticks
        let ticks_style = if self.focus_idx == 3 {
            Style::default().fg(theme::CYAN)
        } else {
            Style::default().fg(theme::GRAY)
        };
        let ticks_line = Line::from(vec![
            Span::styled("  Ticks: ", Style::default().fg(theme::DIM_GRAY)),
            Span::styled(format!("{}", self.sim_ticks), ticks_style),
            Span::styled("  (↑/↓ to adjust)", Style::default().fg(theme::DIM_GRAY)),
        ]);
        f.render_widget(Paragraph::new(ticks_line), rows[2]);
    }
}
