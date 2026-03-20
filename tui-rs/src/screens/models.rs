use std::collections::HashSet;
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph};
use ratatui::Frame;
use tui_input::backend::crossterm::EventHandler;
use tui_input::Input;

use crate::api::rest::ModelInfo;
use crate::theme;

pub struct ModelsScreen {
    pub models: Vec<ModelInfo>,
    filtered_indices: Vec<usize>,
    selected: HashSet<String>,
    list_state: ListState,
    filter_input: Input,
    filter_active: bool,
    pub loading: bool,
    pub error_msg: Option<String>,
}

impl ModelsScreen {
    pub fn new() -> Self {
        let mut s = Self {
            models: Vec::new(),
            filtered_indices: Vec::new(),
            selected: HashSet::new(),
            list_state: ListState::default(),
            filter_input: Input::default(),
            filter_active: false,
            loading: true,
            error_msg: None,
        };
        s.list_state.select(Some(0));
        s
    }

    pub fn set_models(&mut self, models: Vec<ModelInfo>) {
        self.models = models;
        self.loading = false;
        self.apply_filter();
    }

    pub fn set_error(&mut self, msg: String) {
        self.loading = false;
        self.error_msg = Some(msg);
    }

    fn apply_filter(&mut self) {
        let query = self.filter_input.value().to_lowercase();
        self.filtered_indices = self
            .models
            .iter()
            .enumerate()
            .filter(|(_, m)| query.is_empty() || m.name.to_lowercase().contains(&query))
            .map(|(i, _)| i)
            .collect();
        if let Some(sel) = self.list_state.selected() {
            if sel >= self.filtered_indices.len() {
                self.list_state.select(Some(0));
            }
        }
        if self.filtered_indices.is_empty() {
            self.list_state.select(None);
        }
    }

    fn focused_model(&self) -> Option<&ModelInfo> {
        self.list_state
            .selected()
            .and_then(|i| self.filtered_indices.get(i))
            .and_then(|&idx| self.models.get(idx))
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> super::Action {
        if self.filter_active {
            match key.code {
                KeyCode::Esc => {
                    self.filter_active = false;
                    return super::Action::None;
                }
                KeyCode::Enter => {
                    self.filter_active = false;
                    return super::Action::None;
                }
                _ => {
                    self.filter_input
                        .handle_event(&crossterm::event::Event::Key(key));
                    self.apply_filter();
                    return super::Action::None;
                }
            }
        }

        match key.code {
            KeyCode::Char('/') => {
                self.filter_active = true;
                super::Action::None
            }
            KeyCode::Char(' ') => {
                if let Some(model) = self.focused_model() {
                    let name = model.name.clone();
                    if !self.selected.remove(&name) {
                        self.selected.insert(name);
                    }
                }
                super::Action::None
            }
            KeyCode::Enter => {
                let models: Vec<String> = if self.selected.is_empty() {
                    self.focused_model()
                        .map(|m| vec![m.name.clone()])
                        .unwrap_or_default()
                } else {
                    self.selected.iter().cloned().collect()
                };
                if !models.is_empty() {
                    super::Action::SwitchScreen(super::ScreenId::ModeSelect {
                        selected_models: models,
                    })
                } else {
                    super::Action::None
                }
            }
            KeyCode::Up | KeyCode::Char('k') => {
                let len = self.filtered_indices.len();
                if len > 0 {
                    let i = self.list_state.selected().unwrap_or(0);
                    let new = if i == 0 { len - 1 } else { i - 1 };
                    self.list_state.select(Some(new));
                }
                super::Action::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let len = self.filtered_indices.len();
                if len > 0 {
                    let i = self.list_state.selected().unwrap_or(0);
                    let new = if i >= len - 1 { 0 } else { i + 1 };
                    self.list_state.select(Some(new));
                }
                super::Action::None
            }
            KeyCode::Char('q') | KeyCode::Esc => {
                super::Action::SwitchScreen(super::ScreenId::Splash)
            }
            _ => super::Action::None,
        }
    }

    pub fn draw(&mut self, f: &mut Frame, area: Rect) {
        if self.loading {
            let loading = Paragraph::new("Loading models...")
                .alignment(ratatui::layout::Alignment::Center)
                .style(Style::default().fg(theme::YELLOW));
            f.render_widget(loading, area);
            return;
        }

        if let Some(ref err) = self.error_msg {
            let error = Paragraph::new(format!("Error: {}\n\nq: back", err))
                .alignment(ratatui::layout::Alignment::Center)
                .style(Style::default().fg(theme::RED));
            f.render_widget(error, area);
            return;
        }

        // Layout: filter bar (3 if active) + list + footer (1)
        let filter_height = if self.filter_active { 3 } else { 0 };
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(filter_height),
                Constraint::Min(5),
                Constraint::Length(1),
            ])
            .split(area);

        // Filter input
        if self.filter_active {
            let filter_block = Block::default()
                .title(" Filter ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::YELLOW));
            let filter_text = Paragraph::new(self.filter_input.value())
                .block(filter_block)
                .style(Style::default().fg(theme::CYAN));
            f.render_widget(filter_text, chunks[0]);
        }

        // Model list
        let items: Vec<ListItem> = self
            .filtered_indices
            .iter()
            .enumerate()
            .map(|(list_idx, &model_idx)| {
                let model = &self.models[model_idx];
                let is_selected = self.selected.contains(&model.name);
                let is_focused = self.list_state.selected() == Some(list_idx);

                let checkbox = if is_selected { "[✓]" } else { "[ ]" };
                let checkbox_color = if is_selected { theme::GREEN } else { theme::DIM_GRAY };

                let name_style = if is_focused {
                    Style::default()
                        .fg(theme::CYAN)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(theme::ACCENT)
                };

                let size_str = format_size(model.size);
                let backend = if model.backend.is_empty() {
                    "ollama"
                } else {
                    &model.backend
                };

                let line1 = Line::from(vec![
                    Span::styled(
                        format!("{} ", checkbox),
                        Style::default().fg(checkbox_color),
                    ),
                    Span::styled(&model.name, name_style),
                    Span::styled(
                        format!("  (w:{:.2})", model.weight),
                        Style::default().fg(theme::DIM_GRAY),
                    ),
                ]);
                let line2 = Line::from(vec![
                    Span::styled(
                        format!("    {}  [{}]", size_str, backend),
                        Style::default().fg(theme::DIM_GRAY),
                    ),
                ]);

                ListItem::new(vec![line1, line2])
            })
            .collect();

        let list_block = Block::default()
            .title(" Available Models ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::CYAN));

        let list = List::new(items)
            .block(list_block)
            .highlight_style(Style::default().bg(ratatui::style::Color::Rgb(30, 30, 40)));

        f.render_stateful_widget(list, chunks[1], &mut self.list_state);

        // Footer
        let selected_count = self.selected.len();
        let footer_text = if selected_count > 0 {
            format!(
                " {} selected  Space: toggle  Enter: continue  /: filter  q: back",
                selected_count
            )
        } else {
            " Space: toggle  Enter: continue  /: filter  q: back".to_string()
        };
        let footer = Paragraph::new(Line::from(Span::styled(
            footer_text,
            Style::default().fg(theme::DIM_GRAY),
        )));
        f.render_widget(footer, chunks[2]);
    }
}

fn format_size(bytes: i64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{} MB", bytes / 1_048_576)
    } else if bytes > 0 {
        format!("{} KB", bytes / 1024)
    } else {
        "—".to_string()
    }
}
