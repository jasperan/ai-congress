use std::time::Instant;
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::theme;

const BANNER: &str = r#" █████╗ ██╗     ██████╗ ██████╗ ███╗   ██╗ ██████╗ ██████╗ ███████╗███████╗███████╗
██╔══██╗██║    ██╔════╝██╔═══██╗████╗  ██║██╔════╝ ██╔══██╗██╔════╝██╔════╝██╔════╝
███████║██║    ██║     ██║   ██║██╔██╗ ██║██║  ███╗██████╔╝█████╗  ███████╗███████╗
██╔══██║██║    ██║     ██║   ██║██║╚██╗██║██║   ██║██╔══██╗██╔══╝  ╚════██║╚════██║
██║  ██║██║    ╚██████╗╚██████╔╝██║ ╚████║╚██████╔╝██║  ██║███████╗███████║███████║
╚═╝  ╚═╝╚═╝     ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝"#;

pub struct SplashScreen {
    start_time: Instant,
    pub connected: bool,
    pub checking: bool,
    pub error_msg: Option<String>,
    pub model_count: usize,
}

impl SplashScreen {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            connected: false,
            checking: true,
            error_msg: None,
            model_count: 0,
        }
    }

    pub fn set_connected(&mut self, model_count: usize) {
        self.connected = true;
        self.checking = false;
        self.model_count = model_count;
        self.error_msg = None;
    }

    pub fn set_error(&mut self, msg: String) {
        self.connected = false;
        self.checking = false;
        self.error_msg = Some(msg);
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> super::Action {
        match key.code {
            KeyCode::Enter if self.connected => super::Action::SwitchScreen(super::ScreenId::Models),
            KeyCode::Char('r') => {
                self.checking = true;
                self.error_msg = None;
                super::Action::Retry
            }
            KeyCode::Char('q') | KeyCode::Esc => super::Action::Quit,
            _ => super::Action::None,
        }
    }

    pub fn draw(&self, f: &mut Frame, area: Rect) {
        // Compute fade-in opacity (spring physics approximation)
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let opacity = (1.0 - (-3.0 * elapsed).exp()).clamp(0.0, 1.0);

        // Color lerp from dark (#1C1C1E) to primary (#5f87ff)
        let r = (0x1C as f64 + (0x5F - 0x1C) as f64 * opacity) as u8;
        let g = (0x1C as f64 + (0x87 - 0x1C) as f64 * opacity) as u8;
        let b = (0x1E as f64 + (0xFF - 0x1E) as f64 * opacity) as u8;
        let banner_color = ratatui::style::Color::Rgb(r, g, b);

        let banner_lines: Vec<Line> = BANNER
            .lines()
            .map(|l| Line::from(Span::styled(l, Style::default().fg(banner_color))))
            .collect();

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(25),
                Constraint::Length(7),
                Constraint::Length(2),
                Constraint::Length(2),
                Constraint::Percentage(25),
            ])
            .split(area);

        let banner_widget = Paragraph::new(banner_lines).alignment(Alignment::Center);
        f.render_widget(banner_widget, chunks[1]);

        // Status line with colored dot
        let (dot_color, status_text) = if self.checking {
            (theme::YELLOW, "Connecting to backend...".to_string())
        } else if self.connected {
            (theme::GREEN, format!("Connected — {} models available", self.model_count))
        } else if let Some(ref err) = self.error_msg {
            (theme::RED, format!("Connection failed: {}", err))
        } else {
            (theme::GRAY, "Waiting...".to_string())
        };

        let status = Paragraph::new(Line::from(vec![
            Span::styled("● ", Style::default().fg(dot_color)),
            Span::styled(status_text, Style::default().fg(theme::GRAY)),
        ]))
        .alignment(Alignment::Center);
        f.render_widget(status, chunks[2]);

        // Hint bar
        let hint = if self.connected {
            "Enter: continue  q: quit"
        } else if self.checking {
            "q: quit"
        } else {
            "r: retry  q: quit"
        };
        let hints = Paragraph::new(Line::from(Span::styled(
            hint,
            Style::default().fg(theme::DIM_GRAY),
        )))
        .alignment(Alignment::Center);
        f.render_widget(hints, chunks[3]);
    }
}
