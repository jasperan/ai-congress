use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::theme;

pub struct KeyBinding {
    pub key: &'static str,
    pub description: &'static str,
}

pub const SPLASH_BINDINGS: &[KeyBinding] = &[
    KeyBinding { key: "Enter", description: "Continue" },
    KeyBinding { key: "r", description: "Retry connection" },
    KeyBinding { key: "q", description: "Quit" },
];

pub const MODELS_BINDINGS: &[KeyBinding] = &[
    KeyBinding { key: "Space", description: "Toggle model selection" },
    KeyBinding { key: "Enter", description: "Continue with selected" },
    KeyBinding { key: "↑/↓", description: "Navigate list" },
    KeyBinding { key: "/", description: "Filter models" },
    KeyBinding { key: "q", description: "Back" },
];

pub const MODE_SELECT_BINDINGS: &[KeyBinding] = &[
    KeyBinding { key: "←/→", description: "Switch Chat / Simulation" },
    KeyBinding { key: "Tab", description: "Cycle fields" },
    KeyBinding { key: "Ctrl+S", description: "Cycle swarm mode" },
    KeyBinding { key: "Ctrl+D", description: "Cycle voting mode" },
    KeyBinding { key: "Ctrl+B", description: "Cycle backend" },
    KeyBinding { key: "Enter", description: "Launch" },
    KeyBinding { key: "Esc", description: "Back" },
];

pub const CHAT_BINDINGS: &[KeyBinding] = &[
    KeyBinding { key: "Tab", description: "Toggle Focus/Grid layout" },
    KeyBinding { key: "j/k", description: "Navigate models / Scroll feed" },
    KeyBinding { key: "1-9", description: "Jump to model" },
    KeyBinding { key: "Enter", description: "Agent inspector" },
    KeyBinding { key: "p", description: "Follow-up prompt" },
    KeyBinding { key: "q", description: "Back / Results" },
];

pub const SIMULATION_BINDINGS: &[KeyBinding] = &[
    KeyBinding { key: "Tab", description: "Toggle Focus/Grid layout" },
    KeyBinding { key: "j/k", description: "Navigate agents / Scroll feed" },
    KeyBinding { key: "Enter", description: "Agent inspector" },
    KeyBinding { key: "g", description: "Persuasion graph" },
    KeyBinding { key: "Home/End", description: "Feed scroll start/end" },
    KeyBinding { key: "q", description: "Back" },
];

pub const RESULTS_BINDINGS: &[KeyBinding] = &[
    KeyBinding { key: "e", description: "Export transcript" },
    KeyBinding { key: "n", description: "New session" },
    KeyBinding { key: "j/k", description: "Scroll" },
    KeyBinding { key: "Esc", description: "Back to models" },
];

pub const GLOBAL_BINDINGS: &[KeyBinding] = &[
    KeyBinding { key: "F1", description: "Toggle this help" },
    KeyBinding { key: "Ctrl+C", description: "Quit" },
];

pub fn draw_help_overlay(
    f: &mut Frame,
    area: Rect,
    screen_name: &str,
    screen_bindings: &[KeyBinding],
) {
    let total_lines = screen_bindings.len() + GLOBAL_BINDINGS.len() + 5;
    let height = (total_lines as u16 + 4).min(area.height.saturating_sub(4));
    let width = 55u16.min(area.width.saturating_sub(4));

    let popup = centered_rect(width, height, area);

    // Dark overlay background
    f.render_widget(Clear, popup);
    let bg = Block::default().style(Style::default().bg(ratatui::style::Color::Rgb(15, 15, 20)));
    f.render_widget(bg, popup);

    let block = Block::default()
        .title(" Help ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::CYAN));
    let inner = block.inner(popup);
    f.render_widget(block, popup);

    let mut lines = Vec::new();

    // Screen-specific section
    lines.push(Line::from(Span::styled(
        format!("─── {} ───", screen_name),
        Style::default()
            .fg(theme::YELLOW)
            .add_modifier(Modifier::BOLD),
    )));

    for kb in screen_bindings {
        lines.push(Line::from(vec![
            Span::styled(
                format!("{:<14}", kb.key),
                Style::default().fg(theme::CYAN),
            ),
            Span::styled(kb.description, Style::default().fg(theme::GRAY)),
        ]));
    }

    lines.push(Line::from(""));

    // Global section
    lines.push(Line::from(Span::styled(
        "─── Global ───",
        Style::default()
            .fg(theme::YELLOW)
            .add_modifier(Modifier::BOLD),
    )));

    for kb in GLOBAL_BINDINGS {
        lines.push(Line::from(vec![
            Span::styled(
                format!("{:<14}", kb.key),
                Style::default().fg(theme::CYAN),
            ),
            Span::styled(kb.description, Style::default().fg(theme::GRAY)),
        ]));
    }

    let para = Paragraph::new(lines);
    f.render_widget(para, inner);
}

fn centered_rect(width: u16, height: u16, r: Rect) -> Rect {
    let v = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(r.height.saturating_sub(height) / 2),
            Constraint::Length(height),
            Constraint::Min(0),
        ])
        .split(r);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(r.width.saturating_sub(width) / 2),
            Constraint::Length(width),
            Constraint::Min(0),
        ])
        .split(v[1])[1]
}
