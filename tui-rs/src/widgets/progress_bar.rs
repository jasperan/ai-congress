use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::theme;

pub fn draw_progress_bar(f: &mut Frame, area: Rect, progress: f64, label: &str) {
    let width = area.width as usize;
    let label_len = label.len() + 2;
    let bar_width = width.saturating_sub(label_len);
    let filled = (progress * bar_width as f64).round() as usize;
    let empty = bar_width.saturating_sub(filled);

    let color = progress_color(progress);

    let line = Line::from(vec![
        Span::styled(format!("{} ", label), Style::default().fg(theme::DIM_GRAY)),
        Span::styled("█".repeat(filled), Style::default().fg(color)),
        Span::styled("░".repeat(empty), Style::default().fg(theme::DARK_GRAY)),
    ]);

    let para = Paragraph::new(vec![line]);
    f.render_widget(para, area);
}

pub fn progress_color(progress: f64) -> Color {
    if progress >= 0.9 {
        theme::GREEN
    } else if progress >= 0.5 {
        theme::CYAN
    } else {
        theme::BLUE
    }
}
