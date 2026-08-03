use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

use super::AmendmentData;
use crate::theme;

pub fn draw_amendment_tracker(f: &mut Frame, area: Rect, amendments: &[AmendmentData]) {
    let block = Block::default()
        .title(" Amendments ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::INFO));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height == 0 || amendments.is_empty() {
        return;
    }

    let lines: Vec<Line> = amendments
        .iter()
        .take(inner.height as usize)
        .map(|a| {
            let status_color = match a.status.as_str() {
                "passed" => theme::SUCCESS,
                "failed" => theme::ERROR,
                _ => theme::WARNING,
            };
            let proposer_last = a.proposer.split_whitespace().last().unwrap_or(&a.proposer);
            let max_text = (inner.width as usize).saturating_sub(30);
            let text = if a.text.len() > max_text {
                format!("{}...", &a.text[..max_text.saturating_sub(3)])
            } else {
                a.text.clone()
            };

            let mut spans = vec![
                Span::styled(format!("#{} ", a.id), Style::default().fg(theme::MUTED)),
                Span::styled(
                    format!("[{}] ", a.status.to_uppercase()),
                    Style::default()
                        .fg(status_color)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("by {} ", proposer_last),
                    Style::default().fg(theme::INFO),
                ),
                Span::styled(text, Style::default().fg(theme::SUBTEXT)),
            ];

            if a.status != "pending" {
                spans.push(Span::styled(
                    format!(" {}-{}", a.yea, a.nay),
                    Style::default().fg(theme::MUTED),
                ));
            }

            Line::from(spans)
        })
        .collect();

    let para = Paragraph::new(lines);
    f.render_widget(para, inner);
}
