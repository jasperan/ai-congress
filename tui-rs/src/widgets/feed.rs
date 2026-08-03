use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

use super::{FeedEntryData, FeedEntryType};
use crate::theme;

pub fn draw_feed(f: &mut Frame, area: Rect, entries: &[FeedEntryData], scroll: u16) {
    let scroll_label = if scroll > 0 {
        format!(" Discussion Feed [SCROLLED -{}] ", scroll)
    } else {
        " Discussion Feed ".to_string()
    };

    let block = Block::default()
        .title(scroll_label)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::INFO));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height == 0 || entries.is_empty() {
        return;
    }

    let visible_count = inner.height as usize;
    let end = entries.len().saturating_sub(scroll as usize);
    let start = end.saturating_sub(visible_count);

    let lines: Vec<Line> = entries[start..end]
        .iter()
        .rev()
        .take(visible_count)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|entry| {
            let (icon, icon_color) = match entry.entry_type {
                FeedEntryType::Speech => (">>", theme::INFO),
                FeedEntryType::Vote => ("##", theme::SUCCESS),
                FeedEntryType::System => ("**", theme::WARNING),
                FeedEntryType::ModelResponse => (">>", theme::INFO),
                FeedEntryType::FinalAnswer => ("★★", theme::INFO),
                FeedEntryType::Lobby => ("$$", theme::SECONDARY),
                FeedEntryType::Filibuster => ("!!", theme::ERROR),
                FeedEntryType::Amendment => ("&&", theme::INFO),
                FeedEntryType::DirectAddress => ("->", theme::SECONDARY),
            };

            let name_color = entry
                .party
                .as_deref()
                .map(|p| theme::party_color(p))
                .unwrap_or(theme::SUBTEXT);

            let max_content = (inner.width as usize).saturating_sub(20);
            let content = if entry.content.len() > max_content {
                format!(
                    "{}...",
                    &entry.content[..max_content.saturating_sub(3)]
                )
            } else {
                entry.content.clone()
            };

            Line::from(vec![
                Span::styled(
                    format!("[{:>3}] ", entry.tick_or_index),
                    Style::default().fg(theme::MUTED),
                ),
                Span::styled(format!("{} ", icon), Style::default().fg(icon_color)),
                Span::styled(
                    format!("{}: ", entry.agent_name),
                    Style::default()
                        .fg(name_color)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(content, Style::default().fg(theme::SUBTEXT)),
            ])
        })
        .collect();

    let para = Paragraph::new(lines);
    f.render_widget(para, inner);
}
