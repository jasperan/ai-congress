use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
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
        .border_style(Style::default().fg(theme::CYAN));

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
                FeedEntryType::Speech => (">>", theme::ACCENT),
                FeedEntryType::Vote => ("##", theme::GREEN),
                FeedEntryType::System => ("**", theme::YELLOW),
                FeedEntryType::ModelResponse => (">>", theme::ACCENT),
                FeedEntryType::FinalAnswer => ("★★", theme::CYAN),
                FeedEntryType::Lobby => ("$$", theme::PURPLE),
                FeedEntryType::Filibuster => ("!!", theme::RED),
                FeedEntryType::Amendment => ("&&", theme::CYAN),
                FeedEntryType::DirectAddress => ("->", Color::Magenta),
            };

            let name_color = entry
                .party
                .as_deref()
                .map(|p| theme::party_color(p))
                .unwrap_or(theme::GRAY);

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
                    Style::default().fg(theme::DIM_GRAY),
                ),
                Span::styled(format!("{} ", icon), Style::default().fg(icon_color)),
                Span::styled(
                    format!("{}: ", entry.agent_name),
                    Style::default()
                        .fg(name_color)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(content, Style::default().fg(theme::GRAY)),
            ])
        })
        .collect();

    let para = Paragraph::new(lines);
    f.render_widget(para, inner);
}
