use ratatui::layout::Rect;
use textwrap;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use super::AgentPaneData;
use crate::theme;

pub fn draw_agent_pane(f: &mut Frame, area: Rect, data: &AgentPaneData) {
    let border_color = if data.active {
        theme::GREEN
    } else if data.selected {
        theme::CYAN
    } else {
        theme::DARK_GRAY
    };

    let status = if data.active {
        Span::styled(" [streaming] ", Style::default().fg(theme::GREEN))
    } else if data.latency_ms > 0 {
        Span::styled(
            format!(" [{}ms] ", data.latency_ms),
            Style::default().fg(theme::DIM_GRAY),
        )
    } else {
        Span::styled(" [idle] ", Style::default().fg(theme::DIM_GRAY))
    };

    // Build title line
    let mut title_spans = vec![
        Span::styled(
            data.name.clone(),
            Style::default()
                .fg(theme::CYAN)
                .add_modifier(Modifier::BOLD),
        ),
    ];
    if let Some(ref sub) = data.subtitle {
        title_spans.push(Span::styled(
            format!(" ({})", sub),
            Style::default().fg(theme::DIM_GRAY),
        ));
    }

    // Sentiment indicator
    if let Some(sentiment) = data.sentiment {
        let (indicator, color) = sentiment_indicator(sentiment);
        title_spans.push(Span::styled(
            format!(" {}", indicator),
            Style::default().fg(color),
        ));
    }

    title_spans.push(status);

    let block = Block::default()
        .title(Line::from(title_spans))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height == 0 || inner.width == 0 {
        return;
    }

    let mut lines: Vec<Line> = Vec::new();

    // Sentiment sparkline (if history available)
    if let Some(ref history) = data.sentiment_history {
        if !history.is_empty() {
            let sparkline = render_sentiment_sparkline(history, inner.width as usize);
            lines.push(sparkline);
        }
    }

    // Content: live tokens if active, else last_response
    let text = if data.active && !data.tokens.is_empty() {
        &data.tokens
    } else if !data.last_response.is_empty() {
        &data.last_response
    } else {
        ""
    };

    if !text.is_empty() {
        let content =
            tail_lines(text, inner.height.saturating_sub(lines.len() as u16) as usize, inner.width as usize);
        for line_str in content.lines() {
            lines.push(Line::from(Span::styled(
                line_str.to_string(),
                Style::default().fg(theme::GRAY),
            )));
        }
    }

    let para = Paragraph::new(lines);
    f.render_widget(para, inner);
}

fn sentiment_indicator(score: f64) -> (String, Color) {
    let (prefix, color) = if score > 0.1 {
        ("+", theme::GREEN)
    } else if score < -0.1 {
        ("-", theme::RED)
    } else {
        ("=", theme::YELLOW)
    };
    (format!("{}{:.1}", prefix, score.abs()), color)
}

pub fn render_sentiment_sparkline(history: &[f64], width: usize) -> Line<'static> {
    let chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let display_width = width.min(history.len());
    let start = history.len().saturating_sub(display_width);

    let spans: Vec<Span> = history[start..]
        .iter()
        .map(|&val| {
            // Map -1.0..1.0 to 0..7
            let normalized = ((val + 1.0) / 2.0).clamp(0.0, 1.0);
            let idx = (normalized * 7.0).round() as usize;
            let ch = chars[idx.min(7)];
            let color = if val > 0.1 {
                theme::GREEN
            } else if val < -0.1 {
                theme::RED
            } else {
                theme::YELLOW
            };
            Span::styled(ch.to_string(), Style::default().fg(color))
        })
        .collect();

    Line::from(spans)
}

fn tail_lines(text: &str, max_lines: usize, line_width: usize) -> String {
    if max_lines == 0 || line_width == 0 {
        return String::new();
    }
    let wrapped: Vec<String> = textwrap::wrap(text, line_width)
        .into_iter()
        .map(|cow| cow.into_owned())
        .collect();
    let start = wrapped.len().saturating_sub(max_lines);
    wrapped[start..].join("\n")
}
