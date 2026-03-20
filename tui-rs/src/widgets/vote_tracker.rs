use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

use super::VoteTrackerData;
use crate::theme;

pub fn draw_vote_tracker(f: &mut Frame, area: Rect, data: &VoteTrackerData) {
    let block = Block::default()
        .title(" Vote Tracker ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::CYAN));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height == 0 {
        return;
    }

    let mut lines: Vec<Line> = Vec::new();

    // Line 1: Tally
    let pending = data
        .total_agents
        .saturating_sub((data.yea + data.nay + data.abstain) as usize);
    lines.push(Line::from(vec![
        Span::styled("YEA: ", Style::default().fg(theme::DIM_GRAY)),
        Span::styled(
            format!("{}", data.yea),
            Style::default()
                .fg(theme::GREEN)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  NAY: ", Style::default().fg(theme::DIM_GRAY)),
        Span::styled(
            format!("{}", data.nay),
            Style::default()
                .fg(theme::RED)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ABSTAIN: ", Style::default().fg(theme::DIM_GRAY)),
        Span::styled(
            format!("{}", data.abstain),
            Style::default().fg(theme::YELLOW),
        ),
        Span::styled(
            format!("  PENDING: {}", pending),
            Style::default().fg(theme::DIM_GRAY),
        ),
    ]));

    // Line 2: Per-agent votes
    if !data.per_agent_votes.is_empty() {
        let vote_spans: Vec<Span> = data
            .per_agent_votes
            .iter()
            .map(|(name, party, vote)| {
                let color = theme::vote_color(vote);
                let last_name = name.split_whitespace().last().unwrap_or(name);
                let p = party.chars().next().unwrap_or(' ');
                Span::styled(
                    format!("{}({}):{} ", last_name, p, vote.to_uppercase()),
                    Style::default().fg(color),
                )
            })
            .collect();
        lines.push(Line::from(vote_spans));
    }

    // Confidence bar (for chat mode)
    if let Some(conf) = data.confidence {
        let bar_width = inner.width.saturating_sub(15) as usize;
        let filled = (conf / 100.0 * bar_width as f64).round() as usize;
        let empty = bar_width.saturating_sub(filled);
        lines.push(Line::from(vec![
            Span::styled("Confidence: ", Style::default().fg(theme::DIM_GRAY)),
            Span::styled(
                "█".repeat(filled),
                Style::default().fg(theme::GREEN),
            ),
            Span::styled(
                "░".repeat(empty),
                Style::default().fg(theme::DARK_GRAY),
            ),
            Span::styled(
                format!(" {:.1}%", conf),
                Style::default().fg(theme::CYAN),
            ),
        ]));
    }

    // Weight bars (for chat mode)
    if let Some(ref breakdown) = data.vote_breakdown {
        for (name, weight) in breakdown {
            let bar_width = 20usize;
            let filled = (weight * bar_width as f64).round() as usize;
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{:>12} ", name),
                    Style::default().fg(theme::ACCENT),
                ),
                Span::styled(
                    "█".repeat(filled),
                    Style::default().fg(theme::BLUE),
                ),
                Span::styled(
                    format!(" {:.2}", weight),
                    Style::default().fg(theme::DIM_GRAY),
                ),
            ]));
        }
    }

    // Persuasion edges (top 3)
    if !data.persuasion_edges.is_empty() {
        let mut sorted = data.persuasion_edges.clone();
        sorted.sort_by(|a, b| {
            b.2.partial_cmp(&a.2)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_edges: Vec<Span> = sorted
            .iter()
            .take(3)
            .map(|(from, to, strength)| {
                let from_last = from.split_whitespace().last().unwrap_or(from);
                let to_last = to.split_whitespace().last().unwrap_or(to);
                Span::styled(
                    format!("{}>{}({:.2}) ", from_last, to_last, strength),
                    Style::default().fg(theme::PURPLE),
                )
            })
            .collect();
        if !top_edges.is_empty() {
            lines.push(Line::from(top_edges));
        }
    }

    // Result banner
    if let Some(ref result) = data.result {
        let color = match result.to_uppercase().as_str() {
            r if r.contains("PASSED") || r.contains("YEA") => theme::GREEN,
            r if r.contains("FAILED") || r.contains("NAY") => theme::RED,
            _ => theme::YELLOW,
        };
        lines.push(Line::from(Span::styled(
            format!("RESULT: {}", result),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        )));
    }

    let para = Paragraph::new(lines);
    f.render_widget(para, inner);
}
