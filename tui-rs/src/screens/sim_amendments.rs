// Vote tracker and amendment tracker rendering.
// Extracted from simulation.rs (plan 4.4.6) — mechanical split, no logic changes.

use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use crate::theme;
use super::simulation::{party_abbrev, truncate, SimulationScreen};

impl SimulationScreen {
    pub(crate) fn draw_vote_tracker(&self, f: &mut Frame, area: Rect) {
        let total_cast = self.yea_count + self.nay_count + self.abstain_count;
        let pending =
            self.agents.len() as u32 - total_cast.min(self.agents.len() as u32);

        let block = Block::default()
            .title(Line::from(vec![Span::styled(
                " Vote Tracker ",
                Style::default()
                    .fg(theme::WARNING)
                    .add_modifier(Modifier::BOLD),
            )]))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::DIM));

        let mut lines: Vec<Line> = Vec::new();

        lines.push(Line::from(vec![
            Span::styled(" YEA: ", Style::default().fg(theme::MUTED)),
            Span::styled(
                format!("{}", self.yea_count),
                Style::default()
                    .fg(theme::SUCCESS)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  NAY: ", Style::default().fg(theme::MUTED)),
            Span::styled(
                format!("{}", self.nay_count),
                Style::default()
                    .fg(theme::ERROR)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  ABSTAIN: ", Style::default().fg(theme::MUTED)),
            Span::styled(
                format!("{}", self.abstain_count),
                Style::default()
                    .fg(theme::WARNING)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  PENDING: {}", pending),
                Style::default().fg(theme::MUTED),
            ),
        ]));

        if !self.votes.is_empty() {
            let mut vote_spans: Vec<Span> = Vec::new();
            vote_spans.push(Span::raw(" "));

            for (name, vote) in &self.votes {
                let party = self
                    .agents
                    .iter()
                    .find(|a| a.name == *name)
                    .map(|a| party_abbrev(&a.party))
                    .unwrap_or("?");
                let short = name.split_whitespace().last().unwrap_or(name);
                let vc = theme::vote_color(&vote.vote);

                vote_spans.push(Span::styled(
                    format!("{}({}): ", short, party),
                    Style::default().fg(theme::SUBTEXT),
                ));
                vote_spans.push(Span::styled(
                    format!("{} ", vote.vote.to_uppercase()),
                    Style::default().fg(vc).add_modifier(Modifier::BOLD),
                ));
            }

            lines.push(Line::from(vote_spans));
        }

        if !self.persuasion_edges.is_empty() {
            let mut persu_spans = vec![Span::styled(
                " Influence: ",
                Style::default().fg(theme::MUTED),
            )];
            let mut sorted_edges = self.persuasion_edges.clone();
            sorted_edges
                .sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            for (inf, infd, str_val) in sorted_edges.iter().take(3) {
                let inf_short = inf.split_whitespace().last().unwrap_or(inf);
                let infd_short = infd.split_whitespace().last().unwrap_or(infd);
                persu_spans.push(Span::styled(
                    format!("{}>{} ({:.2}) ", inf_short, infd_short, str_val),
                    Style::default().fg(theme::SECONDARY),
                ));
            }
            lines.push(Line::from(persu_spans));
        }

        if let Some(ref result) = self.simulation_result {
            let result_color = if result.contains("PASSED") {
                theme::SUCCESS
            } else if result.contains("FAILED") {
                theme::ERROR
            } else {
                theme::WARNING
            };

            lines.push(Line::from(vec![Span::styled(
                format!(" RESULT: {} ", result),
                Style::default()
                    .fg(result_color)
                    .add_modifier(Modifier::BOLD),
            )]));
        }

        let paragraph = Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: true });
        f.render_widget(paragraph, area);
    }
    pub(crate) fn draw_amendment_tracker(&self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .title(Line::from(vec![Span::styled(
                " Amendments ",
                Style::default()
                    .fg(theme::INFO)
                    .add_modifier(Modifier::BOLD),
            )]))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::DIM));

        let mut lines: Vec<Line> = Vec::new();

        for amend in &self.amendments {
            let status_color = match amend.status.as_str() {
                "passed" => theme::SUCCESS,
                "failed" => theme::ERROR,
                _ => theme::WARNING,
            };

            let proposer_short = amend
                .proposer
                .split_whitespace()
                .last()
                .unwrap_or(&amend.proposer);
            let inner_w = area.width.saturating_sub(4) as usize;

            lines.push(Line::from(vec![
                Span::styled(
                    format!(" #{} ", amend.id),
                    Style::default()
                        .fg(theme::INFO)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("[{}] ", amend.status.to_uppercase()),
                    Style::default()
                        .fg(status_color)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("by {} ", proposer_short),
                    Style::default().fg(theme::MUTED),
                ),
                Span::styled(
                    truncate(&amend.text, inner_w.saturating_sub(30)),
                    Style::default().fg(theme::SUBTEXT),
                ),
            ]));

            if amend.yea > 0 || amend.nay > 0 {
                lines.push(Line::from(vec![
                    Span::raw("   "),
                    Span::styled(
                        format!("YEA: {} ", amend.yea),
                        Style::default().fg(theme::SUCCESS),
                    ),
                    Span::styled(
                        format!("NAY: {}", amend.nay),
                        Style::default().fg(theme::ERROR),
                    ),
                ]));
            }
        }

        if lines.is_empty() {
            lines.push(Line::from(Span::styled(
                " No amendments proposed yet",
                Style::default().fg(theme::MUTED),
            )));
        }

        let paragraph = Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: true });
        f.render_widget(paragraph, area);
    }
}
