// Simulation phases: phase-aware focus layout, prediction gauge, and bill diff.
// Extracted from simulation.rs (plan 4.4.6) — mechanical split, no logic changes.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};
use similar::{ChangeTag, TextDiff};

use crate::theme;
use super::simulation::SimulationScreen;

// ── Phase Layout Types ───────────────────────────────────────────────────────

#[derive(Debug)]
pub(crate) enum PhaseLayout {
    Introduction,
    Committee,
    FloorDebate,
    Amendments,
    FinalArguments,
    Voting,
}

pub(crate) struct FocusConstraints {
    left_pct: u16,
    feed_pct: u16,
    vote_pct: u16,
    amendment_pct: u16,
    show_prediction: bool,
}


pub(crate) fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

// ── Phase methods ──────────────────────────────────────────────────────

impl SimulationScreen {
    pub(crate) fn phase_layout(&self) -> PhaseLayout {
        match self.phase.as_str() {
            "INTRODUCTION" => PhaseLayout::Introduction,
            "COMMITTEE" => PhaseLayout::Committee,
            "FLOOR_DEBATE" => PhaseLayout::FloorDebate,
            "AMENDMENTS" => PhaseLayout::Amendments,
            "FINAL_ARGUMENTS" => PhaseLayout::FinalArguments,
            "VOTING" => PhaseLayout::Voting,
            _ => PhaseLayout::Introduction,
        }
    }
    pub(crate) fn focus_constraints(&self) -> FocusConstraints {
        match self.phase_layout() {
            PhaseLayout::Introduction => FocusConstraints {
                left_pct: 65, feed_pct: 70, vote_pct: 30, amendment_pct: 0, show_prediction: false,
            },
            PhaseLayout::Committee => FocusConstraints {
                left_pct: 50, feed_pct: 55, vote_pct: 25, amendment_pct: 20, show_prediction: false,
            },
            PhaseLayout::FloorDebate => FocusConstraints {
                left_pct: 45, feed_pct: 70, vote_pct: 30, amendment_pct: 0, show_prediction: false,
            },
            PhaseLayout::Amendments => FocusConstraints {
                left_pct: 45, feed_pct: 35, vote_pct: 25, amendment_pct: 40, show_prediction: false,
            },
            PhaseLayout::FinalArguments => FocusConstraints {
                left_pct: 50, feed_pct: 45, vote_pct: 25, amendment_pct: 0, show_prediction: true,
            },
            PhaseLayout::Voting => FocusConstraints {
                left_pct: 25, feed_pct: 30, vote_pct: 50, amendment_pct: 0, show_prediction: true,
            },
        }
    }
    pub(crate) fn draw_focus(&self, f: &mut Frame, area: Rect) {
        let constraints = self.focus_constraints();

        let main_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(constraints.left_pct),
                Constraint::Percentage(100 - constraints.left_pct),
            ])
            .split(area);

        self.draw_agent_pane_stack(f, main_chunks[0]);

        // Right panel: build constraints dynamically
        let mut right_constraints = vec![];
        right_constraints.push(Constraint::Percentage(constraints.feed_pct));
        right_constraints.push(Constraint::Percentage(constraints.vote_pct));
        if constraints.amendment_pct > 0 {
            right_constraints.push(Constraint::Percentage(constraints.amendment_pct));
        }
        if constraints.show_prediction {
            right_constraints.push(Constraint::Length(4)); // prediction gauge
        }

        let right_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(right_constraints)
            .split(main_chunks[1]);

        let mut chunk_idx = 0;
        self.draw_discussion_feed(f, right_chunks[chunk_idx]);
        chunk_idx += 1;
        self.draw_vote_tracker(f, right_chunks[chunk_idx]);
        chunk_idx += 1;
        if constraints.amendment_pct > 0 && chunk_idx < right_chunks.len() {
            if !self.previous_bill_text.is_empty() && self.bill_text != self.previous_bill_text {
                // Split amendment area: top half amendments, bottom half diff
                let amend_chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(right_chunks[chunk_idx]);
                self.draw_amendment_tracker(f, amend_chunks[0]);
                self.draw_bill_diff(f, amend_chunks[1]);
            } else {
                self.draw_amendment_tracker(f, right_chunks[chunk_idx]);
            }
            chunk_idx += 1;
        }
        if constraints.show_prediction && chunk_idx < right_chunks.len() {
            self.draw_prediction_gauge(f, right_chunks[chunk_idx]);
        }
    }
    pub(crate) fn draw_prediction_gauge(&self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .title(" Prediction ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::WARNING));
        let inner = block.inner(area);
        f.render_widget(block, area);

        if inner.width < 10 || inner.height == 0 {
            return;
        }

        let yea_score: f64 = self.agent_sentiment.values().filter(|&&s| s > 0.0).map(|s| s.abs()).sum();
        let nay_score: f64 = self.agent_sentiment.values().filter(|&&s| s < 0.0).map(|s| s.abs()).sum();
        let total = yea_score + nay_score;
        let yea_pct = if total > 0.0 { yea_score / total } else { 0.5 };

        let bar_width = inner.width.saturating_sub(2) as usize;
        let yea_chars = (yea_pct * bar_width as f64).round() as usize;
        let nay_chars = bar_width.saturating_sub(yea_chars);

        let bar = Line::from(vec![
            Span::styled("█".repeat(yea_chars), Style::default().fg(theme::SUCCESS)),
            Span::styled("█".repeat(nay_chars), Style::default().fg(theme::ERROR)),
        ]);
        let label = Line::from(vec![
            Span::styled(format!(" YEA {:.0}%", yea_pct * 100.0), Style::default().fg(theme::SUCCESS)),
            Span::styled(" │ ", Style::default().fg(theme::DIM)),
            Span::styled(format!("NAY {:.0}% ", (1.0 - yea_pct) * 100.0), Style::default().fg(theme::ERROR)),
        ]);

        let para = Paragraph::new(vec![bar, label]).alignment(Alignment::Center);
        f.render_widget(para, inner);
    }
    pub(crate) fn draw_bill_diff(&self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .title(" Bill Text Changes ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::INFO));
        let inner = block.inner(area);
        f.render_widget(block, area);

        if self.previous_bill_text.is_empty() || self.bill_text.is_empty() || inner.height == 0 {
            return;
        }

        let diff = TextDiff::from_lines(&self.previous_bill_text, &self.bill_text);
        let lines: Vec<Line> = diff
            .iter_all_changes()
            .take(inner.height as usize)
            .map(|change| {
                let (sign, color) = match change.tag() {
                    ChangeTag::Delete => ("-", theme::ERROR),
                    ChangeTag::Insert => ("+", theme::SUCCESS),
                    ChangeTag::Equal => (" ", theme::SUBTEXT),
                };
                Line::from(Span::styled(
                    format!("{} {}", sign, change.value().trim_end()),
                    Style::default().fg(color),
                ))
            })
            .collect();

        let para = Paragraph::new(lines);
        f.render_widget(para, inner);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_layout_default_phase() {
        let screen = SimulationScreen::new("t".into(), 10, "m".into());
        let layout = screen.phase_layout();
        match layout {
            super::PhaseLayout::Introduction => {}
            _ => panic!("default WAITING phase should map to Introduction"),
        }
    }
}
