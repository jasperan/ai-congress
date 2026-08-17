// Network-layout rendering: agent pane stack, grid, and discussion feed.
// Extracted from simulation.rs (plan 4.4.6) — mechanical split, no logic changes.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Frame,
};

use rattles::presets::prelude as spinners;

use crate::theme;
use super::sim_sentiment::{sentiment_indicator, tail_lines};
use super::simulation::{party_abbrev, state_abbrev, truncate, FeedEntryType, SimulationScreen};

impl SimulationScreen {
    pub(crate) fn draw_agent_pane_stack(&self, f: &mut Frame, area: Rect) {
        if self.agents.is_empty() {
            let block = Block::default()
                .title(" Agents ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::DIM));
            let p = Paragraph::new("Waiting for agents...").block(block);
            f.render_widget(p, area);
            return;
        }

        let num_visible = 4.min(self.agents.len());
        let constraints: Vec<Constraint> = (0..num_visible)
            .map(|_| Constraint::Ratio(1, num_visible as u32))
            .collect();

        let pane_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(area);

        let start = if self.agents.len() <= num_visible {
            0
        } else if self.selected_agent + num_visible > self.agents.len() {
            self.agents.len() - num_visible
        } else {
            self.selected_agent
        };

        for (i, chunk) in pane_chunks.iter().enumerate() {
            let agent_idx = start + i;
            if agent_idx < self.agents.len() {
                let is_selected = agent_idx == self.selected_agent;
                self.draw_single_agent_pane(f, *chunk, agent_idx, is_selected);
            }
        }
    }
    pub(crate) fn draw_single_agent_pane(
        &self,
        f: &mut Frame,
        area: Rect,
        agent_idx: usize,
        is_selected: bool,
    ) {
        let agent = &self.agents[agent_idx];
        let stream = self.agent_streams.get(&agent.name);
        let is_active = stream.map(|s| s.active).unwrap_or(false);
        let sentiment = self.agent_sentiment.get(&agent.name).copied().unwrap_or(0.0);

        let border_color = if is_active {
            theme::SUCCESS
        } else if is_selected {
            theme::INFO
        } else {
            theme::DIM
        };

        let party_char = party_abbrev(&agent.party);
        let party_color = theme::party_color(&agent.party);
        let sentiment_display = sentiment_indicator(sentiment);

        let status = if is_active {
            let frame = spinners::dots().current_frame();
            Span::styled(
                format!(" {} generating... ", frame),
                Style::default()
                    .fg(theme::SUCCESS)
                    .add_modifier(Modifier::BOLD),
            )
        } else if let Some(s) = stream {
            Span::styled(
                format!(" idle {}ms ", s.latency_ms),
                Style::default().fg(theme::MUTED),
            )
        } else {
            Span::styled(" waiting ", Style::default().fg(theme::MUTED))
        };

        let title = Line::from(vec![
            Span::styled(
                format!(" {} ", agent.name),
                Style::default()
                    .fg(party_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(
                    "({}{})",
                    party_char,
                    if !agent.state.is_empty() {
                        format!("-{}", state_abbrev(&agent.state))
                    } else {
                        String::new()
                    }
                ),
                Style::default().fg(party_color),
            ),
            Span::styled(
                format!(" [{}] ", sentiment_display.0),
                Style::default().fg(sentiment_display.1),
            ),
            status,
        ]);

        let block = Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color));

        let content = if let Some(s) = stream {
            if s.active && !s.tokens.is_empty() {
                s.tokens.clone()
            } else if !s.last_response.is_empty() {
                s.last_response.clone()
            } else {
                String::from("(no output yet)")
            }
        } else {
            String::from("(waiting)")
        };

        let inner_height = area.height.saturating_sub(2) as usize;
        let display_text =
            tail_lines(&content, inner_height, area.width.saturating_sub(2) as usize);

        let paragraph = Paragraph::new(display_text)
            .block(block)
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(theme::INFO));

        f.render_widget(paragraph, area);
    }
    pub(crate) fn draw_grid(&self, f: &mut Frame, area: Rect) {
        if self.agents.is_empty() {
            let block = Block::default()
                .title(" Grid ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::DIM));
            let p = Paragraph::new("Waiting for agents...").block(block);
            f.render_widget(p, area);
            return;
        }

        let v_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(6), Constraint::Length(5)])
            .split(area);

        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(v_chunks[0]);

        for row_idx in 0..2 {
            let cols = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Percentage(20),
                    Constraint::Percentage(20),
                    Constraint::Percentage(20),
                    Constraint::Percentage(20),
                    Constraint::Percentage(20),
                ])
                .split(rows[row_idx]);

            for col_idx in 0..5 {
                let agent_idx = row_idx * 5 + col_idx;
                if agent_idx < self.agents.len() {
                    self.draw_grid_cell(f, cols[col_idx], agent_idx);
                }
            }
        }

        self.draw_vote_tracker(f, v_chunks[1]);
    }
    pub(crate) fn draw_grid_cell(&self, f: &mut Frame, area: Rect, agent_idx: usize) {
        let agent = &self.agents[agent_idx];
        let stream = self.agent_streams.get(&agent.name);
        let is_active = stream.map(|s| s.active).unwrap_or(false);
        let sentiment = self.agent_sentiment.get(&agent.name).copied().unwrap_or(0.0);

        let border_color = if is_active {
            theme::SUCCESS
        } else {
            theme::DIM
        };
        let party_color = theme::party_color(&agent.party);
        let party_char = party_abbrev(&agent.party);

        let short_name = if agent.name.len() > (area.width as usize).saturating_sub(12) {
            let parts: Vec<&str> = agent.name.split_whitespace().collect();
            if parts.len() >= 2 {
                parts.last().unwrap_or(&"?").to_string()
            } else {
                truncate(&agent.name, area.width as usize - 4).to_string()
            }
        } else {
            agent.name.clone()
        };

        let sent_display = sentiment_indicator(sentiment);

        let title = Line::from(vec![
            Span::styled(
                format!(" {} ", short_name),
                Style::default()
                    .fg(party_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("({}) ", party_char),
                Style::default().fg(party_color),
            ),
            Span::styled(sent_display.0.clone(), Style::default().fg(sent_display.1)),
        ]);

        let block = Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color));

        let inner_w = area.width.saturating_sub(2) as usize;
        let status_line = if is_active {
            Line::from(Span::styled(
                "generating...",
                Style::default().fg(theme::SUCCESS),
            ))
        } else if let Some(s) = stream {
            Line::from(Span::styled(
                format!("idle {}ms", s.latency_ms),
                Style::default().fg(theme::MUTED),
            ))
        } else {
            Line::from(Span::styled(
                "waiting",
                Style::default().fg(theme::MUTED),
            ))
        };

        let snippet = if let Some(s) = stream {
            let text = if s.active { &s.tokens } else { &s.last_response };
            if text.is_empty() {
                String::new()
            } else {
                truncate(text, inner_w * 2).to_string()
            }
        } else {
            String::new()
        };

        let vote_line = if let Some(v) = self.votes.get(&agent.name) {
            Line::from(Span::styled(
                format!("VOTE: {}", v.vote.to_uppercase()),
                Style::default()
                    .fg(theme::vote_color(&v.vote))
                    .add_modifier(Modifier::BOLD),
            ))
        } else {
            Line::default()
        };

        let mut lines = vec![status_line];
        if !snippet.is_empty() {
            lines.push(Line::from(Span::styled(
                truncate(&snippet, inner_w),
                Style::default().fg(theme::INFO),
            )));
        }
        lines.push(vote_line);

        let paragraph = Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: true });
        f.render_widget(paragraph, area);
    }
    pub(crate) fn draw_discussion_feed(&self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .title(Line::from(vec![
                Span::styled(
                    " Discussion Feed ",
                    Style::default()
                        .fg(theme::INFO)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(" ({} msgs) ", self.feed.len()),
                    Style::default().fg(theme::MUTED),
                ),
            ]))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::DIM));

        if self.feed.is_empty() {
            let p = Paragraph::new("Waiting for discussion to begin...")
                .block(block)
                .style(Style::default().fg(theme::MUTED));
            f.render_widget(p, area);
            return;
        }

        let inner_height = area.height.saturating_sub(2) as usize;
        let inner_width = area.width.saturating_sub(2) as usize;

        let items: Vec<ListItem> = self
            .feed
            .iter()
            .rev()
            .take(inner_height.max(1))
            .rev()
            .map(|entry| {
                let party_color = theme::party_color(&entry.party);
                let (icon, icon_color) = match entry.entry_type {
                    FeedEntryType::Speech => (">>", theme::INFO),
                    FeedEntryType::Vote => ("##", theme::WARNING),
                    FeedEntryType::System => ("**", theme::MUTED),
                    FeedEntryType::Lobby => ("$$", theme::SECONDARY),
                    FeedEntryType::Filibuster => ("!!", theme::ERROR),
                    FeedEntryType::Amendment => ("&&", theme::INFO),
                    FeedEntryType::DirectAddress => ("->", theme::SECONDARY),
                };

                let content_max =
                    inner_width.saturating_sub(entry.agent_name.len() + 12);
                let content_str = truncate(&entry.content, content_max);

                let line = Line::from(vec![
                    Span::styled(
                        format!("[T{:>3}] ", entry.tick),
                        Style::default().fg(theme::MUTED),
                    ),
                    Span::styled(format!("{} ", icon), Style::default().fg(icon_color)),
                    Span::styled(
                        format!("{}: ", entry.agent_name),
                        Style::default()
                            .fg(party_color)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(content_str, Style::default().fg(theme::SUBTEXT)),
                ]);

                ListItem::new(line)
            })
            .collect();

        let list = List::new(items).block(block);
        f.render_widget(list, area);
    }
}
