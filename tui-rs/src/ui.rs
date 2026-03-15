use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Frame,
};

use crate::app::{App, FeedEntryType, LayoutMode};
use crate::theme;

/// Main draw function dispatches to Focus or Grid layout.
pub fn draw(f: &mut Frame, app: &App) {
    let size = f.area();

    // Top-level layout: header (3) + body (flex) + status bar (1)
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(1),
        ])
        .split(size);

    draw_header(f, app, chunks[0]);
    draw_status_bar(f, app, chunks[2]);

    match app.layout_mode {
        LayoutMode::Focus => draw_focus(f, app, chunks[1]),
        LayoutMode::Grid => draw_grid(f, app, chunks[1]),
    }
}

// ── Header ──────────────────────────────────────────────────────────────────

fn draw_header(f: &mut Frame, app: &App, area: Rect) {
    let progress = if app.max_ticks > 0 {
        app.current_tick as f64 / app.max_ticks as f64
    } else {
        0.0
    };

    let tps = compute_tps(app);

    let title_line = Line::from(vec![
        Span::styled(" AI CONGRESS ", Style::default().fg(theme::CYAN).add_modifier(Modifier::BOLD)),
        Span::styled(" | ", Style::default().fg(theme::DARK_GRAY)),
        Span::styled(
            format!("Tick {}/{}", app.current_tick, app.max_ticks),
            Style::default().fg(theme::ACCENT),
        ),
        Span::styled(" | ", Style::default().fg(theme::DARK_GRAY)),
        Span::styled(&app.phase_name, Style::default().fg(theme::YELLOW).add_modifier(Modifier::BOLD)),
        Span::styled(" | ", Style::default().fg(theme::DARK_GRAY)),
        Span::styled(&app.model, Style::default().fg(theme::PURPLE)),
        Span::styled(" | ", Style::default().fg(theme::DARK_GRAY)),
        Span::styled(format!("{:.1} tok/s", tps), Style::default().fg(theme::GREEN)),
    ]);

    let topic_line = Line::from(vec![
        Span::styled(" Topic: ", Style::default().fg(theme::DIM_GRAY)),
        Span::styled(truncate(&app.topic, area.width as usize - 10), Style::default().fg(theme::ACCENT)),
    ]);

    let progress_line = Line::from(vec![
        Span::raw(" "),
        Span::styled(
            make_progress_bar(progress, (area.width as usize).saturating_sub(4)),
            Style::default().fg(progress_color(progress)),
        ),
        Span::raw(" "),
    ]);

    let block = Block::default()
        .borders(Borders::BOTTOM)
        .border_style(Style::default().fg(theme::DARK_GRAY));

    let text = vec![title_line, topic_line, progress_line];
    let paragraph = Paragraph::new(text).block(block);
    f.render_widget(paragraph, area);
}

// ── Focus Layout ────────────────────────────────────────────────────────────

fn draw_focus(f: &mut Frame, app: &App, area: Rect) {
    // Split horizontally: agent panes (55%) | right panel (45%)
    let h_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(area);

    // Left: stack of agent panes (show selected + neighbors)
    draw_agent_pane_stack(f, app, h_chunks[0]);

    // Right: feed (65%) + vote tracker (35%)
    let r_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
        .split(h_chunks[1]);

    draw_discussion_feed(f, app, r_chunks[0]);
    draw_vote_tracker(f, app, r_chunks[1]);
}

fn draw_agent_pane_stack(f: &mut Frame, app: &App, area: Rect) {
    if app.agents.is_empty() {
        let block = Block::default()
            .title(" Agents ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::DARK_GRAY));
        let p = Paragraph::new("Waiting for agents...").block(block);
        f.render_widget(p, area);
        return;
    }

    // Show up to 4 agents, centered on selected
    let num_visible = 4.min(app.agents.len());
    let constraints: Vec<Constraint> = (0..num_visible)
        .map(|_| Constraint::Ratio(1, num_visible as u32))
        .collect();

    let pane_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    let start = if app.agents.len() <= num_visible {
        0
    } else if app.selected_agent + num_visible > app.agents.len() {
        app.agents.len() - num_visible
    } else {
        app.selected_agent
    };

    for (i, chunk) in pane_chunks.iter().enumerate() {
        let agent_idx = start + i;
        if agent_idx < app.agents.len() {
            let is_selected = agent_idx == app.selected_agent;
            draw_single_agent_pane(f, app, *chunk, agent_idx, is_selected);
        }
    }
}

fn draw_single_agent_pane(f: &mut Frame, app: &App, area: Rect, agent_idx: usize, is_selected: bool) {
    let agent = &app.agents[agent_idx];
    let stream = app.agent_streams.get(&agent.name);
    let is_active = stream.map(|s| s.active).unwrap_or(false);

    let border_color = if is_active {
        theme::GREEN
    } else if is_selected {
        theme::CYAN
    } else {
        theme::DARK_GRAY
    };

    let party_char = party_abbrev(&agent.party);
    let party_color = theme::party_color(&agent.party);

    let status = if is_active {
        Span::styled(" generating... ", Style::default().fg(theme::GREEN).add_modifier(Modifier::BOLD))
    } else if let Some(s) = stream {
        Span::styled(format!(" idle {}ms ", s.latency_ms), Style::default().fg(theme::DIM_GRAY))
    } else {
        Span::styled(" waiting ", Style::default().fg(theme::DIM_GRAY))
    };

    let title = Line::from(vec![
        Span::styled(
            format!(" {} ", agent.name),
            Style::default().fg(party_color).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("({}{})", party_char, if !agent.state.is_empty() { format!("-{}", state_abbrev(&agent.state)) } else { String::new() }),
            Style::default().fg(party_color),
        ),
        status,
    ]);

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color));

    // Content: show current token stream or last response
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

    // Take last N lines that fit
    let inner_height = area.height.saturating_sub(2) as usize;
    let display_text = tail_lines(&content, inner_height, area.width.saturating_sub(2) as usize);

    let paragraph = Paragraph::new(display_text)
        .block(block)
        .wrap(Wrap { trim: false })
        .style(Style::default().fg(theme::ACCENT));

    f.render_widget(paragraph, area);
}

// ── Grid Layout ─────────────────────────────────────────────────────────────

fn draw_grid(f: &mut Frame, app: &App, area: Rect) {
    if app.agents.is_empty() {
        let block = Block::default()
            .title(" Grid ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::DARK_GRAY));
        let p = Paragraph::new("Waiting for agents...").block(block);
        f.render_widget(p, area);
        return;
    }

    // Split body: grid (80%) + vote summary (20%)
    let v_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(6), Constraint::Length(5)])
        .split(area);

    // 2 rows x 5 cols
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
            if agent_idx < app.agents.len() {
                draw_grid_cell(f, app, cols[col_idx], agent_idx);
            }
        }
    }

    draw_vote_tracker(f, app, v_chunks[1]);
}

fn draw_grid_cell(f: &mut Frame, app: &App, area: Rect, agent_idx: usize) {
    let agent = &app.agents[agent_idx];
    let stream = app.agent_streams.get(&agent.name);
    let is_active = stream.map(|s| s.active).unwrap_or(false);

    let border_color = if is_active { theme::GREEN } else { theme::DARK_GRAY };
    let party_color = theme::party_color(&agent.party);
    let party_char = party_abbrev(&agent.party);

    // Compact name
    let short_name = if agent.name.len() > (area.width as usize).saturating_sub(8) {
        let parts: Vec<&str> = agent.name.split_whitespace().collect();
        if parts.len() >= 2 {
            parts.last().unwrap_or(&"?").to_string()
        } else {
            truncate(&agent.name, area.width as usize - 4).to_string()
        }
    } else {
        agent.name.clone()
    };

    let title = Line::from(vec![
        Span::styled(
            format!(" {} ", short_name),
            Style::default().fg(party_color).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("({})", party_char),
            Style::default().fg(party_color),
        ),
    ]);

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color));

    // Status + snippet
    let inner_w = area.width.saturating_sub(2) as usize;
    let status_line = if is_active {
        Line::from(Span::styled("generating...", Style::default().fg(theme::GREEN)))
    } else if let Some(s) = stream {
        Line::from(Span::styled(
            format!("idle {}ms", s.latency_ms),
            Style::default().fg(theme::DIM_GRAY),
        ))
    } else {
        Line::from(Span::styled("waiting", Style::default().fg(theme::DIM_GRAY)))
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

    // Check vote
    let vote_line = if let Some(v) = app.votes.get(&agent.name) {
        Line::from(Span::styled(
            format!("VOTE: {}", v.vote.to_uppercase()),
            Style::default().fg(theme::vote_color(&v.vote)).add_modifier(Modifier::BOLD),
        ))
    } else {
        Line::default()
    };

    let mut lines = vec![status_line];
    if !snippet.is_empty() {
        lines.push(Line::from(Span::styled(
            truncate(&snippet, inner_w),
            Style::default().fg(theme::ACCENT),
        )));
    }
    lines.push(vote_line);

    let paragraph = Paragraph::new(lines).block(block).wrap(Wrap { trim: true });
    f.render_widget(paragraph, area);
}

// ── Discussion Feed ─────────────────────────────────────────────────────────

fn draw_discussion_feed(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .title(Line::from(vec![
            Span::styled(" Discussion Feed ", Style::default().fg(theme::CYAN).add_modifier(Modifier::BOLD)),
            Span::styled(
                format!(" ({} msgs) ", app.feed.len()),
                Style::default().fg(theme::DIM_GRAY),
            ),
        ]))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::DARK_GRAY));

    if app.feed.is_empty() {
        let p = Paragraph::new("Waiting for discussion to begin...")
            .block(block)
            .style(Style::default().fg(theme::DIM_GRAY));
        f.render_widget(p, area);
        return;
    }

    let inner_height = area.height.saturating_sub(2) as usize;
    let inner_width = area.width.saturating_sub(2) as usize;

    let items: Vec<ListItem> = app
        .feed
        .iter()
        .rev()
        .take(inner_height.max(1))
        .rev()
        .map(|entry| {
            let party_color = theme::party_color(&entry.party);
            let (icon, icon_color) = match entry.entry_type {
                FeedEntryType::Speech => (">>", theme::ACCENT),
                FeedEntryType::Vote => ("##", theme::YELLOW),
                FeedEntryType::System => ("**", theme::DIM_GRAY),
            };

            let content_max = inner_width.saturating_sub(entry.agent_name.len() + 12);
            let content_str = truncate(&entry.content, content_max);

            let line = Line::from(vec![
                Span::styled(
                    format!("[T{:>3}] ", entry.tick),
                    Style::default().fg(theme::DIM_GRAY),
                ),
                Span::styled(format!("{} ", icon), Style::default().fg(icon_color)),
                Span::styled(
                    format!("{}: ", entry.agent_name),
                    Style::default().fg(party_color).add_modifier(Modifier::BOLD),
                ),
                Span::styled(content_str, Style::default().fg(theme::GRAY)),
            ]);

            ListItem::new(line)
        })
        .collect();

    let list = List::new(items).block(block);
    f.render_widget(list, area);
}

// ── Vote Tracker ────────────────────────────────────────────────────────────

fn draw_vote_tracker(f: &mut Frame, app: &App, area: Rect) {
    let total_cast = app.yea_count + app.nay_count + app.abstain_count;
    let pending = app.agents.len() as u32 - total_cast.min(app.agents.len() as u32);

    let block = Block::default()
        .title(Line::from(vec![
            Span::styled(" Vote Tracker ", Style::default().fg(theme::YELLOW).add_modifier(Modifier::BOLD)),
        ]))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::DARK_GRAY));

    let mut lines: Vec<Line> = Vec::new();

    // Summary line
    lines.push(Line::from(vec![
        Span::styled(" YEA: ", Style::default().fg(theme::DIM_GRAY)),
        Span::styled(
            format!("{}", app.yea_count),
            Style::default().fg(theme::GREEN).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  NAY: ", Style::default().fg(theme::DIM_GRAY)),
        Span::styled(
            format!("{}", app.nay_count),
            Style::default().fg(theme::RED).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ABSTAIN: ", Style::default().fg(theme::DIM_GRAY)),
        Span::styled(
            format!("{}", app.abstain_count),
            Style::default().fg(theme::YELLOW).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("  PENDING: {}", pending),
            Style::default().fg(theme::DIM_GRAY),
        ),
    ]));

    // Individual votes
    if !app.votes.is_empty() {
        let mut vote_spans: Vec<Span> = Vec::new();
        vote_spans.push(Span::raw(" "));

        for (name, vote) in &app.votes {
            let party = app
                .agents
                .iter()
                .find(|a| a.name == *name)
                .map(|a| party_abbrev(&a.party))
                .unwrap_or("?");
            let short = name.split_whitespace().last().unwrap_or(name);
            let vc = theme::vote_color(&vote.vote);

            vote_spans.push(Span::styled(
                format!("{}({}): ", short, party),
                Style::default().fg(theme::GRAY),
            ));
            vote_spans.push(Span::styled(
                format!("{} ", vote.vote.to_uppercase()),
                Style::default().fg(vc).add_modifier(Modifier::BOLD),
            ));
        }

        lines.push(Line::from(vote_spans));
    }

    // Result if complete
    if let Some(ref result) = app.simulation_result {
        let result_color = if result.contains("PASSED") {
            theme::GREEN
        } else if result.contains("FAILED") {
            theme::RED
        } else {
            theme::YELLOW
        };

        lines.push(Line::from(vec![
            Span::styled(
                format!(" RESULT: {} ", result),
                Style::default().fg(result_color).add_modifier(Modifier::BOLD),
            ),
        ]));
    }

    let paragraph = Paragraph::new(lines).block(block).wrap(Wrap { trim: true });
    f.render_widget(paragraph, area);
}

// ── Status Bar ──────────────────────────────────────────────────────────────

fn draw_status_bar(f: &mut Frame, app: &App, area: Rect) {
    let sparkline = render_sparkline(&app.throughput_buckets, 20);
    let tps = compute_tps(app);

    let status = if app.running { "LIVE" } else { "DONE" };
    let status_color = if app.running { theme::GREEN } else { theme::DIM_GRAY };

    let line = Line::from(vec![
        Span::styled(
            format!(" {} ", status),
            Style::default().fg(Color::Black).bg(status_color).add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
        Span::styled(sparkline, Style::default().fg(theme::CYAN)),
        Span::styled(format!(" {:.1} tok/s", tps), Style::default().fg(theme::ACCENT)),
        Span::styled(
            format!("  {} total", app.total_tokens),
            Style::default().fg(theme::DIM_GRAY),
        ),
        Span::raw("  "),
        Span::styled("Tab", Style::default().fg(theme::CYAN).add_modifier(Modifier::BOLD)),
        Span::styled(":layout ", Style::default().fg(theme::GRAY)),
        Span::styled("j/k", Style::default().fg(theme::CYAN).add_modifier(Modifier::BOLD)),
        Span::styled(":agent ", Style::default().fg(theme::GRAY)),
        Span::styled("q", Style::default().fg(theme::CYAN).add_modifier(Modifier::BOLD)),
        Span::styled(":quit", Style::default().fg(theme::GRAY)),
    ]);

    let paragraph = Paragraph::new(line);
    f.render_widget(paragraph, area);
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn compute_tps(app: &App) -> f64 {
    // Average of last 5 seconds
    let recent: u32 = app.throughput_buckets.iter().rev().take(5).sum();
    recent as f64 / 5.0
}

fn render_sparkline(buckets: &std::collections::VecDeque<u32>, width: usize) -> String {
    let chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let visible: Vec<u32> = buckets.iter().rev().take(width).rev().copied().collect();
    let max_val = *visible.iter().max().unwrap_or(&1).max(&1);

    visible
        .iter()
        .map(|&v| {
            let idx = ((v as f64 / max_val as f64) * 7.0) as usize;
            chars[idx.min(7)]
        })
        .collect()
}

fn make_progress_bar(progress: f64, width: usize) -> String {
    let filled = (progress * width as f64) as usize;
    let empty = width.saturating_sub(filled);
    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}

fn progress_color(progress: f64) -> Color {
    if progress >= 0.9 {
        theme::GREEN
    } else if progress >= 0.5 {
        theme::CYAN
    } else {
        theme::BLUE
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len > 3 {
        format!("{}...", &s[..max_len - 3])
    } else {
        s[..max_len].to_string()
    }
}

fn party_abbrev(party: &str) -> &str {
    match party.to_uppercase().as_str() {
        "REPUBLICAN" => "R",
        "DEMOCRATIC" | "DEMOCRAT" => "D",
        "INDEPENDENT" => "I",
        _ => "?",
    }
}

fn state_abbrev(state: &str) -> String {
    // Common US state abbreviations
    match state {
        "Alabama" => "AL", "Alaska" => "AK", "Arizona" => "AZ", "Arkansas" => "AR",
        "California" => "CA", "Colorado" => "CO", "Connecticut" => "CT", "Delaware" => "DE",
        "Florida" => "FL", "Georgia" => "GA", "Hawaii" => "HI", "Idaho" => "ID",
        "Illinois" => "IL", "Indiana" => "IN", "Iowa" => "IA", "Kansas" => "KS",
        "Kentucky" => "KY", "Louisiana" => "LA", "Maine" => "ME", "Maryland" => "MD",
        "Massachusetts" => "MA", "Michigan" => "MI", "Minnesota" => "MN", "Mississippi" => "MS",
        "Missouri" => "MO", "Montana" => "MT", "Nebraska" => "NE", "Nevada" => "NV",
        "New Hampshire" => "NH", "New Jersey" => "NJ", "New Mexico" => "NM", "New York" => "NY",
        "North Carolina" => "NC", "North Dakota" => "ND", "Ohio" => "OH", "Oklahoma" => "OK",
        "Oregon" => "OR", "Pennsylvania" => "PA", "Rhode Island" => "RI",
        "South Carolina" => "SC", "South Dakota" => "SD", "Tennessee" => "TN", "Texas" => "TX",
        "Utah" => "UT", "Vermont" => "VT", "Virginia" => "VA", "Washington" => "WA",
        "West Virginia" => "WV", "Wisconsin" => "WI", "Wyoming" => "WY",
        _ => return state[..2.min(state.len())].to_uppercase(),
    }
    .to_string()
}

fn tail_lines(text: &str, max_lines: usize, line_width: usize) -> String {
    if text.is_empty() || max_lines == 0 {
        return String::new();
    }

    // Simple word-wrap then take tail
    let mut wrapped: Vec<String> = Vec::new();
    for line in text.lines() {
        if line.is_empty() {
            wrapped.push(String::new());
            continue;
        }
        let mut current = String::new();
        for word in line.split_whitespace() {
            if current.is_empty() {
                current = word.to_string();
            } else if current.len() + 1 + word.len() <= line_width {
                current.push(' ');
                current.push_str(word);
            } else {
                wrapped.push(current);
                current = word.to_string();
            }
        }
        if !current.is_empty() {
            wrapped.push(current);
        }
    }

    let start = wrapped.len().saturating_sub(max_lines);
    wrapped[start..].join("\n")
}
