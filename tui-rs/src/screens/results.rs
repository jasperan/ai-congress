use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use crate::theme;
use super::{Action, ScreenId, SessionResult};

pub struct ResultsScreen {
    pub result: SessionResult,
    pub scroll: u16,
    pub export_message: Option<String>,
}

impl ResultsScreen {
    pub fn new(result: SessionResult) -> Self {
        Self {
            result,
            scroll: 0,
            export_message: None,
        }
    }

    // ── Key handler ──────────────────────────────────────────────────────────

    pub fn handle_key(&mut self, key: KeyEvent) -> Action {
        match key.code {
            KeyCode::Esc | KeyCode::Char('b') => {
                return Action::SwitchScreen(ScreenId::Models);
            }
            KeyCode::Char('n') => {
                return Action::SwitchScreen(ScreenId::ModeSelect {
                    selected_models: Vec::new(),
                });
            }
            KeyCode::Char('e') => {
                self.export_markdown();
            }
            KeyCode::Char('j') | KeyCode::Down => {
                self.scroll = self.scroll.saturating_add(1);
            }
            KeyCode::Char('k') | KeyCode::Up => {
                self.scroll = self.scroll.saturating_sub(1);
            }
            _ => {}
        }
        Action::None
    }

    // ── Export ───────────────────────────────────────────────────────────────

    pub fn export_markdown(&mut self) {
        let timestamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
        let filename = format!("./congress-export-{}.md", timestamp);

        let content = self.build_markdown();
        match std::fs::write(&filename, &content) {
            Ok(_) => {
                self.export_message = Some(format!("Exported to {}", filename));
            }
            Err(e) => {
                self.export_message = Some(format!("Export failed: {}", e));
            }
        }
    }

    fn build_markdown(&self) -> String {
        let mut md = String::new();
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");

        match &self.result {
            SessionResult::Chat {
                final_answer,
                confidence,
                vote_breakdown,
                responses,
            } => {
                md.push_str("# AI Congress Chat Session\n\n");
                md.push_str(&format!("*Exported: {}*\n\n", timestamp));

                md.push_str("## Final Answer\n\n");
                md.push_str(final_answer);
                md.push_str("\n\n");

                md.push_str(&format!(
                    "**Confidence:** {:.1}%\n\n",
                    confidence * 100.0
                ));

                if let Some(obj) = vote_breakdown.as_object() {
                    if !obj.is_empty() {
                        md.push_str("## Vote Breakdown\n\n");
                        for (model, weight) in obj {
                            md.push_str(&format!(
                                "- **{}**: {:.3}\n",
                                model,
                                weight.as_f64().unwrap_or(0.0)
                            ));
                        }
                        md.push('\n');
                    }
                }

                if !responses.is_empty() {
                    md.push_str("## Model Responses\n\n");
                    for (model, response) in responses {
                        md.push_str(&format!("### {}\n\n", model));
                        md.push_str(response);
                        md.push_str("\n\n");
                    }
                }
            }

            SessionResult::Simulation {
                result,
                yea,
                nay,
                abstain,
                amendments,
                historical_accuracy,
            } => {
                md.push_str("# AI Congress Simulation Session\n\n");
                md.push_str(&format!("*Exported: {}*\n\n", timestamp));

                md.push_str(&format!("## Result: {}\n\n", result));
                md.push_str(&format!(
                    "**YEA:** {}  **NAY:** {}  **ABSTAIN:** {}\n\n",
                    yea, nay, abstain
                ));

                if let Some(acc) = historical_accuracy {
                    md.push_str(&format!("**Historical Accuracy:** {:.1}%\n\n", acc * 100.0));
                }

                if !amendments.is_empty() {
                    md.push_str("## Amendments\n\n");
                    for (i, amendment) in amendments.iter().enumerate() {
                        md.push_str(&format!("### Amendment {}\n\n", i + 1));
                        md.push_str(&format!("{}\n\n", amendment));
                    }
                }
            }
        }

        md
    }

    // ── Draw ─────────────────────────────────────────────────────────────────

    pub fn draw(&self, f: &mut Frame, area: Rect) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(0),
                Constraint::Length(1),
            ])
            .split(area);

        self.draw_header(f, rows[0]);
        self.draw_body(f, rows[1]);
        self.draw_footer(f, rows[2]);
    }

    fn draw_header(&self, f: &mut Frame, area: Rect) {
        let (title, color) = match &self.result {
            SessionResult::Chat { .. } => ("CHAT RESULTS", theme::CYAN),
            SessionResult::Simulation { result, .. } => {
                let c = if result.to_uppercase().contains("PASSED") {
                    theme::GREEN
                } else if result.to_uppercase().contains("FAILED") {
                    theme::RED
                } else {
                    theme::YELLOW
                };
                ("SIMULATION RESULTS", c)
            }
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(color));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let line = Line::from(vec![
            Span::styled(
                format!(" {} ", title),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "| Esc:models  n:new session  e:export  j/k:scroll",
                Style::default().fg(theme::DIM_GRAY),
            ),
        ]);
        f.render_widget(Paragraph::new(vec![line]), inner);
    }

    fn draw_body(&self, f: &mut Frame, area: Rect) {
        match &self.result {
            SessionResult::Chat {
                final_answer,
                confidence,
                vote_breakdown,
                responses,
            } => self.draw_chat_body(f, area, final_answer, *confidence, vote_breakdown, responses),

            SessionResult::Simulation {
                result,
                yea,
                nay,
                abstain,
                amendments,
                historical_accuracy,
            } => self.draw_sim_body(
                f,
                area,
                result,
                *yea,
                *nay,
                *abstain,
                amendments,
                *historical_accuracy,
            ),
        }
    }

    fn draw_chat_body(
        &self,
        f: &mut Frame,
        area: Rect,
        final_answer: &str,
        confidence: f64,
        vote_breakdown: &serde_json::Value,
        responses: &[(String, String)],
    ) {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
            .split(area);

        // Left: final answer + confidence bar
        let left_rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(cols[0]);

        let answer_block = Block::default()
            .title(" Final Answer ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::CYAN));
        let answer_inner = answer_block.inner(left_rows[0]);
        f.render_widget(answer_block, left_rows[0]);
        let answer_para = Paragraph::new(final_answer)
            .style(Style::default().fg(theme::GRAY))
            .wrap(Wrap { trim: false })
            .scroll((self.scroll, 0));
        f.render_widget(answer_para, answer_inner);

        // Confidence bar
        let conf_block = Block::default()
            .title(" Confidence ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::ACCENT));
        let conf_inner = conf_block.inner(left_rows[1]);
        f.render_widget(conf_block, left_rows[1]);

        let bar_width = conf_inner.width.saturating_sub(15) as usize;
        let filled = (confidence * bar_width as f64).round() as usize;
        let empty = bar_width.saturating_sub(filled);
        let mut conf_lines = vec![
            Line::from(vec![
                Span::styled("Confidence: ", Style::default().fg(theme::DIM_GRAY)),
                Span::styled("█".repeat(filled), Style::default().fg(theme::GREEN)),
                Span::styled("░".repeat(empty), Style::default().fg(theme::DARK_GRAY)),
                Span::styled(
                    format!(" {:.1}%", confidence * 100.0),
                    Style::default().fg(theme::CYAN),
                ),
            ]),
        ];

        // Vote breakdown bars
        if let Some(obj) = vote_breakdown.as_object() {
            for (model, weight) in obj {
                let w = weight.as_f64().unwrap_or(0.0);
                let bar_w = 20usize;
                let filled_w = (w * bar_w as f64).round() as usize;
                conf_lines.push(Line::from(vec![
                    Span::styled(
                        format!("{:>14} ", model),
                        Style::default().fg(theme::ACCENT),
                    ),
                    Span::styled(
                        "█".repeat(filled_w),
                        Style::default().fg(theme::BLUE),
                    ),
                    Span::styled(
                        format!(" {:.3}", w),
                        Style::default().fg(theme::DIM_GRAY),
                    ),
                ]));
            }
        }

        f.render_widget(Paragraph::new(conf_lines), conf_inner);

        // Right: per-model responses
        let resp_block = Block::default()
            .title(" Model Responses ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::BLUE));
        let resp_inner = resp_block.inner(cols[1]);
        f.render_widget(resp_block, cols[1]);

        let mut resp_lines: Vec<Line> = Vec::new();
        for (model, response) in responses {
            resp_lines.push(Line::from(Span::styled(
                format!("[ {} ]", model),
                Style::default()
                    .fg(theme::CYAN)
                    .add_modifier(Modifier::BOLD),
            )));
            let short = if response.len() > 200 {
                format!("{}...", &response[..197])
            } else {
                response.clone()
            };
            for line_str in short.lines() {
                resp_lines.push(Line::from(Span::styled(
                    line_str.to_string(),
                    Style::default().fg(theme::GRAY),
                )));
            }
            resp_lines.push(Line::from(""));
        }

        let scroll_offset = self.scroll.min(resp_lines.len().saturating_sub(1) as u16);
        let para = Paragraph::new(resp_lines).scroll((scroll_offset, 0));
        f.render_widget(para, resp_inner);
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_sim_body(
        &self,
        f: &mut Frame,
        area: Rect,
        result: &str,
        yea: u32,
        nay: u32,
        abstain: u32,
        amendments: &[serde_json::Value],
        historical_accuracy: Option<f64>,
    ) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5),
                Constraint::Min(0),
            ])
            .split(area);

        // Result banner
        let banner_color = if result.to_uppercase().contains("PASSED") {
            theme::GREEN
        } else if result.to_uppercase().contains("FAILED") {
            theme::RED
        } else {
            theme::YELLOW
        };

        let banner_block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(banner_color));
        let banner_inner = banner_block.inner(rows[0]);
        f.render_widget(banner_block, rows[0]);

        let mut banner_lines = vec![
            Line::from(Span::styled(
                format!("  {}  ", result.to_uppercase()),
                Style::default()
                    .fg(banner_color)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(vec![
                Span::styled("YEA: ", Style::default().fg(theme::DIM_GRAY)),
                Span::styled(
                    format!("{}", yea),
                    Style::default().fg(theme::GREEN).add_modifier(Modifier::BOLD),
                ),
                Span::styled("  NAY: ", Style::default().fg(theme::DIM_GRAY)),
                Span::styled(
                    format!("{}", nay),
                    Style::default().fg(theme::RED).add_modifier(Modifier::BOLD),
                ),
                Span::styled("  ABSTAIN: ", Style::default().fg(theme::DIM_GRAY)),
                Span::styled(
                    format!("{}", abstain),
                    Style::default().fg(theme::YELLOW),
                ),
            ]),
        ];

        if let Some(acc) = historical_accuracy {
            banner_lines.push(Line::from(vec![
                Span::styled("Historical Accuracy: ", Style::default().fg(theme::DIM_GRAY)),
                Span::styled(
                    format!("{:.1}%", acc * 100.0),
                    Style::default().fg(theme::ACCENT),
                ),
            ]));
        }

        f.render_widget(Paragraph::new(banner_lines), banner_inner);

        // Amendments
        if !amendments.is_empty() {
            let amend_block = Block::default()
                .title(" Amendment Outcomes ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::CYAN));
            let amend_inner = amend_block.inner(rows[1]);
            f.render_widget(amend_block, rows[1]);

            let mut amend_lines: Vec<Line> = Vec::new();
            for (i, amendment) in amendments.iter().enumerate() {
                let text = amendment
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("(no text)");
                let status = amendment
                    .get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let status_color = if status.to_uppercase().contains("PASS") {
                    theme::GREEN
                } else if status.to_uppercase().contains("FAIL") {
                    theme::RED
                } else {
                    theme::YELLOW
                };
                let ay = amendment.get("yea").and_then(|v| v.as_u64()).unwrap_or(0);
                let an = amendment.get("nay").and_then(|v| v.as_u64()).unwrap_or(0);

                amend_lines.push(Line::from(vec![
                    Span::styled(
                        format!("A{}: ", i + 1),
                        Style::default().fg(theme::ACCENT),
                    ),
                    Span::styled(
                        format!("{} ", status.to_uppercase()),
                        Style::default().fg(status_color).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("({}/{}) ", ay, an),
                        Style::default().fg(theme::DIM_GRAY),
                    ),
                    Span::styled(text.to_string(), Style::default().fg(theme::GRAY)),
                ]));
            }

            let scroll_offset =
                self.scroll.min(amend_lines.len().saturating_sub(1) as u16);
            f.render_widget(
                Paragraph::new(amend_lines).scroll((scroll_offset, 0)),
                amend_inner,
            );
        }
    }

    fn draw_footer(&self, f: &mut Frame, area: Rect) {
        let msg = self
            .export_message
            .clone()
            .unwrap_or_else(|| "Esc:models  n:new  e:export  j/k:scroll".to_string());
        let color = if self.export_message.is_some() {
            theme::GREEN
        } else {
            theme::DIM_GRAY
        };
        let line = Line::from(Span::styled(msg, Style::default().fg(color)));
        f.render_widget(Paragraph::new(vec![line]), area);
    }
}
