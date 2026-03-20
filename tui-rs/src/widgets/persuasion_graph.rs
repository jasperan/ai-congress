use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

use crate::theme;

pub struct GraphNode {
    pub name: String,
    pub party: String,
}

pub struct GraphEdge {
    pub from: usize,
    pub to: usize,
    pub strength: f64,
}

pub fn draw_persuasion_graph(
    f: &mut Frame,
    area: Rect,
    nodes: &[GraphNode],
    edges: &[GraphEdge],
) {
    let block = Block::default()
        .title(" Persuasion Network ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::PURPLE));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if nodes.is_empty() || inner.width < 10 || inner.height < 5 {
        return;
    }

    let w = inner.width as usize;
    let h = inner.height as usize;
    let mut buf: Vec<Vec<(char, Color)>> = vec![vec![(' ', theme::DARK_GRAY); w]; h];

    // Position nodes in circle
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;
    let radius = (cx.min(cy) - 2.0).max(1.0);

    let positions: Vec<(f64, f64)> = nodes
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / nodes.len() as f64
                - std::f64::consts::FRAC_PI_2;
            (cx + radius * angle.cos(), cy + radius * angle.sin())
        })
        .collect();

    // Draw edges
    for edge in edges {
        if edge.from >= nodes.len() || edge.to >= nodes.len() {
            continue;
        }
        let (x0, y0) = positions[edge.from];
        let (x1, y1) = positions[edge.to];
        let color = if edge.strength > 0.5 {
            theme::GREEN
        } else {
            theme::RED
        };
        let ch = if edge.strength > 0.7 {
            '━'
        } else if edge.strength > 0.4 {
            '─'
        } else {
            '┄'
        };
        draw_line(&mut buf, x0, y0, x1, y1, ch, color);

        // Arrowhead
        let dx = x1 - x0;
        let dy = y1 - y0;
        let arrow = if dx.abs() > dy.abs() {
            if dx > 0.0 { '▶' } else { '◀' }
        } else {
            if dy > 0.0 { '▼' } else { '▲' }
        };
        set_char(&mut buf, x1, y1, arrow, color);
    }

    // Draw node labels (overwrite edges)
    for (i, node) in nodes.iter().enumerate() {
        let (x, y) = positions[i];
        let label: String = node.name.chars().take(8).collect();
        let color = theme::party_color(&node.party);
        let start_x = (x as isize - label.len() as isize / 2).max(0) as usize;
        for (j, ch) in label.chars().enumerate() {
            set_char(&mut buf, (start_x + j) as f64, y, ch, color);
        }
    }

    // Convert to ratatui Lines
    let lines: Vec<Line> = buf
        .iter()
        .map(|row| {
            Line::from(
                row.iter()
                    .map(|(ch, color)| Span::styled(ch.to_string(), Style::default().fg(*color)))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();

    f.render_widget(Paragraph::new(lines), inner);
}

fn set_char(buf: &mut [Vec<(char, Color)>], x: f64, y: f64, ch: char, color: Color) {
    let xi = x.round() as usize;
    let yi = y.round() as usize;
    if yi < buf.len() && xi < buf[0].len() {
        buf[yi][xi] = (ch, color);
    }
}

fn draw_line(
    buf: &mut [Vec<(char, Color)>],
    x0: f64, y0: f64, x1: f64, y1: f64,
    ch: char, color: Color,
) {
    let steps = ((x1 - x0).abs().max((y1 - y0).abs()) as usize).max(1);
    for i in 0..=steps {
        let t = i as f64 / steps as f64;
        let x = x0 + (x1 - x0) * t;
        let y = y0 + (y1 - y0) * t;
        set_char(buf, x, y, ch, color);
    }
}
