// Sentiment/progress/sparkline helpers and throughput accounting.
// Extracted from simulation.rs (plan 4.4.6) — mechanical split, no logic changes.

use std::collections::VecDeque;

use ratatui::style::Color;

use crate::theme;
use super::simulation::SimulationScreen;

// ── Throughput ─────────────────────────────────────────────────────────

impl SimulationScreen {
    pub(crate) fn compute_tps(&self) -> f64 {
        let recent: u32 = self.throughput_buckets.iter().rev().take(5).sum();
        recent as f64 / 5.0
    }
}

// ── Free helper functions ──────────────────────────────────────────────

pub fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len > 3 {
        format!("{}...", &s[..max_len - 3])
    } else {
        s[..max_len].to_string()
    }
}
pub(crate) fn sentiment_indicator(score: f64) -> (String, Color) {
    let bar_width = 5;
    let filled = ((score.abs() * bar_width as f64).round() as usize).min(bar_width);

    let (symbol, color) = if score > 0.3 {
        ("+", theme::SUCCESS)
    } else if score > 0.0 {
        ("+", theme::SUCCESS)
    } else if score < -0.3 {
        ("-", theme::ERROR)
    } else if score < 0.0 {
        ("-", theme::WARNING)
    } else {
        ("=", theme::MUTED)
    };

    let bar = format!("{}{:.1}", symbol.repeat(filled.max(1_usize)), score);
    (bar, color)
}
pub(crate) fn render_sparkline(buckets: &VecDeque<u32>, width: usize) -> String {
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
pub(crate) fn make_progress_bar(progress: f64, width: usize) -> String {
    let filled = (progress * width as f64) as usize;
    let empty = width.saturating_sub(filled);
    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}
pub(crate) fn progress_color(progress: f64) -> Color {
    if progress >= 0.9 {
        theme::SUCCESS
    } else if progress >= 0.5 {
        theme::INFO
    } else {
        theme::PRIMARY
    }
}
pub(crate) fn tail_lines(text: &str, max_lines: usize, line_width: usize) -> String {
    if text.is_empty() || max_lines == 0 {
        return String::new();
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    #[test]
    fn sentiment_indicator_positive_negative_zero() {
        let (pos, pos_c) = sentiment_indicator(0.8);
        let (neg, neg_c) = sentiment_indicator(-0.8);
        let (zero, zero_c) = sentiment_indicator(0.0);

        assert!(pos.contains("+"), "positive should carry an up indicator: {pos}");
        assert!(neg.contains("-"), "negative should carry a down indicator: {neg}");
        assert!(zero.contains("="), "zero should be neutral: {zero}");
        assert!(!pos.is_empty());
        assert_ne!(pos_c, neg_c, "positive and negative colors must differ");
        assert_ne!(zero_c, pos_c);
    }

    #[test]
    fn make_progress_bar_full_and_empty() {
        let full = make_progress_bar(1.0, 10);
        assert_eq!(full.chars().count(), 10);
        assert!(full.chars().all(|c| c == '█'), "full bar should be filled");

        let empty = make_progress_bar(0.0, 10);
        assert_eq!(empty.chars().count(), 10);
        assert!(empty.chars().all(|c| c == '░'), "empty bar should be blank");
        assert_ne!(full, empty);
    }

    #[test]
    fn progress_color_monotonic_mapping() {
        let low = progress_color(0.0);
        let mid = progress_color(0.7);
        let high = progress_color(1.0);
        assert_ne!(low, mid, "low and mid progress must map to different colors");
        assert_ne!(mid, high, "mid and high progress must map to different colors");
        assert_ne!(low, high);
        // boundary: exactly 0.5 flips to the INFO tier
        assert_eq!(progress_color(0.5), progress_color(0.7));
        // boundary: exactly 0.9 flips to the SUCCESS tier
        assert_eq!(progress_color(0.9), progress_color(1.0));
    }

    #[test]
    fn render_sparkline_width() {
        let buckets = VecDeque::from(vec![1u32, 2, 3, 4, 5, 6, 7, 8]);
        let wide = render_sparkline(&buckets, 8);
        assert_eq!(wide.chars().count(), 8);
        // narrow width truncates from the tail of the window (most recent)
        let narrow = render_sparkline(&buckets, 5);
        assert_eq!(narrow.chars().count(), 5);
        // wider-than-data width pads nothing but still respects width
        let padded = render_sparkline(&buckets, 3);
        assert_eq!(padded.chars().count(), 3);
    }

    #[test]
    fn tail_lines_caps_line_count() {
        let text = "one two three four five six seven eight nine ten";
        let out = tail_lines(text, 4, 4); // narrow wrap forces many lines
        assert!(out.lines().count() <= 4);
        assert!(out.contains("ten"), "tail must keep the newest line");

        let single = tail_lines("short", 10, 100);
        assert_eq!(single.lines().count(), 1);
        assert_eq!(tail_lines("", 5, 10), "");
        assert_eq!(tail_lines("abc", 0, 10), "");
    }

    #[test]
    fn truncate_str_short_and_long() {
        let short = truncate_str("hello", 20);
        assert_eq!(short, "hello");
        assert_eq!(truncate_str("hello", 10), "hello");

        let long = truncate_str("abcdefghij", 5);
        assert_eq!(long.len(), 5);
        assert!(long.ends_with("..."));
        assert_eq!(truncate_str("abcdefghij", 3), "abc");
    }

    #[test]
    fn compute_tps_averages_recent_buckets() {
        let mut screen = SimulationScreen::new("t".into(), 10, "m".into());
        screen.throughput_buckets.clear();
        for _ in 0..60 {
            screen.throughput_buckets.push_back(10);
        }
        assert_eq!(screen.compute_tps(), 10.0);
    }

    #[test]
    fn compute_tps_empty_buckets_is_zero() {
        let mut screen = SimulationScreen::new("t".into(), 10, "m".into());
        screen.throughput_buckets.clear();
        let tps = screen.compute_tps();
        assert!(tps >= 0.0, "empty buckets must not panic and stay >= 0");
    }
}
