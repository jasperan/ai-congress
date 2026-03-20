/// Render throughput sparkline from bucket data.
pub fn render_throughput_sparkline(buckets: &[u32], width: usize) -> String {
    let chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let display = &buckets[buckets.len().saturating_sub(width)..];
    let max = display.iter().copied().max().unwrap_or(1).max(1);

    display
        .iter()
        .map(|&v| {
            let idx = ((v as f64 / max as f64) * 7.0).round() as usize;
            chars[idx.min(7)]
        })
        .collect()
}

/// Compute tokens per second from the last N throughput buckets.
pub fn compute_tps(buckets: &[u32], window: usize) -> f64 {
    let n = buckets.len().min(window);
    let start = buckets.len().saturating_sub(n);
    let sum: u32 = buckets[start..].iter().sum();
    if n > 0 {
        sum as f64 / n as f64
    } else {
        0.0
    }
}
