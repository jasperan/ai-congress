use ratatui::style::Color;

// ── Catppuccin Mocha ────────────────────────────────────────────────────────

pub const BG: Color = Color::Rgb(0x1e, 0x1e, 0x2e);
pub const SURFACE: Color = Color::Rgb(0x18, 0x18, 0x25);
pub const ELEVATED: Color = Color::Rgb(0x31, 0x32, 0x44);
pub const HIGHEST: Color = Color::Rgb(0x45, 0x47, 0x5a);
pub const TEXT: Color = Color::Rgb(0xcd, 0xd6, 0xf4);
pub const SUBTEXT: Color = Color::Rgb(0xa6, 0xad, 0xc8);
pub const MUTED: Color = Color::Rgb(0x6c, 0x70, 0x86);
pub const DIM: Color = Color::Rgb(0x58, 0x5b, 0x70);
pub const PRIMARY: Color = Color::Rgb(0x89, 0xb4, 0xfa);
pub const SECONDARY: Color = Color::Rgb(0xcb, 0xa6, 0xf7);
pub const INFO: Color = Color::Rgb(0x89, 0xdc, 0xeb);
pub const SUCCESS: Color = Color::Rgb(0xa6, 0xe3, 0xa1);
pub const WARNING: Color = Color::Rgb(0xf9, 0xe2, 0xaf);
pub const ERROR: Color = Color::Rgb(0xf3, 0x8b, 0xa8);

// ── Legacy aliases (preserve existing call sites) ────────────────────────────

pub const CYAN: Color = INFO;
pub const BLUE: Color = PRIMARY;
pub const GREEN: Color = SUCCESS;
pub const RED: Color = ERROR;
pub const YELLOW: Color = WARNING;
pub const GRAY: Color = SUBTEXT;
pub const DIM_GRAY: Color = MUTED;
pub const DARK_GRAY: Color = DIM;
pub const ACCENT: Color = INFO;
pub const PURPLE: Color = SECONDARY;

// ── Party colors ─────────────────────────────────────────────────────────────

pub const REPUBLICAN: Color = ERROR;
pub const DEMOCRAT: Color = PRIMARY;
pub const INDEPENDENT: Color = WARNING;

/// Returns the color for a given party string.
pub fn party_color(party: &str) -> Color {
    match party.to_uppercase().as_str() {
        "R" | "REPUBLICAN" => REPUBLICAN,
        "D" | "DEMOCRAT" | "DEMOCRATIC" => DEMOCRAT,
        "I" | "INDEPENDENT" => INDEPENDENT,
        _ => SUBTEXT,
    }
}

/// Returns the color for a given vote string.
pub fn vote_color(vote: &str) -> Color {
    match vote.to_lowercase().as_str() {
        "yea" | "yes" => SUCCESS,
        "nay" | "no" => ERROR,
        "abstain" => WARNING,
        _ => SUBTEXT,
    }
}
