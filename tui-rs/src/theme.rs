use ratatui::style::Color;

pub const CYAN: Color = Color::Rgb(0, 215, 255);
pub const BLUE: Color = Color::Rgb(95, 135, 255);
pub const GREEN: Color = Color::Rgb(181, 189, 104);
pub const RED: Color = Color::Rgb(204, 102, 102);
pub const YELLOW: Color = Color::Rgb(255, 255, 0);
pub const GRAY: Color = Color::Rgb(128, 128, 128);
pub const DIM_GRAY: Color = Color::Rgb(102, 102, 102);
pub const DARK_GRAY: Color = Color::Rgb(80, 80, 80);
pub const ACCENT: Color = Color::Rgb(138, 190, 183);
pub const PURPLE: Color = Color::Rgb(149, 117, 205);

// Party colors
pub const REPUBLICAN: Color = RED;
pub const DEMOCRAT: Color = Color::Rgb(100, 149, 237); // cornflower blue
pub const INDEPENDENT: Color = YELLOW;

/// Returns the color for a given party string.
pub fn party_color(party: &str) -> Color {
    match party.to_uppercase().as_str() {
        "R" | "REPUBLICAN" => REPUBLICAN,
        "D" | "DEMOCRAT" | "DEMOCRATIC" => DEMOCRAT,
        "I" | "INDEPENDENT" => INDEPENDENT,
        _ => GRAY,
    }
}

/// Returns the color for a given vote string.
pub fn vote_color(vote: &str) -> Color {
    match vote.to_lowercase().as_str() {
        "yea" | "yes" => GREEN,
        "nay" | "no" => RED,
        "abstain" => YELLOW,
        _ => GRAY,
    }
}
