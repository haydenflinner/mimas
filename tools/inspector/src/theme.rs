//! The "Summer" palette -- a light, warm theme. Each function parses its hex literal at call
//! time rather than being a `const`: `bevy_color`'s hex parser isn't `const fn`, and these are
//! called rarely enough (building a style bundle, not per-op) that it doesn't matter.

use bevy::color::{Color, Srgba};

fn hex(s: &str) -> Color {
    Color::Srgba(Srgba::hex(s).expect("theme.rs hex literal should be valid"))
}

pub fn red() -> Color {
    hex("#c58687")
}
pub fn green() -> Color {
    hex("#91a77a")
}
pub fn yellow() -> Color {
    hex("#c4aa80")
}

/// Primary text.
pub fn text() -> Color {
    hex("#2b3034")
}
/// Secondary text -- labels, headers that aren't the main point of a row.
pub fn subtext1() -> Color {
    hex("#455355")
}
/// Tertiary text -- internals-view detail (raw registers, chunk/ip), de-emphasized on purpose.
pub fn subtext0() -> Color {
    hex("#576869")
}
/// Borders on interactive elements (buttons).
pub fn overlay1() -> Color {
    hex("#829084")
}
/// Borders on static containers (panels).
pub fn overlay0() -> Color {
    hex("#acb5a4")
}
/// Button background.
pub fn surface1() -> Color {
    hex("#e6e1d3")
}
/// Panel background (source panel, frame panels).
pub fn surface0() -> Color {
    hex("#ede8dd")
}
/// Window background.
pub fn base() -> Color {
    hex("#f5efe6")
}
