package tui

import (
	"strconv"
	"strings"

	"charm.land/lipgloss/v2"
	"github.com/jasperan/ai-congress/gotui/internal/huhstyle"
)

// The pane palette is DERIVED FROM THE THEME, never declared here. Writing a
// colour literal in this file would fork the palette away from the single
// definition in internal/huhstyle (which mirrors tui-rs/src/theme.rs).
func palette() (primary, text, subtext, muted, danger, success, warning lipgloss.Style) {
	styles := huhstyle.Styles()
	return lipgloss.NewStyle().Foreground(styles.Focused.Title.GetForeground()),
		lipgloss.NewStyle().Foreground(styles.Focused.Option.GetForeground()),
		lipgloss.NewStyle().Foreground(styles.Focused.Description.GetForeground()),
		lipgloss.NewStyle().Foreground(styles.Focused.TextInput.Placeholder.GetForeground()),
		lipgloss.NewStyle().Foreground(styles.Focused.ErrorMessage.GetForeground()),
		lipgloss.NewStyle().Foreground(lipgloss.Color(huhstyle.HexSuccess)),
		lipgloss.NewStyle().Foreground(lipgloss.Color(huhstyle.HexWarning))
}

// MinPaneWidth is the narrowest a pane is allowed to become.
//
// Bubble Tea can deliver a width of 0 before the first WindowSizeMsg, and a
// user can shrink a terminal to a few columns; a lipgloss width below the
// border's own footprint produces a negative content width, which panics. Every
// pane therefore clamps.
const MinPaneWidth = 16

// clampWidth raises a requested width to the pane floor.
func clampWidth(width int) int {
	if width < MinPaneWidth {
		return MinPaneWidth
	}
	return width
}

// contentWidth is the pane width available to text, after borders and padding.
func contentWidth(width int) int {
	inner := clampWidth(width) - 6 // rounded border (2) + padding (4)
	if inner < 1 {
		return 1
	}
	return inner
}

// Pane renders a titled panel: rounded border, generous padding, primary title.
func Pane(title, body string, width int) string {
	primary, text, _, _, _, _, _ := palette()

	heading := ""
	if title != "" {
		heading = primary.Bold(true).Render(Truncate(title, contentWidth(width))) + "\n"
	}
	return lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		Padding(1, 2).
		Width(clampWidth(width)).
		Render(heading + text.Render(body))
}

// PaneError renders an error panel. A failed API call is a normal state in this
// front-end, not a crash: the user gets the cause and a way onward.
func PaneError(title string, err error, width int) string {
	_, _, _, _, danger, _, _ := palette()
	message := "unknown error"
	if err != nil {
		message = err.Error()
	}
	return Pane(title, danger.Render(message), width)
}

// Fog renders low-emphasis text.
func Fog(body string) string {
	_, _, subtext, _, _, _, _ := palette()
	return subtext.Render(body)
}

// Subtle renders metadata text.
func Subtle(body string) string {
	_, _, _, muted, _, _, _ := palette()
	return muted.Render(body)
}

// Stat renders one "label: value" line for the readouts.
func Stat(label, value string) string {
	_, text, subtext, _, _, _, _ := palette()
	return subtext.Render(label+": ") + text.Render(value)
}

// Good renders a positive readout (a healthy service, a reached consensus).
func Good(body string) string {
	_, _, _, _, _, success, _ := palette()
	return success.Render(body)
}

// Caution renders a warning readout (an open circuit breaker, a low consensus).
func Caution(body string) string {
	_, _, _, _, _, _, warning := palette()
	return warning.Render(body)
}

// Header is the persistent identity bar.
func Header(serviceURL, model string, width int) string {
	primary, _, subtext, _, _, _, _ := palette()
	title := primary.Bold(true).Render("AI Congress") + " " + subtext.Render("Go front-end")
	right := subtext.Render(Truncate(serviceURL+"  "+model, contentWidth(width)))
	gap := clampWidth(width) - lipgloss.Width(title) - lipgloss.Width(right) - 2
	if gap < 1 {
		gap = 1
	}
	if lipgloss.Width(title)+lipgloss.Width(right)+1 > clampWidth(width) {
		return lipgloss.NewStyle().MaxWidth(clampWidth(width)).Render(title)
	}
	return title + strings.Repeat(" ", gap) + right
}

// Footer is the key-hint bar.
func Footer(hint string, width int) string {
	_, _, subtext, _, _, _, _ := palette()
	if lipgloss.Width(hint) > width-2 && width > 2 {
		hint = Truncate(hint, width-2)
	}
	return subtext.Render(hint)
}

// Pad right-pads to n columns.
func Pad(s string, n int) string {
	if lipgloss.Width(s) >= n {
		return s
	}
	return s + strings.Repeat(" ", n-lipgloss.Width(s))
}

// Truncate shortens s to n columns, adding a marker when it cuts.
//
// It counts runes rather than bytes so a multi-byte model name cannot be split
// into invalid UTF-8, and it never indexes out of range for n <= 1.
func Truncate(s string, n int) string {
	if n <= 1 {
		return ""
	}
	runes := []rune(s)
	if len(runes) <= n {
		return s
	}
	return string(runes[:n-1]) + "~"
}

// Bar renders a proportional bar for a 0..1 fraction.
//
// Used for vote tallies and consensus. An out-of-range fraction is clamped
// rather than allowed to produce a negative repeat count.
func Bar(fraction float64, width int) string {
	if width < 1 {
		width = 1
	}
	if fraction < 0 {
		fraction = 0
	}
	if fraction > 1 {
		fraction = 1
	}
	filled := int(fraction*float64(width) + 0.5)
	if filled > width {
		filled = width
	}
	return strings.Repeat("#", filled) + strings.Repeat("-", width-filled)
}

// Percent renders a 0..1 fraction as a fixed-width percentage.
//
// It tolerates both ends: NaN (a division by zero voters) renders as a
// placeholder, and an out-of-range fraction is clamped rather than rendered as
// a negative percentage.
func Percent(fraction float64) string {
	if fraction != fraction { // NaN
		return "  n/a"
	}
	if fraction < 0 {
		fraction = 0
	}
	if fraction > 1 {
		fraction = 1
	}
	return strconv.Itoa(int(fraction*100+0.5)) + "%"
}
