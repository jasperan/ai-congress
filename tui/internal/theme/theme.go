package theme

import "github.com/charmbracelet/lipgloss"

// AI Congress Pi-inspired color palette.
var (
	Primary = lipgloss.Color("#5f87ff") // blue (from Pi theme)
	Accent  = lipgloss.Color("#b5bd68") // green
	Cyan    = lipgloss.Color("#00d7ff") // cyan highlight
	Danger  = lipgloss.Color("#cc6666") // red
	Warning = lipgloss.Color("#ffff00") // yellow
	Muted   = lipgloss.Color("#808080") // gray
	Surface = lipgloss.Color("#1C1C1E") // dark bg
	Text    = lipgloss.Color("#E5E5EA") // light fg
	Dim     = lipgloss.Color("#505050") // darker gray
)

// Panel styles.
var (
	Panel = lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(Muted).
		Padding(1, 2)

	ActivePanel = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(Primary).
			Padding(1, 2)

	GeneratingPanel = lipgloss.NewStyle().
				Border(lipgloss.RoundedBorder()).
				BorderForeground(Accent).
				Padding(1, 2)
)

// Text styles.
var (
	Title = lipgloss.NewStyle().
		Bold(true).
		Foreground(Text)

	Subtitle = lipgloss.NewStyle().
			Bold(true).
			Foreground(Primary)

	MutedText = lipgloss.NewStyle().
			Foreground(Muted)

	ErrorText = lipgloss.NewStyle().
			Bold(true).
			Foreground(Danger)

	ModelName = lipgloss.NewStyle().
			Bold(true).
			Foreground(Cyan)

	TokenText = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#AEAEB2"))

	Throughput = lipgloss.NewStyle().
			Bold(true).
			Foreground(Accent)
)

// Status indicators.
var (
	StatusActive = lipgloss.NewStyle().Foreground(Accent)
	StatusIdle   = lipgloss.NewStyle().Foreground(Muted)
	StatusError  = lipgloss.NewStyle().Foreground(Danger)
)

const StatusDot = "\u25cf" // ●

// Confidence level styles (replaces hazard).
var (
	ConfidenceLow    = lipgloss.NewStyle().Foreground(Danger)
	ConfidenceMedium = lipgloss.NewStyle().Foreground(Warning)
	ConfidenceHigh   = lipgloss.NewStyle().Foreground(Accent)
)

// StatusBar style.
var StatusBar = lipgloss.NewStyle().
	Background(Surface).
	Foreground(Text)

// Key hint styles.
var (
	KeyHint = lipgloss.NewStyle().Foreground(Dim)
	KeyName = lipgloss.NewStyle().Bold(true).Foreground(Primary)
)

// Cursor style.
var Cursor = lipgloss.NewStyle().Background(Primary)

// ConfidenceColor returns a lipgloss.Color based on a 0.0-1.0 confidence level.
func ConfidenceColor(level float64) lipgloss.Color {
	switch {
	case level >= 0.7:
		return Accent
	case level >= 0.4:
		return Warning
	default:
		return Danger
	}
}

// StatusColor returns a lipgloss.Color based on a status string.
func StatusColor(status string) lipgloss.Color {
	switch status {
	case "streaming", "generating", "running":
		return Accent
	case "error", "failed":
		return Danger
	case "idle", "waiting", "complete":
		return Muted
	default:
		return Muted
	}
}
