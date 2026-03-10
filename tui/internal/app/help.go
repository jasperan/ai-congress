// tui/internal/app/help.go
package app

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/jasperan/ai-congress/tui/internal/theme"
)

type HelpModel struct{}

func (h HelpModel) Overlay(bg string, w, h2 int, screen Screen) string {
	bindings := h.bindingsForScreen(screen)

	var b strings.Builder
	b.WriteString(theme.Title.Render("Keyboard Shortcuts") + "\n\n")

	for _, kb := range bindings {
		b.WriteString("  " + theme.KeyName.Render(padRight(kb.key, 12)) + theme.MutedText.Render(kb.desc) + "\n")
	}

	b.WriteString("\n" + theme.MutedText.Render("── Global ──") + "\n")
	b.WriteString("  " + theme.KeyName.Render(padRight("F1", 12)) + theme.MutedText.Render("Toggle help") + "\n")
	b.WriteString("  " + theme.KeyName.Render(padRight("Ctrl+C", 12)) + theme.MutedText.Render("Quit") + "\n")

	helpContent := lipgloss.NewStyle().
		Width(55).
		Padding(1, 2).
		Border(lipgloss.RoundedBorder()).
		BorderForeground(theme.Primary).
		Foreground(theme.Text).
		Render(b.String())

	return lipgloss.Place(
		w, h2,
		lipgloss.Center, lipgloss.Center,
		helpContent,
		lipgloss.WithWhitespaceChars(" "),
		lipgloss.WithWhitespaceForeground(lipgloss.Color("#000000")),
	)
}

type keyBinding struct {
	key  string
	desc string
}

func (h HelpModel) bindingsForScreen(screen Screen) []keyBinding {
	switch screen {
	case ScreenSplash:
		return []keyBinding{
			{"Enter", "Continue"},
			{"r", "Retry connection"},
			{"q", "Quit"},
		}
	case ScreenModels:
		return []keyBinding{
			{"Space", "Toggle model selection"},
			{"Enter", "Chat with selected"},
			{"↑/↓", "Navigate"},
			{"/", "Filter"},
			{"h", "History/Stats"},
			{"q", "Back"},
		}
	case ScreenChat:
		return []keyBinding{
			{"Enter", "Launch chat"},
			{"Tab", "Next field"},
			{"Ctrl+S", "Cycle swarm mode"},
			{"Ctrl+D", "Cycle voting mode"},
			{"Esc", "Back"},
		}
	case ScreenDashboard:
		return []keyBinding{
			{"Tab", "Toggle Focus/Grid"},
			{"1-9", "Select model"},
			{"←/→", "Cycle models"},
			{"↑/↓", "Scroll feed"},
			{"q", "Back"},
		}
	case ScreenHistory:
		return []keyBinding{
			{"q", "Back"},
		}
	default:
		return nil
	}
}

func padRight(s string, n int) string {
	if len(s) >= n {
		return s
	}
	return s + strings.Repeat(" ", n-len(s))
}
