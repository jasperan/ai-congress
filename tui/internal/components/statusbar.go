package components

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/jasperan/ai-congress/tui/internal/theme"
)

type StatusBarData struct {
	Connected  bool
	Status     string
	ModelCount int
	Mode       string
	TokPerSec  float64
	Hints      []KeyHint
}

type KeyHint struct {
	Key  string
	Desc string
}

func RenderStatusBar(d StatusBarData, width int) string {
	var connDot string
	if d.Connected {
		connDot = theme.StatusActive.Render(theme.StatusDot)
	} else {
		connDot = theme.StatusError.Render(theme.StatusDot)
	}

	statusStyle := lipgloss.NewStyle().Foreground(theme.StatusColor(d.Status))
	left := fmt.Sprintf(" %s %s", connDot, statusStyle.Render(d.Status))

	center := fmt.Sprintf("%s | %d models", d.Mode, d.ModelCount)

	tokStr := theme.Throughput.Render(fmt.Sprintf("%.1f tok/s", d.TokPerSec))
	var hints []string
	for _, h := range d.Hints {
		hints = append(hints, fmt.Sprintf("%s %s",
			theme.KeyName.Render(h.Key),
			theme.KeyHint.Render(h.Desc),
		))
	}
	right := tokStr
	if len(hints) > 0 {
		right += "  " + strings.Join(hints, "  ")
	}
	right += " "

	leftW := lipgloss.Width(left)
	centerW := lipgloss.Width(center)
	rightW := lipgloss.Width(right)

	gap1 := width/2 - leftW - centerW/2
	if gap1 < 1 {
		gap1 = 1
	}
	gap2 := width - leftW - gap1 - centerW - rightW
	if gap2 < 1 {
		gap2 = 1
	}

	bar := left + strings.Repeat(" ", gap1) + center + strings.Repeat(" ", gap2) + right
	return theme.StatusBar.Width(width).Render(bar)
}
