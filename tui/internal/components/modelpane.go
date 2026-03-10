package components

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/jasperan/ai-congress/tui/internal/theme"
)

type ModelPaneData struct {
	Name      string
	Active    bool
	Tokens    string
	LatencyMs int
	Selected  bool
	Weight    float64
}

func RenderModelPane(d ModelPaneData, width, height int) string {
	var panelStyle lipgloss.Style
	switch {
	case d.Active:
		panelStyle = theme.GeneratingPanel
	case d.Selected:
		panelStyle = theme.ActivePanel
	default:
		panelStyle = theme.Panel
	}
	panelStyle = panelStyle.Width(width - 2).Height(height - 2)

	var dot string
	if d.Active {
		dot = theme.StatusActive.Render(theme.StatusDot)
	} else {
		dot = theme.StatusIdle.Render(theme.StatusDot)
	}

	name := theme.ModelName.Render(d.Name)
	meta := theme.MutedText.Render(fmt.Sprintf("w:%.2f | %dms", d.Weight, d.LatencyMs))
	header := fmt.Sprintf("%s %s  %s", dot, name, meta)

	bodyHeight := height - 4
	if bodyHeight < 1 {
		bodyHeight = 1
	}
	bodyWidth := width - 6
	if bodyWidth < 1 {
		bodyWidth = 1
	}

	body := TruncateLines(d.Tokens, bodyWidth, bodyHeight)
	if d.Active {
		body += theme.Cursor.Render(" ")
	}

	content := header + "\n" + theme.MutedText.Render(strings.Repeat("─", bodyWidth)) + "\n" + theme.TokenText.Render(body)
	return panelStyle.Render(content)
}

func TruncateLines(text string, width, maxLines int) string {
	if width <= 0 || maxLines <= 0 {
		return ""
	}
	var lines []string
	for _, line := range strings.Split(text, "\n") {
		for len(line) > width {
			lines = append(lines, line[:width])
			line = line[width:]
		}
		lines = append(lines, line)
	}
	if len(lines) > maxLines {
		lines = lines[len(lines)-maxLines:]
	}
	return strings.Join(lines, "\n")
}
