package components

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/jasperan/ai-congress/tui/internal/theme"
)

type FeedEntry struct {
	ModelName   string
	Content     string
	MessageType string
	Confidence  float64
}

type MessageFeedData struct {
	Entries   []FeedEntry
	ScrollPos int
	Height    int
	Width     int
}

func RenderMessageFeed(d MessageFeedData) string {
	if len(d.Entries) == 0 {
		return theme.MutedText.Render("No responses yet...")
	}

	contentWidth := d.Width - 4
	if contentWidth < 1 {
		contentWidth = 1
	}

	var rendered []string
	for _, entry := range d.Entries {
		header := fmt.Sprintf("%s %s",
			theme.ModelName.Render(entry.ModelName),
			theme.MutedText.Render(fmt.Sprintf("[%.0f%%]", entry.Confidence*100)),
		)

		content := entry.Content
		maxContentLen := contentWidth * 3
		if len(content) > maxContentLen {
			content = content[:maxContentLen] + "..."
		}

		var msgStyle lipgloss.Style
		switch entry.MessageType {
		case "error":
			msgStyle = theme.ErrorText
		case "final_answer":
			msgStyle = lipgloss.NewStyle().Bold(true).Foreground(theme.Cyan)
		case "system":
			msgStyle = theme.MutedText
		default:
			msgStyle = theme.TokenText
		}

		rendered = append(rendered, header, msgStyle.Render(content), "")
	}

	allLines := strings.Split(strings.Join(rendered, "\n"), "\n")

	start := d.ScrollPos
	if start < 0 {
		start = 0
	}
	if start >= len(allLines) {
		start = len(allLines) - 1
	}
	if start < 0 {
		start = 0
	}

	end := start + d.Height
	if end > len(allLines) {
		end = len(allLines)
	}

	visible := allLines[start:end]
	return strings.Join(visible, "\n")
}
