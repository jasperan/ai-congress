package components

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/jasperan/ai-congress/tui/internal/theme"
)

type VotingData struct {
	Confidence    float64
	VoteBreakdown map[string]interface{}
	FinalAnswer   string
	Width         int
}

func RenderVotingPanel(d VotingData) string {
	var sections []string
	sections = append(sections, renderConfidenceGauge(d.Confidence, d.Width-4))

	if len(d.VoteBreakdown) > 0 {
		sections = append(sections, "")
		sections = append(sections, theme.Subtitle.Render("Vote Breakdown"))
		for model, votes := range d.VoteBreakdown {
			voteStr := fmt.Sprintf("%v", votes)
			line := fmt.Sprintf("  %s: %s",
				theme.ModelName.Render(model),
				theme.MutedText.Render(voteStr),
			)
			sections = append(sections, line)
		}
	}

	if d.FinalAnswer != "" {
		sections = append(sections, "")
		sections = append(sections, theme.Subtitle.Render("Consensus Answer"))
		answer := d.FinalAnswer
		maxLen := d.Width * 3
		if len(answer) > maxLen {
			answer = answer[:maxLen] + "..."
		}
		sections = append(sections, theme.TokenText.Render("  "+answer))
	}

	return strings.Join(sections, "\n")
}

func renderConfidenceGauge(level float64, width int) string {
	if width < 10 {
		width = 10
	}
	label := fmt.Sprintf("Confidence %.0f%%", level*100)
	barWidth := width - len(label) - 3
	if barWidth < 5 {
		barWidth = 5
	}
	filled := int(level * float64(barWidth))
	if filled > barWidth {
		filled = barWidth
	}
	empty := barWidth - filled
	color := theme.ConfidenceColor(level)
	gaugeStyle := lipgloss.NewStyle().Foreground(color)
	bar := gaugeStyle.Render(strings.Repeat("\u2588", filled)) +
		theme.MutedText.Render(strings.Repeat("\u2591", empty))
	return fmt.Sprintf("%s [%s]", theme.Title.Render(label), bar)
}
