// tui/internal/app/history.go
package app

import (
	"fmt"
	"io"

	"github.com/charmbracelet/bubbles/list"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/jasperan/ai-congress/tui/internal/api"
	"github.com/jasperan/ai-congress/tui/internal/theme"
)

// --- Messages ---

type statsLoadedMsg struct {
	stats *api.StatsResponse
	err   error
}

// --- statsItem implements list.Item ---

type statsItem struct {
	label string
	value string
}

func (i statsItem) FilterValue() string { return i.label }

// --- Custom delegate ---

type statsDelegate struct{}

func (d statsDelegate) Height() int                             { return 2 }
func (d statsDelegate) Spacing() int                            { return 0 }
func (d statsDelegate) Update(_ tea.Msg, _ *list.Model) tea.Cmd { return nil }

func (d statsDelegate) Render(w io.Writer, m list.Model, index int, listItem list.Item) {
	item, ok := listItem.(statsItem)
	if !ok {
		return
	}

	isSelected := index == m.Index()
	var nameStyle lipgloss.Style
	if isSelected {
		nameStyle = lipgloss.NewStyle().Bold(true).Foreground(theme.Cyan)
	} else {
		nameStyle = theme.Title
	}

	cursor := "  "
	if isSelected {
		cursor = theme.Subtitle.Render("> ")
	}

	fmt.Fprintf(w, "%s%s: %s", cursor, nameStyle.Render(item.label), theme.MutedText.Render(item.value))
}

// --- HistoryModel ---

type HistoryModel struct {
	client  *api.Client
	list    list.Model
	loading bool
	errMsg  string
	stats   *api.StatsResponse
}

func NewHistoryModel(client *api.Client) HistoryModel {
	delegate := statsDelegate{}
	l := list.New([]list.Item{}, delegate, 0, 0)
	l.Title = "Enhanced Orchestrator Stats"
	l.SetShowStatusBar(true)
	l.SetFilteringEnabled(false)
	l.Styles.Title = theme.Title
	l.SetShowHelp(false)

	return HistoryModel{
		client:  client,
		list:    l,
		loading: true,
	}
}

func (m HistoryModel) Init() tea.Cmd {
	client := m.client
	return func() tea.Msg {
		stats, err := client.GetEnhancedStats()
		return statsLoadedMsg{stats: stats, err: err}
	}
}

func (m HistoryModel) Update(msg tea.Msg) (HistoryModel, tea.Cmd) {
	switch msg := msg.(type) {
	case statsLoadedMsg:
		m.loading = false
		if msg.err != nil {
			m.errMsg = msg.err.Error()
			return m, nil
		}
		m.stats = msg.stats
		items := []list.Item{
			statsItem{label: "Total Runs", value: fmt.Sprintf("%d", msg.stats.TotalRuns)},
			statsItem{label: "Avg Confidence", value: fmt.Sprintf("%.1f%%", msg.stats.AvgConfidence*100)},
			statsItem{label: "Avg Latency", value: fmt.Sprintf("%dms", msg.stats.AvgLatencyMs)},
		}
		m.list.SetItems(items)
		return m, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "q", "esc":
			return m, func() tea.Msg {
				return SwitchScreenMsg{Screen: ScreenModels}
			}
		}
	}

	var cmd tea.Cmd
	m.list, cmd = m.list.Update(msg)
	return m, cmd
}

func (m HistoryModel) View(width, height int) string {
	if m.errMsg != "" {
		errView := theme.ErrorText.Render("Failed to load stats: "+m.errMsg) +
			"\n\n" + theme.KeyName.Render("q") + theme.KeyHint.Render(" back")
		return lipgloss.Place(width, height, lipgloss.Center, lipgloss.Center, errView)
	}

	if m.loading {
		return lipgloss.Place(width, height, lipgloss.Center, lipgloss.Center,
			theme.MutedText.Render("Loading stats..."))
	}

	m.list.SetSize(width, height-2)

	hints := theme.KeyName.Render("q") + theme.KeyHint.Render(" back")

	return m.list.View() + "\n" + hints
}
