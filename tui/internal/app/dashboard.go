// tui/internal/app/dashboard.go
package app

import (
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/jasperan/ai-congress/tui/internal/api"
	"github.com/jasperan/ai-congress/tui/internal/components"
	"github.com/jasperan/ai-congress/tui/internal/theme"
)

// DashboardMode toggles between focus and grid layouts.
type DashboardMode int

const (
	ModeFocus DashboardMode = iota
	ModeGrid
)

// modelStream tracks per-model response streaming state.
type modelStream struct {
	name      string
	tokens    strings.Builder
	active    bool
	latencyMs int
	weight    float64
	complete  bool
}

// --- DashboardModel ---

type DashboardModel struct {
	client     *api.Client
	throughput *components.ThroughputTracker
	readOnly   bool

	mode        DashboardMode
	status      string // "waiting", "streaming", "complete", "error"
	models      []string
	streams     map[string]*modelStream
	feedEntries []components.FeedEntry
	feedScroll  int
	selectedIdx int

	// Voting results
	confidence    float64
	voteBreakdown map[string]interface{}
	finalAnswer   string

	errMsg string
}

func NewDashboardModel(client *api.Client, tp *components.ThroughputTracker, readOnly bool) DashboardModel {
	return DashboardModel{
		client:     client,
		throughput: tp,
		readOnly:   readOnly,
		mode:       ModeFocus,
		status:     "waiting",
		streams:    make(map[string]*modelStream),
	}
}

func (m DashboardModel) Init() tea.Cmd {
	return nil
}

func (m DashboardModel) Update(msg tea.Msg) (DashboardModel, tea.Cmd) {
	switch msg := msg.(type) {
	case WSEventMsg:
		m.handleWSEvent(msg.Event)
		return m, nil

	case tea.KeyMsg:
		m.errMsg = ""
		return m.handleKey(msg)
	}

	return m, nil
}

func (m *DashboardModel) handleWSEvent(evt api.WSMessage) {
	switch evt.Type {
	case "start":
		m.status = "streaming"

	case "status_update":
		// Personality mode status update
		name := evt.Data["name"]
		status := evt.Data["status"]
		if n, ok := name.(string); ok {
			if _, exists := m.streams[n]; !exists {
				m.streams[n] = &modelStream{name: n}
				m.models = append(m.models, n)
			}
			if s, ok := status.(string); ok {
				if s == "Generating..." {
					m.streams[n].active = true
				} else if s == "Complete" {
					m.streams[n].active = false
					m.streams[n].complete = true
				}
			}
		}

	case "chunk":
		// Streaming token chunk (personality mode)
		name, _ := evt.Data["name"].(string)
		content, _ := evt.Data["content"].(string)
		if name != "" {
			if _, exists := m.streams[name]; !exists {
				m.streams[name] = &modelStream{name: name}
				m.models = append(m.models, name)
			}
			m.streams[name].active = true
			m.streams[name].tokens.WriteString(content)
			m.throughput.Add(1)
		}

	case "model_response":
		model := evt.Model
		content := evt.Content
		if model == "" {
			model, _ = evt.Data["model"].(string)
		}
		if content == "" {
			content, _ = evt.Data["content"].(string)
		}
		if model != "" {
			if _, exists := m.streams[model]; !exists {
				m.streams[model] = &modelStream{name: model}
				m.models = append(m.models, model)
			}
			s := m.streams[model]
			s.tokens.Reset()
			s.tokens.WriteString(content)
			s.active = false
			s.complete = true

			m.feedEntries = append(m.feedEntries, components.FeedEntry{
				ModelName:   model,
				Content:     content,
				MessageType: "model_response",
			})
		}

	case "final_answer":
		m.status = "complete"
		m.finalAnswer = evt.Content
		if m.finalAnswer == "" {
			if c, ok := evt.Data["content"].(string); ok {
				m.finalAnswer = c
			}
		}
		m.confidence = evt.Confidence
		if m.confidence == 0 {
			if c, ok := evt.Data["confidence"].(float64); ok {
				m.confidence = c
			}
		}
		if evt.VoteBreakdown != nil {
			m.voteBreakdown = evt.VoteBreakdown
		} else if vb, ok := evt.Data["vote_breakdown"].(map[string]interface{}); ok {
			m.voteBreakdown = vb
		}

		m.feedEntries = append(m.feedEntries, components.FeedEntry{
			ModelName:   "Congress",
			Content:     m.finalAnswer,
			MessageType: "final_answer",
			Confidence:  m.confidence,
		})

	case "end":
		m.status = "complete"

	case "error":
		m.status = "error"
		m.errMsg = evt.Message
		if m.errMsg == "" {
			if msg, ok := evt.Data["message"].(string); ok {
				m.errMsg = msg
			}
		}
	}
}

func (m DashboardModel) handleKey(msg tea.KeyMsg) (DashboardModel, tea.Cmd) {
	switch msg.String() {
	case "tab":
		if m.mode == ModeFocus {
			m.mode = ModeGrid
		} else {
			m.mode = ModeFocus
		}

	case "1", "2", "3", "4", "5", "6", "7", "8", "9":
		idx := int(msg.String()[0]-'0') - 1
		if idx < len(m.models) {
			m.selectedIdx = idx
		}

	case "up", "k":
		if m.feedScroll > 0 {
			m.feedScroll--
		}
	case "down", "j":
		m.feedScroll++

	case "left", "h":
		if m.selectedIdx > 0 {
			m.selectedIdx--
		}
	case "right", "l":
		if m.selectedIdx < len(m.models)-1 {
			m.selectedIdx++
		}

	case "q", "esc":
		return m, func() tea.Msg {
			return SwitchScreenMsg{Screen: ScreenModels}
		}
	}

	return m, nil
}

func (m DashboardModel) View(width, height int) string {
	if len(m.models) == 0 && m.status == "waiting" {
		return lipgloss.Place(width, height, lipgloss.Center, lipgloss.Center,
			theme.MutedText.Render("Waiting for responses..."))
	}

	statusBarHeight := 1
	mainHeight := height - statusBarHeight

	var main string
	if m.mode == ModeFocus {
		main = m.renderFocusMode(width, mainHeight)
	} else {
		main = m.renderGridMode(width, mainHeight)
	}

	hints := []components.KeyHint{
		{Key: "Tab", Desc: "mode"},
		{Key: "F1", Desc: "help"},
		{Key: "q", Desc: "back"},
	}
	if m.errMsg != "" {
		hints = append([]components.KeyHint{{Key: "!", Desc: m.errMsg}}, hints...)
	}

	bar := components.RenderStatusBar(components.StatusBarData{
		Connected:  true,
		Status:     m.status,
		ModelCount: len(m.models),
		Mode:       "swarm",
		TokPerSec:  m.throughput.CurrentRate(),
		Hints:      hints,
	}, width)

	return lipgloss.JoinVertical(lipgloss.Left, main, bar)
}

func (m DashboardModel) renderFocusMode(width, height int) string {
	leftWidth := width * 60 / 100
	rightWidth := width - leftWidth

	modelPanes := m.buildModelPanes(leftWidth, height, 4)
	left := lipgloss.JoinVertical(lipgloss.Left, modelPanes...)
	left = lipgloss.NewStyle().Width(leftWidth).Height(height).Render(left)

	feedHeight := height * 2 / 3
	votingHeight := height - feedHeight

	feed := components.RenderMessageFeed(components.MessageFeedData{
		Entries:   m.feedEntries,
		ScrollPos: m.feedScroll,
		Height:    feedHeight - 2,
		Width:     rightWidth - 2,
	})
	feedPanel := theme.Panel.Width(rightWidth - 2).Height(feedHeight - 2).Render(feed)

	voting := components.RenderVotingPanel(components.VotingData{
		Confidence:    m.confidence,
		VoteBreakdown: m.voteBreakdown,
		FinalAnswer:   m.finalAnswer,
		Width:         rightWidth - 2,
	})
	votingPanel := theme.Panel.Width(rightWidth - 2).Height(votingHeight - 2).Render(voting)

	right := lipgloss.JoinVertical(lipgloss.Left, feedPanel, votingPanel)
	right = lipgloss.NewStyle().Width(rightWidth).Height(height).Render(right)

	return lipgloss.JoinHorizontal(lipgloss.Top, left, right)
}

func (m DashboardModel) renderGridMode(width, height int) string {
	cols := 5
	rows := 2
	paneWidth := width / cols
	paneHeight := height / rows

	var rowViews []string
	modelIdx := 0
	for r := 0; r < rows; r++ {
		var colViews []string
		for c := 0; c < cols; c++ {
			if modelIdx < len(m.models) {
				pane := m.buildSingleModelPane(modelIdx, paneWidth, paneHeight)
				colViews = append(colViews, pane)
			} else {
				empty := theme.Panel.Width(paneWidth - 2).Height(paneHeight - 2).
					Render(theme.MutedText.Render("empty"))
				colViews = append(colViews, empty)
			}
			modelIdx++
		}
		rowViews = append(rowViews, lipgloss.JoinHorizontal(lipgloss.Top, colViews...))
	}

	return lipgloss.JoinVertical(lipgloss.Left, rowViews...)
}

func (m DashboardModel) buildModelPanes(width, totalHeight, maxPanes int) []string {
	count := len(m.models)
	if count > maxPanes {
		count = maxPanes
	}
	if count == 0 {
		return []string{theme.MutedText.Render("No models responding")}
	}

	paneHeight := totalHeight / count
	var panes []string
	for i := 0; i < count; i++ {
		panes = append(panes, m.buildSingleModelPane(i, width, paneHeight))
	}
	return panes
}

func (m DashboardModel) buildSingleModelPane(idx int, width, height int) string {
	if idx >= len(m.models) {
		return ""
	}

	modelName := m.models[idx]
	stream := m.streams[modelName]

	data := components.ModelPaneData{
		Name:     modelName,
		Selected: idx == m.selectedIdx,
	}

	if stream != nil {
		data.Active = stream.active
		data.Tokens = stream.tokens.String()
		data.LatencyMs = stream.latencyMs
		data.Weight = stream.weight
	}

	return components.RenderModelPane(data, width, height)
}
