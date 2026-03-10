// tui/internal/app/chat.go
package app

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/jasperan/ai-congress/tui/internal/theme"
)

// ChatLaunchData holds the data needed to start a chat session on the dashboard.
type ChatLaunchData struct {
	Prompt      string
	Models      []string
	Mode        string
	Temperature float64
	VotingMode  string
}

// --- ChatModel ---

type ChatModel struct {
	models      []string
	promptInput textinput.Model
	tempInput   textinput.Model
	focusIndex  int
	modeIndex   int
	votingIndex int
	launching   bool
	err         error
}

var swarmModes = []string{"multi_model", "multi_request", "hybrid", "personality", "streaming"}
var votingModes = []string{"classic", "semantic"}

func NewChatModel(selectedModels []string) ChatModel {
	prompt := textinput.New()
	prompt.Placeholder = "Ask anything..."
	prompt.CharLimit = 500
	prompt.Width = 60
	prompt.Prompt = "Prompt: "
	prompt.Focus()

	temp := textinput.New()
	temp.Placeholder = "0.7"
	temp.CharLimit = 4
	temp.Width = 10
	temp.Prompt = "Temperature: "

	return ChatModel{
		models:      selectedModels,
		promptInput: prompt,
		tempInput:   temp,
		focusIndex:  0,
	}
}

func (m ChatModel) Init() tea.Cmd {
	return textinput.Blink
}

func (m ChatModel) Update(msg tea.Msg) (ChatModel, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "tab":
			m.focusIndex = (m.focusIndex + 1) % 2
			if m.focusIndex == 0 {
				m.promptInput.Focus()
				m.tempInput.Blur()
			} else {
				m.promptInput.Blur()
				m.tempInput.Focus()
			}
			return m, nil

		case "shift+tab":
			m.focusIndex = (m.focusIndex - 1 + 2) % 2
			if m.focusIndex == 0 {
				m.promptInput.Focus()
				m.tempInput.Blur()
			} else {
				m.promptInput.Blur()
				m.tempInput.Focus()
			}
			return m, nil

		case "ctrl+s":
			// Cycle swarm mode
			m.modeIndex = (m.modeIndex + 1) % len(swarmModes)
			return m, nil

		case "ctrl+d":
			// Cycle voting mode (decision mode)
			m.votingIndex = (m.votingIndex + 1) % len(votingModes)
			return m, nil

		case "enter":
			if m.launching {
				return m, nil
			}
			prompt := m.promptInput.Value()
			if prompt == "" {
				return m, nil
			}
			m.launching = true

			temp := 0.7
			if v := m.tempInput.Value(); v != "" {
				fmt.Sscanf(v, "%f", &temp)
			}

			data := &ChatLaunchData{
				Prompt:      prompt,
				Models:      m.models,
				Mode:        swarmModes[m.modeIndex],
				Temperature: temp,
				VotingMode:  votingModes[m.votingIndex],
			}

			return m, func() tea.Msg {
				return SwitchScreenMsg{
					Screen: ScreenDashboard,
					Data:   data,
				}
			}

		case "esc":
			return m, func() tea.Msg {
				return SwitchScreenMsg{Screen: ScreenModels}
			}
		}
	}

	var cmd tea.Cmd
	if m.focusIndex == 0 {
		m.promptInput, cmd = m.promptInput.Update(msg)
	} else {
		m.tempInput, cmd = m.tempInput.Update(msg)
	}
	return m, cmd
}

func (m ChatModel) View(width, height int) string {
	title := theme.Title.Render("Launch Chat") + "\n" +
		theme.MutedText.Render(fmt.Sprintf("Models: %s", strings.Join(m.models, ", ")))

	modeLabel := theme.Subtitle.Render("Mode: ") +
		theme.ModelName.Render(swarmModes[m.modeIndex])

	votingLabel := theme.Subtitle.Render("Voting: ") +
		theme.ModelName.Render(votingModes[m.votingIndex])

	form := lipgloss.JoinVertical(lipgloss.Left,
		title,
		"",
		m.promptInput.View(),
		"",
		m.tempInput.View(),
		"",
		modeLabel,
		votingLabel,
	)

	if m.launching {
		form += "\n\n" + theme.MutedText.Render("Launching chat...")
	} else if m.err != nil {
		form += "\n\n" + theme.ErrorText.Render("Error: "+m.err.Error())
	}

	hints := "\n\n" +
		theme.KeyName.Render("Tab") + theme.KeyHint.Render(" switch field") +
		"  " + theme.KeyName.Render("Ctrl+S") + theme.KeyHint.Render(" mode") +
		"  " + theme.KeyName.Render("Ctrl+D") + theme.KeyHint.Render(" voting") +
		"  " + theme.KeyName.Render("Enter") + theme.KeyHint.Render(" launch") +
		"  " + theme.KeyName.Render("Esc") + theme.KeyHint.Render(" back")
	form += hints

	box := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(theme.Primary).
		Padding(2, 4).
		Render(form)

	return lipgloss.Place(width, height, lipgloss.Center, lipgloss.Center, box)
}
