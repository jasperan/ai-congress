// tui/internal/app/app.go
package app

import (
	tea "github.com/charmbracelet/bubbletea"
	"github.com/jasperan/ai-congress/tui/internal/api"
	"github.com/jasperan/ai-congress/tui/internal/components"
)

// ProgramRef holds a shared reference to the tea.Program.
type ProgramRef struct {
	P *tea.Program
}

// Screen identifies the active screen.
type Screen int

const (
	ScreenSplash Screen = iota
	ScreenModels
	ScreenChat
	ScreenDashboard
	ScreenHistory
)

// SwitchScreenMsg requests a transition to a different screen.
type SwitchScreenMsg struct {
	Screen Screen
	Data   interface{}
}

// WSEventMsg wraps a WebSocket event for the Bubble Tea update loop.
type WSEventMsg struct {
	Event api.WSMessage
}

// App is the root Bubble Tea model.
type App struct {
	client     *api.Client
	wsClient   *api.WSClient
	readOnly   bool
	width      int
	height     int
	screen     Screen
	throughput *components.ThroughputTracker

	splash    SplashModel
	models    ModelsModel
	chat      ChatModel
	dashboard DashboardModel
	history   HistoryModel

	showHelp bool
	help     HelpModel

	programRef *ProgramRef
}

// NewApp creates the root application model.
func NewApp(serverURL string, readOnly bool) App {
	client := api.NewClient(serverURL)
	wsClient := api.NewWSClient(serverURL)
	tp := components.NewThroughputTracker()

	return App{
		client:     client,
		wsClient:   wsClient,
		readOnly:   readOnly,
		screen:     ScreenSplash,
		throughput: tp,
		splash:     NewSplashModel(client),
		programRef: &ProgramRef{},
	}
}

// SetProgram stores the tea.Program reference for the WS bridge.
func (a *App) SetProgram(p *tea.Program) {
	a.programRef.P = p
}

// Init delegates to the splash screen.
func (a App) Init() tea.Cmd {
	return a.splash.Init()
}

// Update handles global keys, screen switching, and delegates to sub-models.
func (a App) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		a.width = msg.Width
		a.height = msg.Height
		return a, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c":
			a.wsClient.Close()
			return a, tea.Quit
		case "f1":
			a.showHelp = !a.showHelp
			return a, nil
		}

	case SwitchScreenMsg:
		return a.switchScreen(msg)

	case WSEventMsg:
		if a.screen == ScreenDashboard {
			var cmd tea.Cmd
			a.dashboard, cmd = a.dashboard.Update(msg)
			return a, cmd
		}
		return a, nil
	}

	// Delegate to current screen
	switch a.screen {
	case ScreenSplash:
		var cmd tea.Cmd
		a.splash, cmd = a.splash.Update(msg)
		return a, cmd
	case ScreenModels:
		var cmd tea.Cmd
		a.models, cmd = a.models.Update(msg)
		return a, cmd
	case ScreenChat:
		var cmd tea.Cmd
		a.chat, cmd = a.chat.Update(msg)
		return a, cmd
	case ScreenDashboard:
		var cmd tea.Cmd
		a.dashboard, cmd = a.dashboard.Update(msg)
		return a, cmd
	case ScreenHistory:
		var cmd tea.Cmd
		a.history, cmd = a.history.Update(msg)
		return a, cmd
	}

	return a, nil
}

// View renders the current screen, with an optional help overlay.
func (a App) View() string {
	var content string

	switch a.screen {
	case ScreenSplash:
		content = a.splash.View(a.width, a.height)
	case ScreenModels:
		content = a.models.View(a.width, a.height)
	case ScreenChat:
		content = a.chat.View(a.width, a.height)
	case ScreenDashboard:
		content = a.dashboard.View(a.width, a.height)
	case ScreenHistory:
		content = a.history.View(a.width, a.height)
	default:
		content = "Unknown screen"
	}

	if a.showHelp {
		content = a.renderHelpOverlay(content)
	}

	return content
}

// switchScreen creates a fresh sub-model for the target screen and initialises it.
func (a App) switchScreen(msg SwitchScreenMsg) (tea.Model, tea.Cmd) {
	if a.screen == ScreenDashboard {
		a.wsClient.Close()
	}

	a.screen = msg.Screen

	switch msg.Screen {
	case ScreenSplash:
		a.splash = NewSplashModel(a.client)
		return a, a.splash.Init()

	case ScreenModels:
		a.models = NewModelsModel(a.client)
		return a, a.models.Init()

	case ScreenChat:
		selectedModels, _ := msg.Data.([]string)
		a.chat = NewChatModel(a.client, selectedModels)
		return a, a.chat.Init()

	case ScreenDashboard:
		chatData, _ := msg.Data.(*ChatLaunchData)
		a.dashboard = NewDashboardModel(a.client, a.throughput, a.readOnly)
		cmd := a.dashboard.Init()

		// Start WS connection and send chat request
		if chatData != nil && a.programRef != nil && a.programRef.P != nil {
			prog := a.programRef.P
			wsClient := a.wsClient
			go func() {
				err := wsClient.Connect(func(wsMsg api.WSMessage) {
					prog.Send(WSEventMsg{Event: wsMsg})
				})
				if err != nil {
					prog.Send(WSEventMsg{Event: api.WSMessage{Type: "error", Message: "WebSocket: " + err.Error()}})
					return
				}
				// Send the chat request
				wsClient.SendChat(api.WSChatRequest{
					Prompt:      chatData.Prompt,
					Models:      chatData.Models,
					Mode:        chatData.Mode,
					Stream:      true,
					Temperature: chatData.Temperature,
					VotingMode:  chatData.VotingMode,
				})
			}()
		}

		return a, cmd

	case ScreenHistory:
		a.history = NewHistoryModel(a.client)
		return a, a.history.Init()
	}

	return a, nil
}

func (a App) renderHelpOverlay(background string) string {
	return a.help.Overlay(background, a.width, a.height, a.screen)
}
