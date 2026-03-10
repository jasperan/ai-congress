// tui/internal/app/splash.go
package app

import (
	"fmt"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/harmonica"
	"github.com/charmbracelet/lipgloss"
	"github.com/jasperan/ai-congress/tui/internal/api"
	"github.com/jasperan/ai-congress/tui/internal/theme"
)

const banner = ` █████╗ ██╗     ██████╗ ██████╗ ███╗   ██╗ ██████╗ ██████╗ ███████╗███████╗███████╗
██╔══██╗██║    ██╔════╝██╔═══██╗████╗  ██║██╔════╝ ██╔══██╗██╔════╝██╔════╝██╔════╝
███████║██║    ██║     ██║   ██║██╔██╗ ██║██║  ███╗██████╔╝█████╗  ███████╗███████╗
██╔══██║██║    ██║     ██║   ██║██║╚██╗██║██║   ██║██╔══██╗██╔══╝  ╚════██║╚════██║
██║  ██║██║    ╚██████╗╚██████╔╝██║ ╚████║╚██████╔╝██║  ██║███████╗███████║███████║
╚═╝  ╚═╝╚═╝     ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝`

// --- Messages ---

type connectionCheckMsg struct {
	err        error
	modelCount int
}

type fadeTickMsg time.Time

// --- Model ---

type SplashModel struct {
	client     *api.Client
	connected  bool
	checking   bool
	errMsg     string
	modelCount int
	opacity    float64
	velocity   float64
	spring     harmonica.Spring
}

func NewSplashModel(client *api.Client) SplashModel {
	return SplashModel{
		client:  client,
		opacity: 0,
		spring:  harmonica.NewSpring(harmonica.FPS(60), 5.0, 0.4),
	}
}

func (m SplashModel) Init() tea.Cmd {
	return tea.Batch(
		m.checkConnection(),
		m.fadeTick(),
	)
}

func (m SplashModel) Update(msg tea.Msg) (SplashModel, tea.Cmd) {
	switch msg := msg.(type) {
	case connectionCheckMsg:
		m.checking = false
		if msg.err != nil {
			m.connected = false
			m.errMsg = msg.err.Error()
		} else {
			m.connected = true
			m.errMsg = ""
			m.modelCount = msg.modelCount
		}
		return m, nil

	case fadeTickMsg:
		if m.opacity < 1.0 {
			m.opacity, m.velocity = m.spring.Update(m.opacity, m.velocity, 1.0)
			if m.opacity > 1.0 {
				m.opacity = 1.0
			}
			return m, m.fadeTick()
		}
		return m, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "enter":
			if m.connected {
				return m, func() tea.Msg {
					return SwitchScreenMsg{Screen: ScreenModels}
				}
			}
		case "r":
			m.checking = true
			m.errMsg = ""
			return m, m.checkConnection()
		case "q":
			return m, tea.Quit
		}
	}

	return m, nil
}

func (m SplashModel) View(width, height int) string {
	alpha := m.opacity
	if alpha < 0 {
		alpha = 0
	}
	bannerColor := lerpColor("#1C1C1E", "#5f87ff", alpha)
	bannerStyle := lipgloss.NewStyle().Foreground(lipgloss.Color(bannerColor))
	renderedBanner := bannerStyle.Render(banner)

	var status string
	if m.checking {
		status = theme.MutedText.Render("⟳ Connecting to backend...")
	} else if m.connected {
		status = theme.StatusActive.Render(theme.StatusDot) + " " +
			theme.Title.Render(fmt.Sprintf("Connected — %d models available", m.modelCount))
	} else if m.errMsg != "" {
		status = theme.StatusError.Render(theme.StatusDot) + " " +
			theme.ErrorText.Render("Connection failed: "+m.errMsg)
	} else {
		status = theme.MutedText.Render("Waiting...")
	}

	var hints string
	if m.connected {
		hints = theme.KeyName.Render("Enter") + theme.KeyHint.Render(" browse models") +
			"  " + theme.KeyName.Render("q") + theme.KeyHint.Render(" quit")
	} else {
		hints = theme.KeyName.Render("r") + theme.KeyHint.Render(" retry") +
			"  " + theme.KeyName.Render("q") + theme.KeyHint.Render(" quit")
	}

	content := lipgloss.JoinVertical(lipgloss.Center,
		renderedBanner,
		"",
		status,
		"",
		hints,
	)

	return lipgloss.Place(width, height, lipgloss.Center, lipgloss.Center, content)
}

func (m SplashModel) checkConnection() tea.Cmd {
	client := m.client
	return func() tea.Msg {
		err := client.Ping()
		if err != nil {
			return connectionCheckMsg{err: err}
		}
		models, err := client.ListModels()
		if err != nil {
			return connectionCheckMsg{err: err}
		}
		return connectionCheckMsg{modelCount: len(models)}
	}
}

func (m SplashModel) fadeTick() tea.Cmd {
	return tea.Tick(time.Second/60, func(t time.Time) tea.Msg {
		return fadeTickMsg(t)
	})
}

func lerpColor(from, to string, t float64) string {
	fr, fg, fb := hexToRGB(from)
	tr, tg, tb := hexToRGB(to)

	r := fr + int(float64(tr-fr)*t)
	g := fg + int(float64(tg-fg)*t)
	b := fb + int(float64(tb-fb)*t)

	return fmt.Sprintf("#%02x%02x%02x", r, g, b)
}

func hexToRGB(hex string) (int, int, int) {
	if len(hex) == 7 && hex[0] == '#' {
		var r, g, b int
		fmt.Sscanf(hex[1:], "%02x%02x%02x", &r, &g, &b)
		return r, g, b
	}
	return 0, 0, 0
}
