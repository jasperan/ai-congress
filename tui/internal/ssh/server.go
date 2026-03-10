package ssh

import (
	"fmt"

	tea "github.com/charmbracelet/bubbletea"
	cssh "github.com/charmbracelet/ssh"
	"github.com/charmbracelet/wish"
	bm "github.com/charmbracelet/wish/bubbletea"
	"github.com/jasperan/ai-congress/tui/internal/app"
)

func ListenAndServe(port int, serverURL string) error {
	s, err := wish.NewServer(
		wish.WithAddress(fmt.Sprintf(":%d", port)),
		wish.WithMiddleware(
			bm.Middleware(func(sess cssh.Session) (tea.Model, []tea.ProgramOption) {
				a := app.NewApp(serverURL, true)
				return a, []tea.ProgramOption{tea.WithAltScreen()}
			}),
		),
	)
	if err != nil {
		return fmt.Errorf("failed to create SSH server: %w", err)
	}

	return s.ListenAndServe()
}
