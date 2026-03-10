package cmd

import (
	"fmt"
	"log"
	"os"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/jasperan/ai-congress/tui/internal/app"
	sshsrv "github.com/jasperan/ai-congress/tui/internal/ssh"
	"github.com/spf13/cobra"
)

var (
	serverURL string
	sshPort   int
)

var rootCmd = &cobra.Command{
	Use:   "aicongress-tui",
	Short: "AI Congress Terminal Dashboard",
	Long:  "A Bubble Tea TUI for monitoring AI Congress swarm decision-making in real time.",
	RunE: func(cmd *cobra.Command, args []string) error {
		if sshPort > 0 {
			go func() {
				if err := sshsrv.ListenAndServe(sshPort, serverURL); err != nil {
					log.Printf("SSH server error: %v", err)
				}
			}()
		}

		a := app.NewApp(serverURL, false)

		p := tea.NewProgram(a, tea.WithAltScreen())
		a.SetProgram(p)

		if _, err := p.Run(); err != nil {
			return fmt.Errorf("TUI error: %w", err)
		}
		return nil
	},
}

func init() {
	rootCmd.Flags().StringVar(&serverURL, "server", "http://localhost:8000", "AI Congress backend URL")
	rootCmd.Flags().IntVar(&sshPort, "ssh-port", 0, "SSH server port (0 = disabled)")
}

func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
