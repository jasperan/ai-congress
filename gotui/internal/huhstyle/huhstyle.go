// Package huhstyle adapts charmbracelet/huh forms to the AI Congress palette.
//
// The 14 tokens below are the same Catppuccin Mocha values the Rust TUI
// already ships in tui-rs/src/theme.rs (BG, SURFACE, ELEVATED, HIGHEST, TEXT,
// SUBTEXT, MUTED, DIM, PRIMARY, SECONDARY, INFO, SUCCESS, WARNING, ERROR).
// That file is the source of truth: if the Rust palette changes, change this
// one with it, or the two TUIs will drift apart.
//
// huh's built-in ThemeCatppuccin() is close but not the same: it also paints
// with the auxiliary shades subtext1, overlay1, pink and rosewater, which are
// not part of the Rust TUI's token set. Pass Theme via WithTheme instead.
package huhstyle

import (
	"image/color"
	"os"

	"charm.land/huh/v2"
	"charm.land/lipgloss/v2"
	"golang.org/x/term"
)

// Palette tokens, mirrored from tui-rs/src/theme.rs. Do not add, rename or
// re-value without updating that file too.
const (
	HexBG        = "#1e1e2e" // bg / base
	HexSurface   = "#181825" // surface
	HexElevated  = "#313244" // elevated
	HexHighest   = "#45475a" // highest
	HexText      = "#cdd6f4" // text
	HexSubtext   = "#a6adc8" // subtext
	HexMuted     = "#6c7086" // muted
	HexDim       = "#585b70" // dim
	HexPrimary   = "#89b4fa" // primary (Democrat in the Rust party mapping)
	HexSecondary = "#cba6f7" // secondary
	HexInfo      = "#89dceb" // info
	HexSuccess   = "#a6e3a1" // success (Yea)
	HexWarning   = "#f9e2af" // warning (Independent, abstain)
	HexError     = "#f38ba8" // error (Republican, Nay)
)

var (
	colBG        = lipgloss.Color(HexBG)
	colSurface   = lipgloss.Color(HexSurface)
	colElevated  = lipgloss.Color(HexElevated)
	colHighest   = lipgloss.Color(HexHighest)
	colText      = lipgloss.Color(HexText)
	colSubtext   = lipgloss.Color(HexSubtext)
	colMuted     = lipgloss.Color(HexMuted)
	colDim       = lipgloss.Color(HexDim)
	colPrimary   = lipgloss.Color(HexPrimary)
	colSecondary = lipgloss.Color(HexSecondary)
	colInfo      = lipgloss.Color(HexInfo)
	colSuccess   = lipgloss.Color(HexSuccess)
	colWarning   = lipgloss.Color(HexWarning)
	colError     = lipgloss.Color(HexError)
)

// Tokens exposes the palette for styling outside huh, so no other package has
// to declare a colour literal.
//
// It returns color.Color rather than lipgloss.Color because in lipgloss v2
// lipgloss.Color is a constructor function, not a type.
func Tokens() (bg, surface, elevated, highest, text, subtext, muted, dim,
	primary, secondary, info, success, warning, danger color.Color) {
	return colBG, colSurface, colElevated, colHighest,
		colText, colSubtext, colMuted, colDim,
		colPrimary, colSecondary, colInfo, colSuccess, colWarning, colError
}

// Accessible reports whether the user requested screen-reader mode.
//
// Wire it into every form: huh then swaps the TUI for plain prompts, which is
// the only sane rendering when a screen reader is driving the terminal.
func Accessible() bool { return os.Getenv("ACCESSIBLE") != "" }

// Interactive reports whether stdin is a terminal, i.e. whether a form can
// actually be driven by a human.
//
// Callers must route around a prompt when this is false rather than into it: a
// huh form attached to a pipe or a cron job can never be answered.
//
// This must use term.IsTerminal, NOT an os.ModeCharDevice probe: /dev/null IS a
// character device, so a char-device check reports "interactive" for exactly
// the case it is supposed to reject, and a form would then hang forever.
func Interactive() bool { return term.IsTerminal(int(os.Stdin.Fd())) }

// NOTE ON ACCESSIBILITY, measured against huh v2.0.3:
//
// WithAccessible is only consulted inside Form.RunWithContext. Form.Init,
// Update and View never read the flag, so an *embedded* form (driven as a
// Bubble Tea component) renders identically with and without it: screen-reader
// support is real only for a form run standalone via Form.Run().
//
// Pass WithAccessible(Accessible()) anyway -- it is the correct intent and
// costs nothing -- but never claim accessible support for an embedded form.
// The flows that must be accessible are run standalone on the Accessible()
// path in cmd/ai-congress-tui instead.

// Styles builds token-compliant form styles.
func Styles() *huh.Styles {
	t := huh.ThemeBase(true)

	// Focused field: rounded box, primary border. Focus is a border colour
	// change, never a thickness change, so nothing reflows on tab.
	t.Focused.Base = lipgloss.NewStyle().
		Padding(0, 1).
		Border(lipgloss.RoundedBorder()).
		BorderForeground(colPrimary)
	t.Focused.Card = t.Focused.Base

	t.Focused.Title = t.Focused.Title.Bold(true).Foreground(colPrimary)
	t.Focused.NoteTitle = t.Focused.NoteTitle.Bold(true).Foreground(colPrimary)
	t.Focused.Description = t.Focused.Description.Foreground(colSubtext)
	t.Focused.Directory = t.Focused.Directory.Foreground(colInfo)
	t.Focused.File = t.Focused.File.Foreground(colText)

	// Errors are the one place a semantic colour is mandatory.
	t.Focused.ErrorIndicator = t.Focused.ErrorIndicator.Foreground(colError)
	t.Focused.ErrorMessage = t.Focused.ErrorMessage.Foreground(colError)

	// Select: the cursor row is an inverted selection.
	t.Focused.SelectSelector = t.Focused.SelectSelector.
		Foreground(colBG).
		Background(colPrimary).
		Bold(true)
	t.Focused.Option = t.Focused.Option.Foreground(colText)
	t.Focused.NextIndicator = t.Focused.NextIndicator.Foreground(colPrimary)
	t.Focused.PrevIndicator = t.Focused.PrevIndicator.Foreground(colPrimary)

	// Multi-select: chosen items are accent-coloured, not success-coloured
	// (selection is not a success event).
	t.Focused.MultiSelectSelector = t.Focused.MultiSelectSelector.Foreground(colPrimary)
	t.Focused.SelectedPrefix = t.Focused.SelectedPrefix.Foreground(colPrimary).Bold(true)
	t.Focused.SelectedOption = t.Focused.SelectedOption.Foreground(colPrimary).Bold(true)
	t.Focused.UnselectedPrefix = t.Focused.UnselectedPrefix.Foreground(colMuted)
	t.Focused.UnselectedOption = t.Focused.UnselectedOption.Foreground(colText)

	t.Focused.FocusedButton = t.Focused.FocusedButton.
		Foreground(colBG).
		Background(colPrimary).
		Bold(true)
	t.Focused.BlurredButton = t.Focused.BlurredButton.
		Foreground(colText).
		Background(colElevated)

	t.Focused.TextInput.Cursor = t.Focused.TextInput.Cursor.Foreground(colInfo)
	t.Focused.TextInput.CursorText = t.Focused.TextInput.CursorText.
		Foreground(colBG).
		Background(colInfo)
	t.Focused.TextInput.Placeholder = t.Focused.TextInput.Placeholder.Foreground(colMuted)
	t.Focused.TextInput.Prompt = t.Focused.TextInput.Prompt.Foreground(colPrimary)
	t.Focused.TextInput.Text = t.Focused.TextInput.Text.Foreground(colText)

	// Blurred fields keep their structure but lose the accent.
	t.Blurred = t.Focused
	t.Blurred.Base = t.Blurred.Base.BorderForeground(colDim)
	t.Blurred.Card = t.Blurred.Base
	t.Blurred.FocusedButton = t.Blurred.BlurredButton
	t.Blurred.MultiSelectSelector = lipgloss.NewStyle().SetString("  ")
	t.Blurred.NextIndicator = lipgloss.NewStyle()
	t.Blurred.PrevIndicator = lipgloss.NewStyle()

	t.Help.Ellipsis = t.Help.Ellipsis.Foreground(colMuted)
	t.Help.ShortKey = t.Help.ShortKey.Foreground(colSubtext)
	t.Help.ShortDesc = t.Help.ShortDesc.Foreground(colMuted)
	t.Help.ShortSeparator = t.Help.ShortSeparator.Foreground(colDim)
	t.Help.FullKey = t.Help.FullKey.Foreground(colSubtext)
	t.Help.FullDesc = t.Help.FullDesc.Foreground(colMuted)
	t.Help.FullSeparator = t.Help.FullSeparator.Foreground(colDim)

	t.Group.Title = t.Focused.Title
	t.Group.Description = t.Focused.Description

	return t
}

// Theme adapts Styles to huh's Theme interface so it can be passed to
// Form.WithTheme. The isDark argument is ignored: this project ships a single
// dark palette by design.
func Theme(bool) *huh.Styles { return Styles() }
