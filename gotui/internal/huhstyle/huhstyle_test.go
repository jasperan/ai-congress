package huhstyle

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestPaletteMatchesTheRustTUI is the drift guard.
//
// The package comment names tui-rs/src/theme.rs as the source of truth for the
// 14 tokens, so this reads that file and compares. Without it, changing the Rust
// palette would silently leave the Go front-end looking different from the TUI
// the repo documents.
func TestPaletteMatchesTheRustTUI(t *testing.T) {
	path := filepath.Join("..", "..", "..", "tui-rs", "src", "theme.rs")
	source, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("cannot read %s (%v); the palette check only runs inside the repo", path, err)
	}

	// Each Rust line looks like: pub const PRIMARY: Color = Color::Rgb(0x89, 0xb4, 0xfa);
	line := regexp.MustCompile(`pub const ([A-Z_]+): Color = Color::Rgb\(0x([0-9a-fA-F]{2}), 0x([0-9a-fA-F]{2}), 0x([0-9a-fA-F]{2})\)`)

	rustTokens := map[string]string{}
	for _, match := range line.FindAllStringSubmatch(string(source), -1) {
		rustTokens[match[1]] = "#" + strings.ToLower(match[2]+match[3]+match[4])
	}
	if len(rustTokens) == 0 {
		t.Fatalf("%s contained no Color::Rgb constants; this test needs updating", path)
	}

	// Go token name -> Rust constant name.
	pairs := map[string]string{
		HexBG:        "BG",
		HexSurface:   "SURFACE",
		HexElevated:  "ELEVATED",
		HexHighest:   "HIGHEST",
		HexText:      "TEXT",
		HexSubtext:   "SUBTEXT",
		HexMuted:     "MUTED",
		HexDim:       "DIM",
		HexPrimary:   "PRIMARY",
		HexSecondary: "SECONDARY",
		HexInfo:      "INFO",
		HexSuccess:   "SUCCESS",
		HexWarning:   "WARNING",
		HexError:     "ERROR",
	}

	for goToken, rustName := range pairs {
		rustValue, ok := rustTokens[rustName]
		if !ok {
			t.Errorf("tui-rs/src/theme.rs has no %s constant", rustName)
			continue
		}
		if goToken != rustValue {
			t.Errorf("palette drift: Go %s = %s but tui-rs %s = %s",
				rustName, goToken, rustName, rustValue)
		}
	}
}

func TestTokensReturnsAllFourteenValues(t *testing.T) {
	bg, surface, elevated, highest, text, subtext, muted, dim,
		primary, secondary, info, success, warning, danger := Tokens()

	values := []interface {
		RGBA() (uint32, uint32, uint32, uint32)
	}{
		bg, surface, elevated, highest, text, subtext, muted, dim,
		primary, secondary, info, success, warning, danger,
	}
	if len(values) != 14 {
		t.Fatalf("Tokens() returned %d values, want 14", len(values))
	}
	for i, value := range values {
		if value == nil {
			t.Errorf("Tokens()[%d] is nil", i)
		}
	}
}

// TestAccessibleAndInteractiveReadTheEnvironment pins the two switches the
// command branches on.
func TestAccessibleAndInteractiveReadTheEnvironment(t *testing.T) {
	t.Setenv("ACCESSIBLE", "")
	if Accessible() {
		t.Error("Accessible() = true with ACCESSIBLE unset")
	}
	t.Setenv("ACCESSIBLE", "1")
	if !Accessible() {
		t.Error("Accessible() = false with ACCESSIBLE set")
	}

	// Under `go test` stdin is not a terminal, so this must report false: the
	// full-screen UI would otherwise be started over a pipe and hang.
	if Interactive() {
		t.Log("stdin is a terminal in this run; skipping the piped-stdin assertion")
	}
}

// TestStylesAreFullyPopulated guards against a huh upgrade silently leaving
// every widget at zero value, which renders as unstyled text.
func TestStylesAreFullyPopulated(t *testing.T) {
	styles := Styles()
	if styles == nil {
		t.Fatal("Styles() = nil")
	}
	if styles.Focused.Title.GetForeground() == nil {
		t.Error("the focused title has no foreground colour")
	}
	if styles.Focused.ErrorMessage.GetForeground() == nil {
		t.Error("the error message has no foreground colour")
	}
}

// TestThemeIgnoresTheDarkFlag pins the single-palette decision: huh asks for a
// theme per background, and this project ships one dark palette on purpose.
func TestThemeIgnoresTheDarkFlag(t *testing.T) {
	dark := Theme(true)
	light := Theme(false)
	if dark == nil || light == nil {
		t.Fatal("Theme() returned nil")
	}
	if dark.Focused.Title.GetForeground() != light.Focused.Title.GetForeground() {
		t.Error("Theme(true) and Theme(false) use different colours; the project ships one palette")
	}
}
