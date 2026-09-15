package tui

import (
	"strings"
	"testing"

	"github.com/jasperan/ai-congress/gotui/internal/api"
)

// narrowWidths are the widths that have historically broken lipgloss-based
// TUIs: 0 before the first WindowSizeMsg, and the handful of columns a user
// gets shrinking a terminal. A negative width is included because a resize
// event can arrive out of order.
var narrowWidths = []int{-5, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 12}

// sampleCouncil is a fully populated council, so the renderers walk every
// branch rather than the empty-case early return.
func sampleCouncil() *Council {
	council := NewCouncil("Should AI systems be regulated by federal law?", api.ModeDeliberation,
		[]string{"qwen3.5:9b", "gemma4:latest"})
	council.Apply(api.StreamEvent{Type: "start", Message: "Processing with 2 models..."})
	council.Apply(api.StreamEvent{Type: "status_update", Name: "qwen3.5:9b", Status: "Generating..."})
	council.Apply(api.StreamEvent{Type: "model_response", Model: "qwen3.5:9b", Content: "Yes."})
	council.Apply(api.StreamEvent{
		Type:       "final_answer",
		Content:    "Regulate, carefully, with a phased approach and periodic review.",
		Confidence: 0.82,
		Mode:       "deliberation",
		VoteBreakdown: map[string]api.VoteGroup{
			"yes": voteGroup(2.4, "qwen3.5:9b"),
			"no":  voteGroup(1.1, "gemma4:latest"),
		},
		SemanticVote: &api.SemanticVote{
			Consensus:    0.66,
			WinningModel: "qwen3.5:9b",
			Clusters: []api.Cluster{
				{ID: 1, Label: "regulate softly", Models: []string{"qwen3.5:9b"}, KeyClaims: []string{"gradual"}},
			},
		},
		Data: &api.VerdictData{
			Rounds: []api.Round{
				{Name: "opening", Outputs: []api.DebateEntry{
					{Agent: "Pragmatist", Model: "qwen3.5:9b", Response: "ship it", Success: true},
				}},
			},
			FinalPositions: []api.ModelResponse{
				{Agent: "Pragmatist", Model: "qwen3.5:9b", Response: "final", Success: true},
			},
		},
	})
	return council
}

// TestRenderersSurviveNarrowTerminals is the regression guard for the
// documented prior bug: a lipgloss width below its border footprint produced a
// negative content width and panicked. Nothing here may panic, at any width.
func TestRenderersSurviveNarrowTerminals(t *testing.T) {
	council := sampleCouncil()
	models := []api.ModelInfo{{Name: "qwen3.5:9b", Weight: 0.87, Backend: "ollama"}}
	triads := []api.TriadInfo{{Name: "architecture", Description: "built to disagree"}}
	standings := []api.LeaderboardRow{{Model: "qwen3.5:9b", Weight: 0.9, WinRate: 0.5, Participations: 3}}

	renderers := map[string]func(int) string{
		"Pane":               func(w int) string { return Pane("Council", "body text", w) },
		"Pane with no title": func(w int) string { return Pane("", "body text", w) },
		"PaneError":          func(w int) string { return PaneError("Problem", errUnreachable(), w) },
		"Header":             func(w int) string { return Header("http://127.0.0.1:8000", "deliberation", w) },
		"Footer":             func(w int) string { return Footer("tab next - enter submit - ctrl+c quit", w) },
		"TallyRows":          func(w int) string { return TallyRows(council, w) },
		"MemberRows":         func(w int) string { return MemberRows(council, w) },
		"RoundRows":          func(w int) string { return RoundRows(council, w) },
		"TranscriptRows":     func(w int) string { return TranscriptRows(council, w) },
		"VerdictRows":        func(w int) string { return VerdictRows(council, w) },
		"SemanticRows":       func(w int) string { return SemanticRows(council, w) },
		"LeaderboardRows":    func(w int) string { return LeaderboardRows(standings, w) },
		"ModelRows":          func(w int) string { return ModelRows(models, w) },
		"TriadRows":          func(w int) string { return TriadRows(triads, w) },
		"Bar":                func(w int) string { return Bar(0.5, w) },
	}

	for name, render := range renderers {
		for _, width := range narrowWidths {
			func() {
				defer func() {
					if recovered := recover(); recovered != nil {
						t.Errorf("%s panicked at width %d: %v", name, width, recovered)
					}
				}()
				render(width)
			}()
		}
	}
}

// errUnreachable is a stand-in error for the error pane.
func errUnreachable() error { return api.ErrUnreachable }

func TestPaneAlwaysRendersItsBody(t *testing.T) {
	for _, width := range narrowWidths {
		rendered := Pane("Title", "the body", width)
		if !strings.Contains(rendered, "the body") {
			t.Errorf("Pane at width %d dropped its body:\n%s", width, rendered)
		}
	}
}

// TestClampWidthNeverGoesBelowTheFloor is the mechanism behind the guard.
func TestClampWidthNeverGoesBelowTheFloor(t *testing.T) {
	for _, width := range narrowWidths {
		if got := clampWidth(width); got < MinPaneWidth {
			t.Errorf("clampWidth(%d) = %d, want at least %d", width, got, MinPaneWidth)
		}
		if got := contentWidth(width); got < 1 {
			t.Errorf("contentWidth(%d) = %d, want at least 1", width, got)
		}
	}
}

func TestTruncateNeverExceedsItsLimit(t *testing.T) {
	inputs := []string{"", "a", "qwen3.5:9b", "a much longer model name than expected", "ünïcödé"}
	for _, in := range inputs {
		for _, n := range []int{-3, 0, 1, 2, 5, 40} {
			got := Truncate(in, n)
			if n <= 1 {
				if got != "" {
					t.Errorf("Truncate(%q, %d) = %q, want empty", in, n, got)
				}
				continue
			}
			if len([]rune(got)) > n {
				t.Errorf("Truncate(%q, %d) = %q, which is longer than the limit", in, n, got)
			}
		}
	}
}

// TestTruncateKeepsValidUTF8 guards against splitting a multi-byte model name.
func TestTruncateKeepsValidUTF8(t *testing.T) {
	got := Truncate("qwen✓模型:9b", 5)
	if !strings.HasPrefix(got, "qwe") {
		t.Errorf("Truncate() = %q, want a prefix of the input", got)
	}
}

func TestBarClampsOutOfRangeFractions(t *testing.T) {
	cases := map[float64]int{-1: 0, 0: 0, 1: 10, 2: 10}
	for fraction, want := range cases {
		bar := Bar(fraction, 10)
		if strings.Count(bar, "#") != want {
			t.Errorf("Bar(%v, 10) = %q, want %d filled cells", fraction, bar, want)
		}
		if len([]rune(bar)) != 10 {
			t.Errorf("Bar(%v, 10) = %q, want exactly 10 cells", fraction, bar)
		}
	}
	// A non-positive width must still produce something, not a panic.
	if Bar(0.5, 0) == "" {
		t.Error("Bar with a zero width produced nothing")
	}
}

func TestPercentHandlesNaN(t *testing.T) {
	nan := 0.0
	nan = nan / nan
	if got := Percent(nan); !strings.Contains(got, "n/a") {
		t.Errorf("Percent(NaN) = %q, want a placeholder rather than a nonsense number", got)
	}
	if got := Percent(0.825); got != "83%" {
		t.Errorf("Percent(0.825) = %q, want 83%%", got)
	}
	// A fraction outside 0..1 cannot be a share of anything; clamping to the
	// range beats printing a negative percentage.
	if got := Percent(-1); got != "0%" {
		t.Errorf("Percent(-1) = %q, want 0%%", got)
	}
	if got := Percent(5); got != "100%" {
		t.Errorf("Percent(5) = %q, want 100%%", got)
	}
}

func TestPadNeverShortens(t *testing.T) {
	if got := Pad("abc", 6); got != "abc   " {
		t.Errorf("Pad() = %q", got)
	}
	if got := Pad("abcdef", 3); got != "abcdef" {
		t.Errorf("Pad() shortened the input to %q", got)
	}
}
