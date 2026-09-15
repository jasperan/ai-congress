package tui

import (
	"context"
	"errors"
	"strings"
	"testing"

	"charm.land/bubbletea/v2"
	"charm.land/huh/v2"
	"github.com/jasperan/ai-congress/gotui/internal/api"
	"github.com/jasperan/ai-congress/gotui/internal/session"
)

// These tests drive the model by handing it messages directly. They never drain
// tea.Cmd, because huh re-arms the text input's cursor blink on every update and
// each blink tick sleeps for about half a second; a naive drain makes the suite
// hang. Commands are inspected for their type instead.

func testOptions(t *testing.T) Options {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	t.Setenv("HOME", dir)
	// A blank value means "not set" for both of these.
	t.Setenv("ACCESSIBLE", "")
	t.Setenv(session.EnvAPIKey, "")
	t.Setenv(session.EnvOraclePassword, "")
	return Options{ProjectRoot: dir, Settings: session.Defaults()}
}

func TestNewStartsOnTheConnectForm(t *testing.T) {
	model := New(testOptions(t))
	// Render the way a running program does. huh only builds a group's view
	// once Form.Init has activated it, and it is Init that the bubbletea
	// runtime calls before the first paint.
	_ = model.Init()
	_, _ = model.Update(tea.WindowSizeMsg{Width: 100, Height: 32})

	if model.screen != ScreenConnect {
		t.Fatalf("expected the connect screen, got %v", model.screen)
	}
	if model.form == nil {
		t.Fatal("the connect form must be built up front")
	}
	if model.form.State != huh.StateNormal {
		t.Fatalf("expected a normal form state, got %v", model.form.State)
	}
	if view := model.View().Content; !strings.Contains(view, "Service URL") {
		t.Fatalf("the connect form should render its first question, got %q", view)
	}
}

// TestKeyReleaseDoesNotAdvanceTheForm pins the bubbletea v2 press/release bug:
// handling both would fire every binding twice per keystroke.
func TestKeyReleaseDoesNotAdvanceTheForm(t *testing.T) {
	model := New(testOptions(t))

	_, _ = model.Update(tea.KeyReleaseMsg{Code: 'a'})
	if model.form == nil {
		t.Fatal("a key release must not complete or clear the form")
	}

	_, _ = model.Update(tea.KeyPressMsg{Code: 'q', Text: "q"})
	if model.screen != ScreenConnect {
		t.Fatalf("typing must stay on the connect screen, got %v", model.screen)
	}
}

// TestNarrowWindowDoesNotPanic is the model-level half of the narrow-width
// guard: a WindowSizeMsg of 0 arrives before the first real size, and a user can
// shrink a terminal to a few columns.
func TestNarrowWindowDoesNotPanic(t *testing.T) {
	model := New(testOptions(t))
	_ = model.Init()

	for _, width := range []int{0, 1, 2, 3, 4, 5, 6, 20} {
		func() {
			defer func() {
				if recovered := recover(); recovered != nil {
					t.Errorf("Update/View panicked at width %d: %v", width, recovered)
				}
			}()
			_, _ = model.Update(tea.WindowSizeMsg{Width: width, Height: 10})
			_ = model.View().Content
		}()
	}
}

func TestCtrlCQuits(t *testing.T) {
	model := New(testOptions(t))
	_, cmd := model.Update(tea.KeyPressMsg{Code: 'c', Mod: tea.ModCtrl})
	if cmd == nil {
		t.Fatal("ctrl+c produced no command")
	}
	// The command must be tea.Quit. Comparing the rendered message is the only
	// stable way without running the runtime.
	msg := cmd()
	if _, ok := msg.(tea.QuitMsg); !ok {
		t.Fatalf("ctrl+c produced %T, want tea.QuitMsg", msg)
	}
}

// TestStreamEventFoldsIntoTheCouncilAndReArms checks the stream pump: each
// frame must both update the council and schedule the next read.
func TestStreamEventFoldsIntoTheCouncilAndReArms(t *testing.T) {
	model := New(testOptions(t))
	_ = model.Init()
	model.screen = ScreenRunning
	model.councilState = NewCouncil("regulate?", api.ModeMultiModel, []string{"a", "b"})
	model.stream = &api.Stream{} // a non-nil stream is all the pump checks for

	_, cmd := model.Update(streamEventMsg{event: api.StreamEvent{
		Type: "model_response", Model: "a", Content: "yes",
	}})

	if model.councilState.Completed() != 1 {
		t.Errorf("Completed() = %d, want 1", model.councilState.Completed())
	}
	if cmd == nil {
		t.Fatal("the stream pump was not re-armed, so the run would stall after one frame")
	}
	if model.screen != ScreenRunning {
		t.Errorf("screen = %v, want it to stay on the run", model.screen)
	}
}

// TestStreamCloseMovesToTheResultScreen checks that a finished run always lands
// somewhere the user can act from, even when the service failed.
func TestStreamCloseMovesToTheResultScreen(t *testing.T) {
	model := New(testOptions(t))
	_ = model.Init()
	model.screen = ScreenRunning
	model.councilState = NewCouncil("regulate?", api.ModeMultiModel, []string{"a"})
	model.stream = &api.Stream{}

	_, cmd := model.Update(streamClosedMsg{err: errors.New("connection lost")})

	if model.screen != ScreenResult {
		t.Fatalf("screen = %v, want the result screen", model.screen)
	}
	if model.form == nil {
		t.Fatal("the result screen must offer a way onward")
	}
	if cmd == nil {
		t.Error("the result form was not initialised")
	}
	if model.failure == nil || !strings.Contains(model.failure.Error(), "connection lost") {
		t.Errorf("failure = %v, want the stream error reported", model.failure)
	}
	if model.StreamError() == nil {
		t.Error("StreamError() = nil, so the command cannot report the failure after the screen closes")
	}
}

// TestCancelledStreamIsNotAFailure: aborting with ctrl+c closes the socket,
// which surfaces as a read error. Reporting that as a problem would be noise.
func TestCancelledStreamIsNotAFailure(t *testing.T) {
	model := New(testOptions(t))
	_ = model.Init()
	model.screen = ScreenRunning
	model.councilState = NewCouncil("q", api.ModeMultiModel, []string{"a"})
	model.stream = &api.Stream{}

	_, _ = model.Update(streamClosedMsg{err: context.Canceled})

	if model.failure != nil {
		t.Errorf("failure = %v, want nil for a user-initiated abort", model.failure)
	}
}

func TestStreamCloseFailsMembersThatNeverReported(t *testing.T) {
	model := New(testOptions(t))
	_ = model.Init()
	model.screen = ScreenRunning
	model.councilState = NewCouncil("q", api.ModeMultiModel, []string{"a", "b"})
	model.stream = &api.Stream{}

	_, _ = model.Update(streamClosedMsg{err: nil})

	for _, member := range model.councilState.Members {
		if member.Status == MemberPending {
			t.Errorf("member %q is still pending after the stream closed", member.Name)
		}
	}
}

func TestStandingsReadoutRenders(t *testing.T) {
	model := New(testOptions(t))
	_ = model.Init()
	model.screen = ScreenStandings
	model.busy = "Loading standings"

	_, cmd := model.Update(standingsLoadedMsg{rows: []api.LeaderboardRow{
		{Model: "qwen3.5:9b", Weight: 0.9, WinRate: 0.5, Participations: 4},
	}})

	if cmd == nil {
		t.Error("the readout form was not initialised")
	}
	view := model.View().Content
	for _, want := range []string{"Model standings", "qwen3.5:9b", "0.900"} {
		if !strings.Contains(view, want) {
			t.Errorf("standings view is missing %q:\n%s", want, view)
		}
	}
	if model.form == nil {
		t.Fatal("the readout must hand control back with a form")
	}
}

// TestControlRoomReadoutSurvivesTheRealServiceShape checks the untyped map is
// rendered without guessing, and that an unknown key cannot break it.
func TestControlRoomReadoutSurvivesTheRealServiceShape(t *testing.T) {
	model := New(testOptions(t))
	_ = model.Init()
	model.screen = ScreenControlRoom

	_, _ = model.Update(controlRoomLoadedMsg{summary: map[string]any{
		"breaker_open_count": float64(1),
		"circuit_breakers": map[string]any{
			"qwen3.5:9b":    map[string]any{"state": "OPEN", "failure_count": float64(3)},
			"gemma4:latest": map[string]any{"state": "CLOSED"},
		},
		"recent_runs": []any{
			map[string]any{"query": "should AI be regulated?", "status": "complete"},
		},
		"domain_win_rates": []any{
			map[string]any{"domain": "policy", "win_rate": 0.75, "feedback_count": float64(4)},
		},
		"a_section_this_ui_does_not_know": map[string]any{"nested": true},
	}})

	view := model.View().Content
	for _, want := range []string{"Control room", "open circuit breakers", "OPEN", "recent runs", "policy"} {
		if !strings.Contains(view, want) {
			t.Errorf("control room is missing %q:\n%s", want, view)
		}
	}
}

func TestControlRoomEmptySummarySaysSo(t *testing.T) {
	model := New(testOptions(t))
	_ = model.Init()
	model.screen = ScreenControlRoom

	_, _ = model.Update(controlRoomLoadedMsg{summary: map[string]any{}})
	if view := model.View().Content; !strings.Contains(view, "no observability data") {
		t.Errorf("empty control room rendered %q", view)
	}
}

func TestUnreachableServiceShowsAnActionableError(t *testing.T) {
	model := New(testOptions(t))
	_ = model.Init()

	_, _ = model.Update(modelsLoadedMsg{err: api.ErrUnreachable})

	if model.failure == nil {
		t.Fatal("no failure was recorded for an unreachable service")
	}
	if !strings.Contains(model.failure.Error(), "start the service") {
		t.Errorf("failure = %q, want it to tell the user what to do", model.failure)
	}
}

func TestFinishCouncilRejectsAModeWithNoModels(t *testing.T) {
	model := New(testOptions(t))
	_ = model.Init()
	model.screen = ScreenCouncil
	model.council = CouncilDefaults()
	model.council.Question = "regulate?"
	model.council.Mode = api.ModeMultiModel
	model.council.Models = nil
	model.form = completedForm(t)

	cmd := model.advance()

	if model.failure == nil {
		t.Fatal("a model-less swarm mode was accepted")
	}
	if !strings.Contains(model.failure.Error(), "at least one model") {
		t.Errorf("failure = %q, want it to name the missing models", model.failure)
	}
	if model.screen != ScreenCouncil {
		t.Errorf("screen = %v, want the user returned to the form", model.screen)
	}
	if cmd == nil {
		t.Error("the form was not rebuilt")
	}
}

// completedForm is a form sitting in the state a submit leaves it in, which is
// what advance() reacts to.
func completedForm(t *testing.T) *huh.Form {
	t.Helper()
	form := huh.NewForm(huh.NewGroup(huh.NewInput().Value(new(string))))
	form.State = huh.StateCompleted
	return form
}

func TestResultFormRoutesBackOrToAnotherRun(t *testing.T) {
	for _, tc := range []struct {
		again bool
		want  Screen
	}{
		{again: true, want: ScreenCouncil},
		{again: false, want: ScreenMenu},
	} {
		model := New(testOptions(t))
		_ = model.Init()
		model.screen = ScreenResult
		model.result = ResultAnswers{Again: tc.again}
		model.councilState = sampleCouncil()
		model.form = completedForm(t)

		model.advance()

		if model.screen != tc.want {
			t.Errorf("again=%v: screen = %v, want %v", tc.again, model.screen, tc.want)
		}
	}
}

func TestViewRendersHeaderOnEveryScreen(t *testing.T) {
	screens := []Screen{
		ScreenConnect, ScreenMenu, ScreenCouncil, ScreenRunning, ScreenResult,
		ScreenStandings, ScreenControlRoom, ScreenTriads, ScreenModels,
	}
	for _, screen := range screens {
		model := New(testOptions(t))
		_ = model.Init()
		model.screen = screen
		model.councilState = sampleCouncil()
		model.models = []api.ModelInfo{{Name: "qwen3.5:9b", Backend: "ollama"}}
		model.triads = []api.TriadInfo{{Name: "architecture"}}
		model.leaderboard = []api.LeaderboardRow{{Model: "qwen3.5:9b"}}
		model.summary = map[string]any{"breaker_open_count": float64(0)}

		view := model.View()
		if view.Content == "" {
			t.Errorf("screen %v rendered nothing", screen)
		}
		if !strings.Contains(view.Content, "AI Congress") {
			t.Errorf("screen %v is missing the identity header", screen)
		}
		if !view.AltScreen {
			t.Errorf("screen %v does not use the alternate screen", screen)
		}
	}
}

func TestStreamErrorIsNilOnAFreshModel(t *testing.T) {
	if err := New(testOptions(t)).StreamError(); err != nil {
		t.Errorf("StreamError() = %v on a fresh model, want nil", err)
	}
}

func TestCloseIsSafeWithNoStreamOrServer(t *testing.T) {
	model := New(testOptions(t))
	model.Close()
	model.Close() // must not panic
}

// TestEscClosesAnOpenForm is the regression guard for a bug class confirmed in
// this workspace by reading huh's source: charm.land/huh/v2's default keymap
// binds Quit to ctrl+c ONLY, and that binding is the sole path that sets
// StateAborted. Escape therefore never aborts a form, so a model that delegates
// esc to the form leaves it installed.
//
// The damage is not cosmetic. No key is handled while a form is open, so the
// orphaned form swallows every later keystroke: the user cannot cancel, and the
// screen behind it becomes unreachable.
func TestEscClosesAnOpenForm(t *testing.T) {
	model := New(testOptions(t))
	_ = model.Init()

	// The council form is the representative mid-flow case.
	model.models = []api.ModelInfo{{Name: "mistral:7b"}}
	model.triads = []api.TriadInfo{{Name: "architecture", Description: "built to disagree"}}
	_ = model.toCouncil()

	if model.screen != ScreenCouncil || model.form == nil {
		t.Fatalf("setup: screen=%v form-nil=%v, want the council form open", model.screen, model.form == nil)
	}
	councilForm := model.form

	_, _ = model.Update(tea.KeyPressMsg{Code: tea.KeyEscape})

	// The council form must be replaced, not merely hidden: a still-installed
	// form keeps consuming input.
	if model.form == councilForm {
		t.Fatal("esc left the council form installed; it would keep swallowing keystrokes")
	}
	if model.screen != ScreenMenu {
		t.Errorf("screen = %v, want the menu, the hub a cancelled form returns to", model.screen)
	}
	if !strings.Contains(model.notice, "Cancelled") {
		t.Errorf("notice = %q, want a cancellation message", model.notice)
	}

	// Input must work again: a live form consumes a key rather than the
	// keystroke being dropped on the floor.
	if model.form == nil {
		t.Fatal("no form was installed, so the app can no longer take input")
	}
	if model.form.State != huh.StateNormal {
		t.Fatalf("menu form state = %v, want a normal form", model.form.State)
	}
	if _, cmd := model.Update(tea.KeyPressMsg{Code: tea.KeyEnter}); cmd == nil {
		t.Error("a keystroke after the cancel produced no command; input is still being swallowed")
	}
}

// TestEscOnTheConnectFormIsDeliberatelyInert pins the one screen excluded from
// esc-cancelling. The connect screen is the entry point, and the menu it would
// cancel to cannot do anything until a connection is configured, so esc there
// must leave the form alone rather than strand the user on a useless menu.
func TestEscOnTheConnectFormIsDeliberatelyInert(t *testing.T) {
	model := New(testOptions(t))
	_ = model.Init()
	if model.screen != ScreenConnect || model.form == nil {
		t.Fatalf("setup: screen=%v form-nil=%v", model.screen, model.form == nil)
	}
	connectForm := model.form

	_, _ = model.Update(tea.KeyPressMsg{Code: tea.KeyEscape})

	if model.screen != ScreenConnect {
		t.Errorf("screen = %v, want to stay on the connect form", model.screen)
	}
	if model.form != connectForm {
		t.Error("the connect form was replaced; esc must not cancel the entry point")
	}
}
