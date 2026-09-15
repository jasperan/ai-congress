package tui

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/jasperan/ai-congress/gotui/internal/api"
)

func TestParseActionFlagsPrefersTheMostSpecificFlag(t *testing.T) {
	cases := []struct {
		name     string
		question string
		health   bool
		models   bool
		triads   bool
		stand    bool
		control  bool
		want     string
		wantAny  bool
	}{
		{name: "nothing", wantAny: false},
		{name: "ask wins over every read", question: "why", health: true, models: true, stand: true,
			want: ActionAsk, wantAny: true},
		{name: "health outranks the lists", health: true, models: true, triads: true,
			want: ActionHealth, wantAny: true},
		{name: "models outrank triads", models: true, triads: true,
			want: ActionListModels, wantAny: true},
		{name: "triads outrank standings", triads: true, stand: true,
			want: ActionListTriads, wantAny: true},
		{name: "standings outrank the control room", stand: true, control: true,
			want: ActionStandings, wantAny: true},
		{name: "control room alone", control: true, want: ActionControlRoom, wantAny: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := ParseActionFlags(tc.question, tc.health, tc.models, tc.triads, tc.stand, tc.control)
			if ok != tc.wantAny {
				t.Fatalf("ok = %v, want %v", ok, tc.wantAny)
			}
			if got != tc.want {
				t.Fatalf("action = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestParseTemperatureEnforcesTheServiceRange(t *testing.T) {
	valid := map[string]float64{"0": 0, "0.7": 0.7, "2": 2, " 1.25 ": 1.25}
	for raw, want := range valid {
		if got := ParseTemperature(raw); got != want {
			t.Errorf("ParseTemperature(%q) = %v, want %v", raw, got, want)
		}
	}
	if err := ValidateTemperature("2.5"); err == nil {
		t.Error("ValidateTemperature(2.5) = nil, want an error")
	}
	if err := ValidateTemperature("-0.1"); err == nil {
		t.Error("ValidateTemperature(-0.1) = nil, want an error")
	}
	// Out-of-range values fall back to the service default rather than being
	// silently sent.
	for _, raw := range []string{"", "nope", "9"} {
		if got := ParseTemperature(raw); got != 0.7 {
			t.Errorf("ParseTemperature(%q) = %v, want the 0.7 default", raw, got)
		}
	}
}

// TestCouncilAnswersToRequestIsModeAware pins the mapping that decides whether
// a run uses the chosen models or the triad's archetypes.
func TestCouncilAnswersToRequestIsModeAware(t *testing.T) {
	base := CouncilAnswers{
		Question:    "regulate?",
		VotingMode:  api.VotingClassic,
		Backend:     api.BackendOllama,
		Temperature: "0.7",
		Models:      []string{"qwen3.5:9b", "gemma4:latest"},
	}

	t.Run("swarm mode passes the models and the context toggles", func(t *testing.T) {
		answers := base
		answers.Mode = api.ModeMultiModel
		answers.UseRAG = true
		answers.SearchWeb = true
		// A triad must be ignored outside deliberation.
		answers.Triad = "architecture"

		request := answers.ToRequest()
		if len(request.Models) != 2 {
			t.Errorf("Models = %v, want both", request.Models)
		}
		if request.Triad != "" {
			t.Errorf("Triad = %q, want it ignored outside deliberation", request.Triad)
		}
		if !request.UseRAG || !request.SearchWeb {
			t.Error("the context toggles were dropped")
		}
		if err := request.Validate(); err != nil {
			t.Errorf("Validate() = %v", err)
		}
	})

	t.Run("deliberation with a triad sends the triad and no models", func(t *testing.T) {
		answers := base
		answers.Mode = api.ModeDeliberation
		answers.Triad = "architecture"
		answers.Evidence = true

		request := answers.ToRequest()
		if request.Triad != "architecture" {
			t.Errorf("Triad = %q", request.Triad)
		}
		if len(request.Models) != 0 {
			t.Errorf("Models = %v, want none: a model list would override the triad's archetypes", request.Models)
		}
		if request.Evidence == nil || !*request.Evidence {
			t.Error("Evidence was not requested")
		}
		if err := request.Validate(); err != nil {
			t.Errorf("Validate() = %v", err)
		}
	})

	t.Run("deliberation without a triad falls back to the models", func(t *testing.T) {
		answers := base
		answers.Mode = api.ModeDeliberation

		request := answers.ToRequest()
		if len(request.Models) != 2 {
			t.Errorf("Models = %v, want both", request.Models)
		}
		if request.Triad != "" {
			t.Errorf("Triad = %q, want empty", request.Triad)
		}
		if err := request.Validate(); err != nil {
			t.Errorf("Validate() = %v", err)
		}
	})
}

// fakeService serves the read endpoints from one handler.
func fakeService(t *testing.T) *api.Client {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			_, _ = w.Write([]byte(`{"status":"healthy"}`))
		case "/api/models":
			_, _ = w.Write([]byte(`[{"name":"qwen3.5:9b","weight":0.87,"backend":"ollama"}]`))
		case "/api/triads":
			_, _ = w.Write([]byte(`{"triads":[{"name":"architecture","description":"tension"}]}`))
		case "/api/observability/leaderboard":
			_, _ = w.Write([]byte(`{"leaderboard":[{"model":"qwen3.5:9b","weight":0.9,"win_rate":0.5,"participations":4}]}`))
		case "/api/observability/summary":
			_, _ = w.Write([]byte(`{"breaker_open_count":1,"circuit_breakers":{"qwen3.5:9b":{"state":"OPEN"}},
				"recent_runs":[{"query":"q","status":"complete"}]}`))
		case "/api/chat":
			_, _ = w.Write([]byte(`{"mode":"multi_model","final_answer":"Yes, carefully.","confidence":0.82,
				"vote_breakdown":{
				  "yes":{"original":"Yes.","original_weight":0.5,"weight":2.4,"votes":[0.5],"models":["qwen3.5:9b"]},
				  "no":{"original":"No.","original_weight":0.5,"weight":1.1,"votes":[0.5],"models":["gemma4:latest"]}
				},
				"responses":[{"model":"qwen3.5:9b","response":"Yes.","success":true},
				             {"model":"gemma4:latest","response":"No.","success":false}]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	return api.NewClient(server.URL, "")
}

func TestRunActionReadsRenderPlainText(t *testing.T) {
	client := fakeService(t)
	ctx := context.Background()

	cases := []struct {
		action string
		want   []string
	}{
		{ActionHealth, []string{"healthy", "service"}},
		{ActionListModels, []string{"qwen3.5:9b", "ollama", "0.870"}},
		{ActionListTriads, []string{"architecture", "tension"}},
		{ActionStandings, []string{"qwen3.5:9b", "0.900", "50%"}},
		{ActionControlRoom, []string{"breaker_open_count", "circuit_breakers", "recent_runs"}},
	}

	for _, tc := range cases {
		t.Run(tc.action, func(t *testing.T) {
			var out bytes.Buffer
			if err := RunAction(ctx, client, ActionRequest{Action: tc.action}, &out); err != nil {
				t.Fatalf("RunAction(%q) = %v", tc.action, err)
			}
			for _, want := range tc.want {
				if !strings.Contains(out.String(), want) {
					t.Errorf("output is missing %q:\n%s", want, out.String())
				}
			}
		})
	}
}

func TestRunActionJSONIsMachineReadable(t *testing.T) {
	client := fakeService(t)
	ctx := context.Background()

	for _, action := range []string{ActionHealth, ActionListModels, ActionListTriads, ActionStandings} {
		t.Run(action, func(t *testing.T) {
			var out bytes.Buffer
			if err := RunAction(ctx, client, ActionRequest{Action: action, JSON: true}, &out); err != nil {
				t.Fatalf("RunAction(%q) = %v", action, err)
			}
			var decoded any
			if err := json.Unmarshal(out.Bytes(), &decoded); err != nil {
				t.Fatalf("output is not valid JSON: %v\n%s", err, out.String())
			}
			if !strings.HasSuffix(out.String(), "\n") {
				t.Error("JSON output is not newline terminated, which breaks piping")
			}
		})
	}
}

// TestRunActionAskPrintsTheVerdict is the scripted equivalent of the result
// screen: answer, confidence, tally and per-model responses.
func TestRunActionAskPrintsTheVerdict(t *testing.T) {
	client := fakeService(t)
	var out bytes.Buffer

	err := RunAction(context.Background(), client, ActionRequest{
		Action:   ActionAsk,
		Question: "regulate?",
		Mode:     api.ModeMultiModel,
		Models:   []string{"qwen3.5:9b", "gemma4:latest"},
	}, &out)
	if err != nil {
		t.Fatalf("RunAction(ask) = %v", err)
	}

	rendered := out.String()
	for _, want := range []string{
		"Yes, carefully.", "confidence", "82%", "consensus",
		"winner", "qwen3.5:9b", "votes", "Yes.", "No.",
	} {
		if !strings.Contains(rendered, want) {
			t.Errorf("run output is missing %q:\n%s", want, rendered)
		}
	}
	// A failed member is marked, not hidden.
	if !strings.Contains(rendered, "! gemma4:latest") {
		t.Errorf("the failed response is not marked:\n%s", rendered)
	}
}

func TestRunActionAskRejectsAnIncompleteRequestBeforeCalling(t *testing.T) {
	// A client pointed at a dead address proves no call is attempted.
	client := api.NewClient("http://127.0.0.1:1", "")

	var out bytes.Buffer
	err := RunAction(context.Background(), client, ActionRequest{
		Action:   ActionAsk,
		Question: "regulate?",
		Mode:     api.ModeMultiModel,
	}, &out)
	if err == nil {
		t.Fatal("an ask with no models was accepted")
	}
	if !strings.Contains(err.Error(), "at least one model") {
		t.Errorf("error = %q, want it to name the missing models", err)
	}
}

func TestRunActionUnknownActionIsAnError(t *testing.T) {
	var out bytes.Buffer
	err := RunAction(context.Background(), fakeService(t), ActionRequest{Action: "wat"}, &out)
	if err == nil {
		t.Fatal("an unknown action was accepted")
	}
	if !strings.Contains(err.Error(), "wat") {
		t.Errorf("error = %q, want it to name the action", err)
	}
}

func TestRunActionPropagatesAServiceFailure(t *testing.T) {
	client := api.NewClient("http://127.0.0.1:1", "")
	err := RunAction(context.Background(), client, ActionRequest{Action: ActionHealth}, &bytes.Buffer{})
	if err == nil {
		t.Fatal("RunAction reported success against a dead service")
	}
	if !errors.Is(err, api.ErrUnreachable) {
		t.Errorf("error = %v, want it to wrap ErrUnreachable", err)
	}
}

func TestFirstNonEmptyPrefersTheFirstValue(t *testing.T) {
	if got := firstNonEmpty("", "  ", "b", "c"); got != "b" {
		t.Errorf("firstNonEmpty() = %q, want b", got)
	}
	if got := firstNonEmpty("", " "); got != "" {
		t.Errorf("firstNonEmpty(blanks) = %q, want empty", got)
	}
}

func TestIndentPrefixesOnlyNonEmptyLines(t *testing.T) {
	got := indent("a\n\nb", "  ")
	if got != "  a\n\n  b" {
		t.Errorf("indent() = %q", got)
	}
	if got := indent("", "  "); got != "  " {
		t.Errorf("indent(empty) = %q", got)
	}
}
