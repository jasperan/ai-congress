package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestValidateBaseURL(t *testing.T) {
	valid := []string{
		"http://127.0.0.1:8000",
		"https://congress.example:9443",
		"http://localhost:8100",
	}
	for _, raw := range valid {
		if err := ValidateBaseURL(raw); err != nil {
			t.Errorf("ValidateBaseURL(%q) = %v, want nil", raw, err)
		}
	}

	invalid := map[string]string{
		"":                     "a service URL is required",
		"   ":                  "a service URL is required",
		"127.0.0.1:8000":       "use an http:// or https:// URL",
		"ftp://127.0.0.1:8000": "use an http:// or https:// URL",
		"http://":              "the URL needs a host",
	}
	for raw, want := range invalid {
		err := ValidateBaseURL(raw)
		if err == nil {
			t.Errorf("ValidateBaseURL(%q) = nil, want %q", raw, want)
			continue
		}
		if !strings.Contains(err.Error(), want) {
			t.Errorf("ValidateBaseURL(%q) = %q, want it to contain %q", raw, err, want)
		}
	}
}

func TestWSURLRewritesScheme(t *testing.T) {
	cases := map[string]string{
		"http://127.0.0.1:8000":  "ws://127.0.0.1:8000/ws/chat",
		"https://congress:9443/": "wss://congress:9443/ws/chat",
		"http://host:8000///":    "ws://host:8000/ws/chat",
	}
	for base, want := range cases {
		client := NewClient(base, "")
		if got := client.WSURL(ChatPath); got != want {
			t.Errorf("NewClient(%q).WSURL() = %q, want %q", base, got, want)
		}
	}
}

func TestCouncilRequestValidate(t *testing.T) {
	if err := (CouncilRequest{Prompt: "why", Models: []string{"m"}, Mode: ModeMultiModel}).Validate(); err != nil {
		t.Errorf("a complete multi_model request was rejected: %v", err)
	}
	if err := (CouncilRequest{Prompt: "  ", Models: []string{"m"}}).Validate(); err == nil {
		t.Error("an empty prompt was accepted")
	}
	if err := (CouncilRequest{Prompt: "why", Mode: ModeMultiModel}).Validate(); err == nil {
		t.Error("multi_model with no models was accepted")
	}
	if err := (CouncilRequest{Prompt: "why", Mode: ModeDeliberation}).Validate(); err == nil {
		t.Error("deliberation with neither a triad nor models was accepted")
	}
	if err := (CouncilRequest{Prompt: "why", Mode: ModeDeliberation, Triad: "architecture"}).Validate(); err != nil {
		t.Errorf("deliberation with a triad was rejected: %v", err)
	}
	// personality mode is deliberately not supported by this front-end: it
	// needs a personalities list there is no picker for. The error must say so
	// rather than let the service answer with a 400.
	err := (CouncilRequest{Prompt: "why", Mode: ModePersonality}).Validate()
	if err == nil {
		t.Error("personality mode was accepted, but this front-end cannot supply personalities")
	} else if !strings.Contains(err.Error(), "personality") {
		t.Errorf("personality error = %q, want it to name the mode", err)
	}
}

func TestSelectableModesExcludePersonality(t *testing.T) {
	for _, mode := range SelectableModes {
		if mode == ModePersonality {
			t.Error("personality mode is selectable, but this front-end sends no personalities")
		}
		if ModeDescription(mode) == "" {
			t.Errorf("ModeDescription(%q) is empty", mode)
		}
	}
	if len(SelectableModes) != len(Modes)-1 {
		t.Errorf("SelectableModes has %d entries, want %d", len(SelectableModes), len(Modes)-1)
	}
}

func TestClientSendsAPIKeyOnlyWhenSet(t *testing.T) {
	var seen []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = append(seen, r.Header.Get("X-API-Key"))
		_, _ = w.Write([]byte(`{"status":"healthy"}`))
	}))
	defer server.Close()

	withKey := NewClient(server.URL, "secret-key")
	if !withKey.HasAPIKey() {
		t.Fatal("HasAPIKey() = false after a key was supplied")
	}
	if _, err := withKey.Health(context.Background()); err != nil {
		t.Fatalf("Health() = %v", err)
	}

	withoutKey := NewClient(server.URL, "   ")
	if withoutKey.HasAPIKey() {
		t.Error("HasAPIKey() = true for a blank key")
	}
	if _, err := withoutKey.Health(context.Background()); err != nil {
		t.Fatalf("Health() = %v", err)
	}

	if len(seen) != 2 {
		t.Fatalf("server saw %d requests, want 2", len(seen))
	}
	if seen[0] != "secret-key" {
		t.Errorf("first request sent X-API-Key %q, want %q", seen[0], "secret-key")
	}
	if seen[1] != "" {
		t.Errorf("second request sent X-API-Key %q, want it absent", seen[1])
	}
}

func TestModelsDecodesTheServiceShape(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models" {
			t.Errorf("path = %q, want /api/models", r.URL.Path)
		}
		_, _ = w.Write([]byte(`[
			{"name":"qwen3.5:9b","size":6100000000,"weight":0.87,"backend":"ollama"},
			{"name":"deepseek-v4-flash","size":0,"weight":1.0,"backend":"pi"}
		]`))
	}))
	defer server.Close()

	models, err := NewClient(server.URL, "").Models(context.Background())
	if err != nil {
		t.Fatalf("Models() = %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("got %d models, want 2", len(models))
	}
	if models[0].Name != "qwen3.5:9b" || models[0].Weight != 0.87 || models[0].Backend != "ollama" {
		t.Errorf("unexpected first model: %+v", models[0])
	}
	if models[1].Backend != "pi" {
		t.Errorf("unexpected second model: %+v", models[1])
	}
}

func TestLeaderboardAndTriadsDecodeDespiteEmptyCollections(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/observability/leaderboard":
			_, _ = w.Write([]byte(`{"leaderboard":[]}`))
		case "/api/triads":
			_, _ = w.Write([]byte(`{"triads":[{"name":"architecture","description":"tension"}]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	client := NewClient(server.URL, "")

	rows, err := client.Leaderboard(context.Background())
	if err != nil {
		t.Fatalf("Leaderboard() = %v", err)
	}
	if len(rows) != 0 {
		t.Errorf("got %d leaderboard rows, want 0", len(rows))
	}

	triads, err := client.Triads(context.Background())
	if err != nil {
		t.Fatalf("Triads() = %v", err)
	}
	if len(triads) != 1 || triads[0].Name != "architecture" {
		t.Errorf("unexpected triads: %+v", triads)
	}
}

func TestHTTPErrorCarriesFastAPIDetail(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnprocessableEntity)
		_, _ = w.Write([]byte(`{"detail":[{"msg":"field required","loc":["body","prompt"]}]}`))
	}))
	defer server.Close()

	_, err := NewClient(server.URL, "").Chat(context.Background(), CouncilRequest{
		Prompt: "why", Models: []string{"m"}, Mode: ModeMultiModel,
	})
	if err == nil {
		t.Fatal("Chat() = nil, want an HTTP error")
	}
	if !strings.Contains(err.Error(), "422") {
		t.Errorf("error %q does not mention the status code", err)
	}
	if !strings.Contains(err.Error(), "field required") {
		t.Errorf("error %q does not carry FastAPI's detail", err)
	}
}

func TestUnreachableServiceIsDistinguishable(t *testing.T) {
	// Port 0 on a closed listener: dialling it fails deterministically.
	client := NewClient("http://127.0.0.1:1", "")
	_, err := client.Health(context.Background())
	if err == nil {
		t.Fatal("Health() = nil against a dead service")
	}
	if !strings.Contains(err.Error(), ErrUnreachable.Error()) {
		t.Errorf("error %q is not wrapped around ErrUnreachable, so the UI cannot explain it", err)
	}
}

func TestChatPostsTheSchemaShape(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		_, _ = w.Write([]byte(`{"final_answer":"yes","confidence":0.8,
			"vote_breakdown":{
			  "alpha":{"original":"alpha","original_weight":0.5,"weight":2.0,"votes":[0.5,1.5],"models":["a","b"]}
			},
			"responses":[{"model":"a","response":"yes","success":true}]}`))
	}))
	defer server.Close()

	result, err := NewClient(server.URL, "").Chat(context.Background(), CouncilRequest{
		Prompt:           "regulate?",
		Models:           []string{"a", "b"},
		Mode:             ModeMultiModel,
		VotingMode:       VotingClassic,
		InferenceBackend: BackendOllama,
		Temperature:      0.7,
	})
	if err != nil {
		t.Fatalf("Chat() = %v", err)
	}
	if body["prompt"] != "regulate?" {
		t.Errorf("prompt = %v, want %q", body["prompt"], "regulate?")
	}
	if body["mode"] != ModeMultiModel {
		t.Errorf("mode = %v, want %q", body["mode"], ModeMultiModel)
	}
	if body["voting_mode"] != VotingClassic {
		t.Errorf("voting_mode = %v, want %q", body["voting_mode"], VotingClassic)
	}
	if result == nil || result.FinalAnswer != "yes" || result.Confidence != 0.8 {
		t.Fatalf("unexpected result: %+v", result)
	}
	group, ok := result.VoteBreakdown["alpha"]
	if !ok {
		t.Fatalf("vote_breakdown = %v, want the alpha group", result.VoteBreakdown)
	}
	if group.Weight != 2.0 {
		t.Errorf("group weight = %v, want 2.0 (the pooled weight, not original_weight)", group.Weight)
	}
	if len(group.Models) != 2 || group.Models[1] != "b" {
		t.Errorf("group models = %v, want both contributors", group.Models)
	}
	if len(group.Votes) != 2 {
		t.Errorf("group votes = %v, want both contributing weights", group.Votes)
	}
}

func TestDeliberatePostsQuestionNotPrompt(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		_, _ = w.Write([]byte(`{"mode":"deliberation","final_answer":"a",
			"rounds":[{"name":"opening","outputs":[{"agent":"Pragmatist","success":true}]}]}`))
	}))
	defer server.Close()

	result, err := NewClient(server.URL, "").Deliberate(context.Background(), DeliberationRequest{
		Question: "regulate?",
		Triad:    "architecture",
	})
	if err != nil {
		t.Fatalf("Deliberate() = %v", err)
	}
	// schemas.py:DeliberationRequest names the field "question", unlike ChatRequest.
	if _, ok := body["prompt"]; ok {
		t.Error("deliberation body used prompt, but the schema expects question")
	}
	if body["question"] != "regulate?" {
		t.Errorf("question = %v, want %q", body["question"], "regulate?")
	}
	if result.RoundCount() != 1 {
		t.Errorf("RoundCount() = %d, want 1", result.RoundCount())
	}
}

func TestModesMatchTheServiceDispatch(t *testing.T) {
	// main.py:websocket_chat dispatches on exactly these strings.
	want := []string{"multi_model", "multi_request", "hybrid", "personality", "deliberation"}
	if len(Modes) != len(want) {
		t.Fatalf("Modes = %v, want %v", Modes, want)
	}
	for i, mode := range want {
		if Modes[i] != mode {
			t.Errorf("Modes[%d] = %q, want %q", i, Modes[i], mode)
		}
		if ModeDescription(mode) == "" {
			t.Errorf("ModeDescription(%q) is empty", mode)
		}
	}
}

// TestVoteBreakdownDecodesTheRealServiceShape is a regression guard.
//
// The first implementation modelled vote_breakdown as map[string]float64, which
// is what swarm_orchestrator.py's Python-side formatting implies (`{value:.1f}`).
// A live /ws/chat run proved otherwise: the key is the normalised RESPONSE TEXT
// and the value is a five-field record. The payload below is copied from that
// run, and `--ask` failed on it before this type existed.
func TestVoteBreakdownDecodesTheRealServiceShape(t *testing.T) {
	payload := `{
	  "mode": "multi_model",
	  "final_answer": "Water is indeed wet.",
	  "confidence": 1.0,
	  "vote_breakdown": {
	    "water is indeed wet.": {
	      "original": "Water is indeed wet.",
	      "original_weight": 0.5,
	      "weight": 0.5,
	      "votes": [0.5],
	      "models": ["smollm2:135m"]
	    }
	  },
	  "responses": [{"model": "smollm2:135m", "response": "Water is indeed wet.", "success": true}]
	}`

	var result CouncilResult
	if err := json.Unmarshal([]byte(payload), &result); err != nil {
		t.Fatalf("the real service payload does not decode: %v", err)
	}

	group, ok := result.VoteBreakdown["water is indeed wet."]
	if !ok {
		t.Fatalf("vote_breakdown = %v, want the group keyed by the normalised answer", result.VoteBreakdown)
	}
	if group.Weight != 0.5 {
		t.Errorf("Weight = %v, want 0.5", group.Weight)
	}
	if group.OriginalWeight != 0.5 {
		t.Errorf("OriginalWeight = %v, want 0.5", group.OriginalWeight)
	}
	if len(group.Votes) != 1 || group.Votes[0] != 0.5 {
		t.Errorf("Votes = %v, want [0.5]", group.Votes)
	}
	if len(group.Models) != 1 || group.Models[0] != "smollm2:135m" {
		t.Errorf("Models = %v", group.Models)
	}
	if result.FinalAnswer != "Water is indeed wet." {
		t.Errorf("FinalAnswer = %q", result.FinalAnswer)
	}
	if result.RoundCount() != 0 {
		t.Errorf("RoundCount() = %d, want 0 for a non-deliberation run", result.RoundCount())
	}
}

// TestVoteBreakdownToleratesAPooledGroup covers the semantic-grouping case,
// where several models pool their weight into one entry.
func TestVoteBreakdownToleratesAPooledGroup(t *testing.T) {
	payload := `{
	  "vote_breakdown": {
	    "yes": {
	      "original": "Yes, with caveats.",
	      "original_weight": 0.9,
	      "weight": 1.6,
	      "votes": [0.9, 0.7],
	      "models": ["qwen3.5:9b", "gemma4:latest"]
	    }
	  }
	}`

	var result CouncilResult
	if err := json.Unmarshal([]byte(payload), &result); err != nil {
		t.Fatalf("a pooled group does not decode: %v", err)
	}
	group := result.VoteBreakdown["yes"]
	if group.Weight != 1.6 || len(group.Models) != 2 || len(group.Votes) != 2 {
		t.Errorf("pooled group = %+v, want both weights and both models", group)
	}
}
