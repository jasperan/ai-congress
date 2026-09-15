// Package api is a peer client for the existing AI Congress FastAPI service
// (src/ai_congress/api/). It mirrors the endpoint set the Rust TUI already
// consumes (tui-rs/src/api/) so that a Go user, a Rust user and a Python user
// see the same congress.
//
// Nothing here reimplements orchestration, voting or deliberation: every
// method is a thin HTTP or WebSocket call against the service the Python CLI
// and the Svelte frontend already use.
package api

// DefaultBaseURL is where the AI Congress service listens by default
// (run_server.py: AI_CONGRESS_PORT defaults to 8000).
const DefaultBaseURL = "http://127.0.0.1:8000"

// HealthResponse is GET /health.
type HealthResponse struct {
	Status string `json:"status"`
}

// ModelInfo is one element of GET /api/models (schemas.py:ModelInfo).
type ModelInfo struct {
	Name    string  `json:"name"`
	Size    int64   `json:"size"`
	Weight  float64 `json:"weight"`
	Backend string  `json:"backend"`
}

// TriadInfo is one entry of GET /api/triads.
type TriadInfo struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

// TriadsResponse is GET /api/triads.
type TriadsResponse struct {
	Triads []TriadInfo `json:"triads"`
}

// CouncilRequest is the body of both POST /api/chat and the /ws/chat envelope.
//
// The field set and defaults mirror schemas.py:ChatRequest; the WebSocket
// handler reads the same names out of the JSON frame (main.py:websocket_chat).
type CouncilRequest struct {
	Prompt           string   `json:"prompt"`
	Models           []string `json:"models"`
	Mode             string   `json:"mode"`
	Temperature      float64  `json:"temperature"`
	VotingMode       string   `json:"voting_mode"`
	InferenceBackend string   `json:"inference_backend"`
	Triad            string   `json:"triad,omitempty"`
	UseRAG           bool     `json:"use_rag"`
	SearchWeb        bool     `json:"search_web"`
	Evidence         *bool    `json:"evidence,omitempty"`
}

// DeliberationRequest is the body of POST /api/deliberation
// (schemas.py:DeliberationRequest).
type DeliberationRequest struct {
	Question         string   `json:"question"`
	Triad            string   `json:"triad,omitempty"`
	Models           []string `json:"models,omitempty"`
	Temperature      float64  `json:"temperature"`
	InferenceBackend string   `json:"inference_backend"`
	Evidence         *bool    `json:"evidence,omitempty"`
}

// DebateEntry is one council member's output inside a deliberation round.
// deliberation.py:RoundResult.outputs is a list of exactly these keys.
type DebateEntry struct {
	Agent    string `json:"agent"`
	Role     string `json:"role"`
	Model    string `json:"model"`
	Response string `json:"response"`
	Success  bool   `json:"success"`
}

// Round is one named deliberation round (deliberation.py:RoundResult).
type Round struct {
	Name    string        `json:"name"`
	Outputs []DebateEntry `json:"outputs"`
}

// ModelResponse is one element of a chat result's "responses".
type ModelResponse struct {
	Model    string `json:"model"`
	Agent    string `json:"agent"`
	Response string `json:"response"`
	Success  bool   `json:"success"`
}

// Cluster is one semantic voting cluster (semantic_voting.py:Cluster).
type Cluster struct {
	ID        int      `json:"id"`
	Label     string   `json:"label"`
	Models    []string `json:"models"`
	KeyClaims []string `json:"key_claims"`
}

// VoteGroup is one entry of vote_breakdown.
//
// The service keys vote_breakdown by the NORMALISED response text (or the
// canonical key of a semantic group), not by model, and the value is a record
// rather than a number. The shape below is verified twice over: against the
// running service, and against voting_engine.py:weighted_majority_vote, which
// builds exactly these five fields.
type VoteGroup struct {
	// Original is the representative response text for the group: the
	// highest-weight member's text.
	Original string `json:"original"`
	// OriginalWeight is that representative's own weight.
	OriginalWeight float64 `json:"original_weight"`
	// Weight is the group's pooled weight, and is the number a tally sums.
	Weight float64 `json:"weight"`
	// Votes is each contributing weight, in the order the models answered.
	Votes []float64 `json:"votes"`
	// Models are the models that pooled into this group.
	Models []string `json:"models"`
}

// SemanticVote is the semantic voting result (semantic_voting.py:SemanticVoteResult).
type SemanticVote struct {
	Winner            string    `json:"winner"`
	WinningModel      string    `json:"winning_model"`
	Consensus         float64   `json:"consensus"`
	DebateTriggered   bool      `json:"debate_triggered"`
	DebateRounds      int       `json:"debate_rounds"`
	DissentingSummary string    `json:"dissenting_summary"`
	Clusters          []Cluster `json:"clusters"`
}

// CouncilResult is the union of the chat and deliberation result shapes.
//
// Both endpoints return a free-form dict rather than a pydantic model, so the
// fields below are the ones the service actually sets
// (swarm_orchestrator.py:multi_model_swarm and :deliberation_swarm). Absent
// keys decode to their zero value, which the UI treats as "not reported".
type CouncilResult struct {
	Mode             string               `json:"mode"`
	FinalAnswer      string               `json:"final_answer"`
	Confidence       float64              `json:"confidence"`
	VoteBreakdown    map[string]VoteGroup `json:"vote_breakdown"`
	SemanticVote     *SemanticVote        `json:"semantic_vote"`
	ModelsUsed       []string             `json:"models_used"`
	Responses        []ModelResponse      `json:"responses"`
	Rounds           []Round              `json:"rounds"`
	AgentsUsed       []string             `json:"agents_used"`
	Triad            string               `json:"triad"`
	DissentReport    map[string]any       `json:"dissent_report"`
	Verdict          map[string]any       `json:"verdict"`
	ContextSources   []map[string]any     `json:"context_sources"`
	Sources          []map[string]any     `json:"sources"`
	WebSearchResults []map[string]any     `json:"web_search_results"`
}

// RoundCount is the number of deliberation rounds the council reported, or 0
// when it did not run the deliberation protocol.
func (r *CouncilResult) RoundCount() int {
	if r == nil {
		return 0
	}
	return len(r.Rounds)
}

// LeaderboardRow is one line of GET /api/observability/leaderboard.
type LeaderboardRow struct {
	Model          string  `json:"model"`
	Weight         float64 `json:"weight"`
	WinRate        float64 `json:"win_rate"`
	Participations int64   `json:"participations"`
}

// LeaderboardResponse is GET /api/observability/leaderboard.
type LeaderboardResponse struct {
	Leaderboard []LeaderboardRow `json:"leaderboard"`
}

// BreakerState is one model's circuit-breaker state.
type BreakerState struct {
	State           string   `json:"state"`
	FailureCount    int64    `json:"failure_count"`
	LastFailureAgeS *float64 `json:"last_failure_age_s"`
}

// RunSummary is one entry of the control room's recent_runs.
type RunSummary struct {
	RunID           string   `json:"run_id"`
	Query           string   `json:"query"`
	Status          string   `json:"status"`
	DurationSeconds *float64 `json:"duration_seconds"`
	EventCount      int64    `json:"event_count"`
	FinalAnswer     string   `json:"final_answer"`
}

// DomainWinRate is one entry of domain_win_rates.
type DomainWinRate struct {
	Domain        string  `json:"domain"`
	FeedbackCount int64   `json:"feedback_count"`
	WinRate       float64 `json:"win_rate"`
}

// ObservabilitySummary is GET /api/observability/summary.
//
// Only the sections this UI renders are typed; the rest is decoded into
// Other so a new service-side section cannot break decoding.
type ObservabilitySummary struct {
	Leaderboard     []LeaderboardRow        `json:"leaderboard"`
	CircuitBreakers map[string]BreakerState `json:"circuit_breakers"`
	BreakerOpen     int64                   `json:"breaker_open_count"`
	RecentRuns      []RunSummary            `json:"recent_runs"`
	DomainWinRates  []DomainWinRate         `json:"domain_win_rates"`
	Other           map[string]any          `json:"-"`
}

// StreamEvent is one WebSocket frame from /ws/chat.
//
// main.py:websocket_chat emits {type, ...} frames; the types are shared,
// status_init, status_update, chunk, model_response, final_answer, end, error.
type StreamEvent struct {
	Type string `json:"type"`

	// start / error / final_answer
	Message string `json:"message"`
	Content string `json:"content"`

	// status_update / model_response / chunk
	Name   string `json:"name"`
	Model  string `json:"model"`
	Status string `json:"status"`

	// status_update carries the finished text on its "complete" frame, with the
	// full response in the same envelope as the status.
	Response string `json:"response"`

	// status_init
	Personalities []PersonalityStatus `json:"personalities"`

	// final_answer
	Confidence         float64              `json:"confidence"`
	SemanticConfidence float64              `json:"semantic_confidence"`
	VoteBreakdown      map[string]VoteGroup `json:"vote_breakdown"`
	SemanticVote       *SemanticVote        `json:"semantic_vote"`
	Mode               string               `json:"mode"`
	Verdict            map[string]any       `json:"verdict"`
	Data               *VerdictData         `json:"data"`
}

// PersonalityStatus is one entry of a status_init frame.
type PersonalityStatus struct {
	Name   string `json:"name"`
	Status string `json:"status"`
}

// VerdictData is the deliberation payload carried inside a final_answer frame.
type VerdictData struct {
	Rounds         []Round         `json:"rounds"`
	Restate        map[string]any  `json:"restate"`
	Steelman       []DebateEntry   `json:"steelman"`
	Dissent        map[string]any  `json:"dissent_report"`
	FinalPositions []ModelResponse `json:"final_positions"`
	AgentsUsed     []string        `json:"agents_used"`
	Metadata       map[string]any  `json:"metadata"`
}

// IsTerminal reports whether the frame ends the stream.
func (e StreamEvent) IsTerminal() bool {
	return e.Type == "end" || e.Type == "error"
}
