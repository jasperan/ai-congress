package api

// --- Models ---

// ModelInfo represents a model from GET /api/models.
type ModelInfo struct {
	Name    string  `json:"name"`
	Size    int64   `json:"size"`
	Weight  float64 `json:"weight"`
	Backend string  `json:"backend"`
}

// --- Chat ---

// ChatRequest is the payload for POST /api/chat.
type ChatRequest struct {
	Prompt           string        `json:"prompt"`
	Models           []string      `json:"models"`
	Mode             string        `json:"mode"`
	Temperature      float64       `json:"temperature"`
	SystemPrompt     string        `json:"system_prompt,omitempty"`
	Personalities    []Personality `json:"personalities,omitempty"`
	UseRAG           bool          `json:"use_rag,omitempty"`
	SearchWeb        bool          `json:"search_web,omitempty"`
	VotingMode       string        `json:"voting_mode"`
	InferenceBackend string        `json:"inference_backend,omitempty"`
}

// Personality is a name + system_prompt pair.
type Personality struct {
	Name         string `json:"name"`
	SystemPrompt string `json:"system_prompt"`
}

// ChatResponse is the response from POST /api/chat.
type ChatResponse struct {
	FinalAnswer    string                   `json:"final_answer"`
	Confidence     float64                  `json:"confidence"`
	VoteBreakdown  map[string]interface{}   `json:"vote_breakdown,omitempty"`
	SemanticVote   map[string]interface{}   `json:"semantic_vote,omitempty"`
	Responses      []ModelResponse          `json:"responses"`
	ModelsUsed     []string                 `json:"models_used,omitempty"`
	ContextSources []map[string]interface{} `json:"context_sources,omitempty"`
}

// ModelResponse is a single model's response within a chat result.
type ModelResponse struct {
	Model      string  `json:"model"`
	EntityName string  `json:"entity_name,omitempty"`
	Response   string  `json:"response"`
	Success    bool    `json:"success"`
	Confidence float64 `json:"confidence,omitempty"`
}

// --- Enhanced Orchestrator ---

// EnhancedChatRequest is the payload for POST /api/chat/enhanced.
type EnhancedChatRequest struct {
	Prompt              string   `json:"prompt"`
	Models              []string `json:"models"`
	Temperature         float64  `json:"temperature"`
	EnableDecomposition bool     `json:"enable_decomposition"`
	EnableDebate        bool     `json:"enable_debate"`
	UseRAG              bool     `json:"use_rag,omitempty"`
	SearchWeb           bool     `json:"search_web,omitempty"`
}

// EnhancedRunResponse is returned by GET /api/enhanced/runs/{run_id}.
type EnhancedRunResponse struct {
	RunID      string                 `json:"run_id"`
	Status     string                 `json:"status"`
	Prompt     string                 `json:"prompt"`
	Result     map[string]interface{} `json:"result,omitempty"`
	Confidence float64                `json:"confidence"`
	CreatedAt  string                 `json:"created_at,omitempty"`
}

// StatsResponse is returned by GET /api/enhanced/stats.
type StatsResponse struct {
	TotalRuns     int     `json:"total_runs"`
	AvgConfidence float64 `json:"avg_confidence"`
	AvgLatencyMs  int     `json:"avg_latency_ms"`
}

// --- Personalities ---

// PersonalityListResponse from GET /api/personality-lists.
// (Returns []string, no struct needed.)

// --- WebSocket ---

// WSMessage represents a WebSocket event from /ws/chat.
type WSMessage struct {
	Type    string                 `json:"type"`
	Data    map[string]interface{} `json:"data,omitempty"`
	Message string                 `json:"message,omitempty"`
	// Fields from model_response events
	Model   string `json:"model,omitempty"`
	Content string `json:"content,omitempty"`
	// Fields from final_answer events
	Confidence    float64                `json:"confidence,omitempty"`
	VoteBreakdown map[string]interface{} `json:"vote_breakdown,omitempty"`
	SemanticVote  map[string]interface{} `json:"semantic_vote,omitempty"`
}

// WSChatRequest is the JSON payload sent over the WebSocket to initiate a chat.
type WSChatRequest struct {
	Prompt           string        `json:"prompt"`
	Models           []string      `json:"models"`
	Mode             string        `json:"mode"`
	Stream           bool          `json:"stream"`
	Temperature      float64       `json:"temperature,omitempty"`
	Personalities    []Personality `json:"personalities,omitempty"`
	VotingMode       string        `json:"voting_mode,omitempty"`
	InferenceBackend string        `json:"inference_backend,omitempty"`
}
