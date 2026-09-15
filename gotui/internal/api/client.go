package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// ErrUnreachable reports that the service could not be contacted at all. It is
// distinct from an HTTP error status so the UI can say "start the server"
// instead of showing a transport error.
var ErrUnreachable = errors.New("AI Congress API unreachable")

// Client is a client for the AI Congress FastAPI service.
//
// It is safe for concurrent use: the council stream and the standings poller
// run in separate goroutines.
type Client struct {
	baseURL string
	apiKey  string
	http    *http.Client
}

// NewClient builds a client for baseURL. A trailing slash is tolerated.
//
// apiKey is optional: /api/chat and the read endpoints are unprotected, and
// the service only demands X-API-Key on mutating routes when
// SecurityConfig.api_key_enabled is set (api/security.py).
func NewClient(baseURL, apiKey string) *Client {
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  strings.TrimSpace(apiKey),
		http: &http.Client{
			// A council run is a long request: several models generate in
			// sequence, and deliberation runs three rounds. The client
			// timeout is deliberately longer than the UI's own per-call
			// deadline, which is what actually reports a hang.
			Timeout: 30 * time.Minute,
		},
	}
}

// BaseURL returns the configured service root.
func (c *Client) BaseURL() string { return c.baseURL }

// HasAPIKey reports whether a key will be sent, so the UI can say so.
func (c *Client) HasAPIKey() bool { return c.apiKey != "" }

// ValidateBaseURL rejects anything that is not an absolute http(s) URL with a
// host, so a typo is caught in the form rather than at request time.
func ValidateBaseURL(raw string) error {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return errors.New("a service URL is required")
	}
	// Checked before url.Parse: "127.0.0.1:8000" fails to parse with a
	// confusing "first path segment in URL cannot contain colon", when what the
	// user actually did was leave off the scheme.
	if !strings.Contains(raw, "://") {
		return errors.New("use an http:// or https:// URL")
	}
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("not a valid URL: %w", err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return errors.New("use an http:// or https:// URL")
	}
	if u.Host == "" {
		return errors.New("the URL needs a host, for example 127.0.0.1:8000")
	}
	return nil
}

// do performs a request and decodes a JSON body into out.
func (c *Client) do(ctx context.Context, method, path string, body any, out any) error {
	var reader io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("encode request: %w", err)
		}
		reader = bytes.NewReader(encoded)
	}

	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, reader)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.apiKey != "" {
		req.Header.Set("X-API-Key", c.apiKey)
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrUnreachable, err)
	}
	defer resp.Body.Close()

	payload, err := io.ReadAll(io.LimitReader(resp.Body, 32<<20))
	if err != nil {
		return fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return fmt.Errorf("%s %s: HTTP %d: %s", method, path, resp.StatusCode, detail(payload))
	}

	if out == nil {
		return nil
	}
	if err := json.Unmarshal(payload, out); err != nil {
		return fmt.Errorf("decode %s body: %w", path, err)
	}
	return nil
}

// detail pulls FastAPI's {"detail": ...} string, falling back to a truncated
// body so an unexpected error page is still diagnosable.
func detail(payload []byte) string {
	var envelope struct {
		Detail any `json:"detail"`
	}
	if err := json.Unmarshal(payload, &envelope); err == nil && envelope.Detail != nil {
		if s, ok := envelope.Detail.(string); ok {
			return s
		}
		if encoded, err := json.Marshal(envelope.Detail); err == nil {
			return string(encoded)
		}
	}
	trimmed := strings.TrimSpace(string(payload))
	if len(trimmed) > 300 {
		trimmed = trimmed[:300] + "..."
	}
	if trimmed == "" {
		return "(empty response)"
	}
	return trimmed
}

// Health calls GET /health.
func (c *Client) Health(ctx context.Context) (*HealthResponse, error) {
	var out HealthResponse
	if err := c.do(ctx, http.MethodGet, "/health", nil, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Models calls GET /api/models.
func (c *Client) Models(ctx context.Context) ([]ModelInfo, error) {
	var out []ModelInfo
	if err := c.do(ctx, http.MethodGet, "/api/models", nil, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// Triads calls GET /api/triads.
func (c *Client) Triads(ctx context.Context) ([]TriadInfo, error) {
	var out TriadsResponse
	if err := c.do(ctx, http.MethodGet, "/api/triads", nil, &out); err != nil {
		return nil, err
	}
	return out.Triads, nil
}

// Leaderboard calls GET /api/observability/leaderboard.
func (c *Client) Leaderboard(ctx context.Context) ([]LeaderboardRow, error) {
	var out LeaderboardResponse
	if err := c.do(ctx, http.MethodGet, "/api/observability/leaderboard", nil, &out); err != nil {
		return nil, err
	}
	return out.Leaderboard, nil
}

// ObservabilitySummary calls GET /api/observability/summary (the control room).
func (c *Client) ObservabilitySummary(ctx context.Context) (map[string]any, error) {
	var out map[string]any
	if err := c.do(ctx, http.MethodGet, "/api/observability/summary", nil, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// Chat calls POST /api/chat.
//
// The blocking endpoint is used for scripted runs only; the interactive UI
// streams the same request over /ws/chat so round progress is visible.
func (c *Client) Chat(ctx context.Context, req CouncilRequest) (*CouncilResult, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	var out CouncilResult
	if err := c.do(ctx, http.MethodPost, "/api/chat", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Deliberate calls POST /api/deliberation.
func (c *Client) Deliberate(ctx context.Context, req DeliberationRequest) (*CouncilResult, error) {
	if strings.TrimSpace(req.Question) == "" {
		return nil, errors.New("the question is empty")
	}
	var out CouncilResult
	if err := c.do(ctx, http.MethodPost, "/api/deliberation", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Validate enforces the bounds the service itself enforces, so a request is
// rejected in the form instead of coming back as an HTTP 422.
func (r CouncilRequest) Validate() error {
	if strings.TrimSpace(r.Prompt) == "" {
		return errors.New("the question is empty")
	}
	switch r.Mode {
	case ModePersonality:
		// Deliberately not supported: personality mode needs a list of
		// {name, system_prompt} objects, and this front-end does not select
		// them. Returning an explicit error beats letting the service answer
		// with a 400 that says less. See the report's "not ported" section.
		return errors.New("personality mode is not available from this front-end; use multi_model or deliberation")

	case ModeDeliberation:
		// Deliberation resolves its own council: from the named triad, or from
		// the model list when no triad is given.
		if strings.TrimSpace(r.Triad) == "" && len(r.Models) == 0 {
			return errors.New("deliberation needs a triad or at least one model")
		}
		return nil

	default:
		if len(r.Models) == 0 {
			return errors.New("choose at least one model")
		}
		return nil
	}
}

// Council modes accepted by the service (cli/main.py --mode help and
// main.py:websocket_chat's mode dispatch).
const (
	ModeMultiModel   = "multi_model"
	ModeMultiRequest = "multi_request"
	ModeHybrid       = "hybrid"
	ModePersonality  = "personality"
	ModeDeliberation = "deliberation"

	VotingClassic  = "classic"
	VotingSemantic = "semantic"

	BackendOllama = "ollama"
	BackendPi     = "pi"
	BackendOpenAI = "openai"
)

// Modes are the swarm modes the service dispatches, in the order the CLI
// documents them (main.py:websocket_chat and cli/main.py --mode).
var Modes = []string{ModeMultiModel, ModeMultiRequest, ModeHybrid, ModePersonality, ModeDeliberation}

// SelectableModes are the modes this front-end offers.
//
// personality is excluded on purpose: the service requires a list of
// {name, system_prompt} objects for it, which this UI has no picker for, and
// offering a mode that can only fail would be worse than not offering it.
// CouncilRequest.Validate enforces the same rule for scripted callers.
var SelectableModes = []string{ModeMultiModel, ModeMultiRequest, ModeHybrid, ModeDeliberation}

// ModeDescription explains a mode in one line, for the form.
func ModeDescription(mode string) string {
	switch mode {
	case ModeMultiModel:
		return "Every model answers once; weighted majority decides."
	case ModeMultiRequest:
		return "One model, several temperatures; weighted majority decides."
	case ModeHybrid:
		return "Every model at several temperatures; weighted majority decides."
	case ModePersonality:
		return "Fixed personalities answer instead of raw models."
	case ModeDeliberation:
		return "Three-round council protocol with a verdict and dissent report."
	default:
		return ""
	}
}

// WSURL converts the service root into the /ws/chat WebSocket URL.
func (c *Client) WSURL(path string) string {
	root := c.baseURL
	switch {
	case strings.HasPrefix(root, "https://"):
		root = "wss://" + strings.TrimPrefix(root, "https://")
	case strings.HasPrefix(root, "http://"):
		root = "ws://" + strings.TrimPrefix(root, "http://")
	}
	return strings.TrimRight(root, "/") + path
}
