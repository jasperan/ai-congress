package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Client wraps the AI Congress REST API.
type Client struct {
	baseURL    string
	httpClient *http.Client
}

// NewClient creates an API client for the given base URL.
func NewClient(baseURL string) *Client {
	return &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 120 * time.Second, // Chat can take a while
		},
	}
}

// Ping checks if the backend is reachable via GET /health.
func (c *Client) Ping() error {
	return c.get("/health", nil)
}

// ListModels returns available Ollama models via GET /api/models.
func (c *Client) ListModels() ([]ModelInfo, error) {
	var result []ModelInfo
	if err := c.get("/api/models", &result); err != nil {
		return nil, err
	}
	return result, nil
}

// Chat sends a chat request via POST /api/chat.
func (c *Client) Chat(req ChatRequest) (*ChatResponse, error) {
	var result ChatResponse
	if err := c.post("/api/chat", req, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// EnhancedChat sends a chat request via POST /api/chat/enhanced.
func (c *Client) EnhancedChat(req EnhancedChatRequest) (*ChatResponse, error) {
	var result ChatResponse
	if err := c.post("/api/chat/enhanced", req, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// GetEnhancedStats returns performance stats via GET /api/enhanced/stats.
func (c *Client) GetEnhancedStats() (*StatsResponse, error) {
	var result StatsResponse
	if err := c.get("/api/enhanced/stats", &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// GetEnhancedRun returns a specific run via GET /api/enhanced/runs/{run_id}.
func (c *Client) GetEnhancedRun(runID string) (*EnhancedRunResponse, error) {
	var result EnhancedRunResponse
	if err := c.get("/api/enhanced/runs/"+runID, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// ListPersonalities returns all personalities via GET /api/personalities.
func (c *Client) ListPersonalities() ([]Personality, error) {
	var result []Personality
	if err := c.get("/api/personalities", &result); err != nil {
		return nil, err
	}
	return result, nil
}

// ListPersonalityLists returns personality set names via GET /api/personality-lists.
func (c *Client) ListPersonalityLists() ([]string, error) {
	var result []string
	if err := c.get("/api/personality-lists", &result); err != nil {
		return nil, err
	}
	return result, nil
}

// GetPersonalityList returns personalities from a set via GET /api/personality-list/{name}.
func (c *Client) GetPersonalityList(name string) ([]Personality, error) {
	var result []Personality
	if err := c.get("/api/personality-list/"+name, &result); err != nil {
		return nil, err
	}
	return result, nil
}

// PullModel pulls a model via POST /api/models/pull/{model_name}.
func (c *Client) PullModel(name string) error {
	return c.post("/api/models/pull/"+name, nil, nil)
}

// --- Internal helpers ---

func (c *Client) get(path string, result interface{}) error {
	resp, err := c.httpClient.Get(c.baseURL + path)
	if err != nil {
		return fmt.Errorf("GET %s: %w", path, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	if result != nil {
		if err := json.NewDecoder(resp.Body).Decode(result); err != nil {
			return fmt.Errorf("decoding response from %s: %w", path, err)
		}
	}
	return nil
}

func (c *Client) post(path string, body interface{}, result interface{}) error {
	var reqBody io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("marshaling request body: %w", err)
		}
		reqBody = bytes.NewReader(data)
	}

	resp, err := c.httpClient.Post(c.baseURL+path, "application/json", reqBody)
	if err != nil {
		return fmt.Errorf("POST %s: %w", path, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	if result != nil {
		if err := json.NewDecoder(resp.Body).Decode(result); err != nil {
			return fmt.Errorf("decoding response from %s: %w", path, err)
		}
	}
	return nil
}
