package api

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

// wsServer starts a test WebSocket endpoint that runs script for each
// connection, then sends whatever the script returns.
func wsServer(t *testing.T, script func(t *testing.T, conn *websocket.Conn)) (*httptest.Server, string) {
	t.Helper()
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != ChatPath {
			t.Errorf("upgrade path = %q, want %q", r.URL.Path, ChatPath)
		}
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("upgrade: %v", err)
			return
		}
		defer conn.Close()
		script(t, conn)
	}))
	t.Cleanup(server.Close)
	return server, "ws" + strings.TrimPrefix(server.URL, "http") + ChatPath
}

// drainToEnd reads events until the channel closes or the deadline passes.
func drainToEnd(t *testing.T, stream *Stream) []StreamEvent {
	t.Helper()
	var events []StreamEvent
	deadline := time.After(20 * time.Second)
	for {
		select {
		case event, ok := <-stream.Events():
			if !ok {
				return events
			}
			events = append(events, event)
		case <-deadline:
			t.Fatal("stream did not close within 20s")
		}
	}
}

func TestStreamDeliversEveryFrameInOrder(t *testing.T) {
	_, url := wsServer(t, func(t *testing.T, conn *websocket.Conn) {
		var request map[string]any
		if err := conn.ReadJSON(&request); err != nil {
			t.Errorf("read request: %v", err)
			return
		}
		if request["prompt"] != "regulate?" {
			t.Errorf("prompt = %v, want %q", request["prompt"], "regulate?")
		}

		frames := []string{
			`{"type":"start","message":"Processing with 2 models..."}`,
			`{"type":"status_update","name":"qwen3.5:9b","status":"Generating..."}`,
			`{"type":"chunk","name":"qwen3.5:9b","content":"Ye"}`,
			`{"type":"chunk","name":"qwen3.5:9b","content":"s"}`,
			`{"type":"status_update","name":"qwen3.5:9b","status":"Complete","response":"Yes"}`,
			`{"type":"model_response","model":"gemma4:latest","content":"No"}`,
			`{"type":"final_answer","content":"Yes","confidence":0.75,
			  "vote_breakdown":{"yes":{"original":"Yes","original_weight":0.5,"weight":2.0,"votes":[0.5],"models":["qwen3.5:9b"]}},
			  "data":{"rounds":[{"name":"opening","outputs":[{"agent":"Pragmatist","response":"yes","success":true}]}]}}`,
			`{"type":"end"}`,
		}
		for _, frame := range frames {
			if err := conn.WriteMessage(websocket.TextMessage, []byte(frame)); err != nil {
				t.Errorf("write frame: %v", err)
				return
			}
		}
	})

	stream, err := StreamCouncil(context.Background(), url, "",
		CouncilRequest{Prompt: "regulate?", Models: []string{"qwen3.5:9b", "gemma4:latest"}, Mode: ModeMultiModel})
	if err != nil {
		t.Fatalf("StreamCouncil() = %v", err)
	}
	defer stream.Close()

	events := drainToEnd(t, stream)
	if len(events) != 8 {
		t.Fatalf("got %d events, want 8: %+v", len(events), events)
	}
	if events[0].Type != "start" || events[0].Message == "" {
		t.Errorf("first event = %+v, want a start frame with a message", events[0])
	}
	if events[7].Type != "end" {
		t.Errorf("last event = %+v, want end", events[7])
	}

	final := events[6]
	if final.Type != "final_answer" {
		t.Fatalf("event 7 = %q, want final_answer", final.Type)
	}
	if final.Confidence != 0.75 {
		t.Errorf("confidence = %v, want 0.75", final.Confidence)
	}
	if group, ok := final.VoteBreakdown["yes"]; !ok || group.Weight != 2.0 {
		t.Errorf("vote_breakdown = %v, want one group keyed \"yes\" weighing 2.0", final.VoteBreakdown)
	} else if len(group.Models) != 1 || group.Models[0] != "qwen3.5:9b" {
		t.Errorf("vote group models = %v", group.Models)
	}
	if final.Data == nil || len(final.Data.Rounds) != 1 || final.Data.Rounds[0].Name != "opening" {
		t.Errorf("deliberation payload did not decode: %+v", final.Data)
	}

	if err := stream.Err(); err != nil {
		t.Errorf("Err() = %v after a clean end, want nil", err)
	}
}

func TestStreamSurfacesAnErrorFrame(t *testing.T) {
	_, url := wsServer(t, func(t *testing.T, conn *websocket.Conn) {
		_, _, _ = conn.ReadMessage()
		_ = conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"start","message":"working"}`))
		_ = conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"error","message":"Ollama is not reachable"}`))
	})

	stream, err := StreamCouncil(context.Background(), url, "",
		CouncilRequest{Prompt: "why", Models: []string{"m"}, Mode: ModeMultiModel})
	if err != nil {
		t.Fatalf("StreamCouncil() = %v", err)
	}
	defer stream.Close()

	events := drainToEnd(t, stream)
	if len(events) != 2 || events[1].Type != "error" {
		t.Fatalf("events = %+v, want a start then an error", events)
	}
	if events[1].IsTerminal() != true {
		t.Error("an error frame should be terminal")
	}
	if err := stream.Err(); err == nil || !strings.Contains(err.Error(), "Ollama is not reachable") {
		t.Errorf("Err() = %v, want the service's message", err)
	}
}

// TestStreamSkipsAnUndecodableFrame pins the decision that one bad frame must
// not kill a long run: the caller is told, and reading continues.
func TestStreamSkipsAnUndecodableFrame(t *testing.T) {
	_, url := wsServer(t, func(t *testing.T, conn *websocket.Conn) {
		_, _, _ = conn.ReadMessage()
		_ = conn.WriteMessage(websocket.TextMessage, []byte(`not json at all`))
		_ = conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"end"}`))
	})

	stream, err := StreamCouncil(context.Background(), url, "",
		CouncilRequest{Prompt: "why", Models: []string{"m"}, Mode: ModeMultiModel})
	if err != nil {
		t.Fatalf("StreamCouncil() = %v", err)
	}
	defer stream.Close()

	events := drainToEnd(t, stream)
	if len(events) != 2 {
		t.Fatalf("events = %+v, want an undecodable marker then end", events)
	}
	if events[0].Type != "undecodable" || events[0].Message == "" {
		t.Errorf("first event = %+v, want an undecodable marker carrying the cause", events[0])
	}
	if events[1].Type != "end" {
		t.Errorf("second event = %+v, want end", events[1])
	}
	if err := stream.Err(); err != nil {
		t.Errorf("Err() = %v, want nil: a bad frame is not a stream failure", err)
	}
}

func TestStreamSendsTheAPIKeyOnUpgrade(t *testing.T) {
	upgrader := websocket.Upgrader{}
	var seen string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = r.Header.Get("X-API-Key")
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		_, _, _ = conn.ReadMessage()
		_ = conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"end"}`))
	}))
	defer server.Close()

	url := "ws" + strings.TrimPrefix(server.URL, "http") + ChatPath
	stream, err := StreamCouncil(context.Background(), url, "  secret-key  ",
		CouncilRequest{Prompt: "why", Models: []string{"m"}, Mode: ModeMultiModel})
	if err != nil {
		t.Fatalf("StreamCouncil() = %v", err)
	}
	defer stream.Close()
	drainToEnd(t, stream)

	if seen != "secret-key" {
		t.Errorf("X-API-Key = %q, want %q (the key must be trimmed)", seen, "secret-key")
	}
}

func TestStreamRejectsAnInvalidRequestBeforeDialling(t *testing.T) {
	stream, err := StreamCouncil(context.Background(), "ws://127.0.0.1:1/ws/chat", "",
		CouncilRequest{Prompt: "  ", Models: []string{"m"}})
	if err == nil {
		stream.Close()
		t.Fatal("StreamCouncil() = nil for an empty prompt")
	}
	if !strings.Contains(err.Error(), "empty") {
		t.Errorf("error = %q, want it to explain the empty question", err)
	}
}

func TestStreamReportsAnUnreachableService(t *testing.T) {
	// A plain HTTP endpoint that never upgrades: the handshake fails.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "not a websocket", http.StatusBadRequest)
	}))
	defer server.Close()

	url := "ws" + strings.TrimPrefix(server.URL, "http") + ChatPath
	stream, err := StreamCouncil(context.Background(), url, "",
		CouncilRequest{Prompt: "why", Models: []string{"m"}, Mode: ModeMultiModel})
	if err == nil {
		stream.Close()
		t.Fatal("StreamCouncil() = nil against a non-WebSocket endpoint")
	}
	if !errors.Is(err, ErrUnreachable) {
		t.Errorf("error = %v, want it to wrap ErrUnreachable so the UI can say 'start the service'", err)
	}
}

func TestStreamCloseIsIdempotentAndUnblocksTheReader(t *testing.T) {
	_, url := wsServer(t, func(t *testing.T, conn *websocket.Conn) {
		_, _, _ = conn.ReadMessage()
		// Never send anything: the client must still be able to give up.
		<-time.After(5 * time.Second)
	})

	stream, err := StreamCouncil(context.Background(), url, "",
		CouncilRequest{Prompt: "why", Models: []string{"m"}, Mode: ModeMultiModel})
	if err != nil {
		t.Fatalf("StreamCouncil() = %v", err)
	}

	stream.Close()
	stream.Close() // must not panic

	// Closing the connection ends the reader, which closes the channel.
	select {
	case _, ok := <-stream.Events():
		if ok {
			// A frame may still be buffered; drain until closed.
			for range stream.Events() {
			}
		}
	case <-time.After(20 * time.Second):
		t.Fatal("Close() did not unblock the reader")
	}
}
