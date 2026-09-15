package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// ChatPath is the streaming chat endpoint (main.py:websocket_chat).
const ChatPath = "/ws/chat"

// readTimeout bounds how long the stream waits for the next frame.
//
// A council run generates for minutes, and the service emits nothing between
// rounds while the models are working, so this only has to be comfortably
// longer than one model's generation.
const readTimeout = 20 * time.Minute

// handshakeTimeout bounds the upgrade itself, which should be immediate.
const handshakeTimeout = 15 * time.Second

// Stream is a live /ws/chat connection.
//
// Events yields the decoded frames in order. The channel closes when the
// service sends "end", sends "error", or the connection drops: consumers can
// therefore range over it and then inspect Err.
type Stream struct {
	events chan StreamEvent
	conn   *websocket.Conn
	cancel context.CancelFunc
	once   sync.Once

	mu  sync.Mutex
	err error
}

// Events is the frame channel. It is closed when the stream ends.
func (s *Stream) Events() <-chan StreamEvent { return s.events }

// Err reports why the stream ended: nil after a clean "end" frame.
func (s *Stream) Err() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

// Close releases the connection. It is safe to call more than once, which
// matters because both the user and the reading goroutine can end a stream.
func (s *Stream) Close() {
	s.once.Do(func() {
		if s.cancel != nil {
			s.cancel()
		}
		if s.conn != nil {
			_ = s.conn.Close()
		}
	})
}

// StreamCouncil opens /ws/chat, sends the request frame and starts reading.
//
// The service answers a connection request with any number of frames before
// it sends "end"; the caller sees each one as it arrives, which is what makes
// round progress and per-model status visible.
func StreamCouncil(ctx context.Context, wsURL, apiKey string, req CouncilRequest) (*Stream, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	header := http.Header{}
	if key := strings.TrimSpace(apiKey); key != "" {
		// The service reads the same header on the upgrade request.
		header.Set("X-API-Key", key)
	}

	dialer := &websocket.Dialer{
		HandshakeTimeout: handshakeTimeout,
		Proxy:            http.ProxyFromEnvironment,
	}

	conn, resp, err := dialer.DialContext(ctx, wsURL, header)
	if err != nil {
		if resp != nil {
			return nil, fmt.Errorf("%w: HTTP %d: %v", ErrUnreachable, resp.StatusCode, err)
		}
		return nil, fmt.Errorf("%w: %v", ErrUnreachable, err)
	}

	if err := conn.WriteJSON(req); err != nil {
		_ = conn.Close()
		return nil, fmt.Errorf("send council request: %w", err)
	}

	streamCtx, cancel := context.WithCancel(ctx)
	stream := &Stream{
		events: make(chan StreamEvent, 64),
		conn:   conn,
		cancel: cancel,
	}
	go stream.read(streamCtx)
	return stream, nil
}

// read decodes frames until the service ends the stream or the connection
// fails. It owns closing the events channel.
func (s *Stream) read(ctx context.Context) {
	defer close(s.events)
	defer s.Close()

	for {
		if err := ctx.Err(); err != nil {
			s.setErr(err)
			return
		}

		_ = s.conn.SetReadDeadline(time.Now().Add(readTimeout))
		_, payload, err := s.conn.ReadMessage()
		if err != nil {
			// A cancelled context closes the socket underneath us; that is a
			// user action, not a failure worth reporting.
			if ctx.Err() != nil {
				s.setErr(ctx.Err())
				return
			}
			if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
				s.setErr(nil)
				return
			}
			s.setErr(fmt.Errorf("read stream: %w", err))
			return
		}

		var event StreamEvent
		if err := json.Unmarshal(payload, &event); err != nil {
			// A frame this client cannot decode must not kill a long run:
			// report it and keep reading.
			select {
			case s.events <- StreamEvent{Type: "undecodable", Message: err.Error()}:
			case <-ctx.Done():
				s.setErr(ctx.Err())
				return
			}
			continue
		}

		select {
		case s.events <- event:
		case <-ctx.Done():
			s.setErr(ctx.Err())
			return
		}

		switch event.Type {
		case "end":
			s.setErr(nil)
			return
		case "error":
			if event.Message != "" {
				s.setErr(errors.New(event.Message))
			} else {
				s.setErr(errors.New("the service reported an error"))
			}
			return
		}
	}
}

func (s *Stream) setErr(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err == nil {
		s.err = err
	}
}
