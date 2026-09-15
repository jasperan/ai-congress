package api

import (
	"context"
	"os"
	"testing"
	"time"
)

// Live-service integration tests.
//
// The repo marks service-dependent tests with @pytest.mark.integration on the
// Python side; this is the Go equivalent. They are skipped unless
// AICONGRESS_E2E_URL points at a running service, so `go test ./...` stays
// hermetic:
//
//	python run_server.py &
//	AICONGRESS_E2E_URL=http://127.0.0.1:8000 AICONGRESS_E2E_MODEL=smollm2:135m \
//	  go test ./internal/api/ -run E2E -v
//
// They exist because the unit tests can only prove the client agrees with
// itself. The vote_breakdown shape, for instance, was wrong in exactly the way
// a faked fixture could not reveal.
func e2eClient(t *testing.T) (*Client, string) {
	t.Helper()
	base := os.Getenv("AICONGRESS_E2E_URL")
	if base == "" {
		t.Skip("set AICONGRESS_E2E_URL to run the live-service integration tests")
	}
	model := os.Getenv("AICONGRESS_E2E_MODEL")
	if model == "" {
		t.Skip("set AICONGRESS_E2E_MODEL to a small installed model to run the live council test")
	}
	return NewClient(base, os.Getenv("AICONGRESS_API_KEY")), model
}

func TestE2EHealthAndCatalogs(t *testing.T) {
	client, _ := e2eClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	health, err := client.Health(ctx)
	if err != nil {
		t.Fatalf("Health() = %v", err)
	}
	if health.Status == "" {
		t.Error("Health() returned an empty status")
	}

	triads, err := client.Triads(ctx)
	if err != nil {
		t.Fatalf("Triads() = %v", err)
	}
	if len(triads) == 0 {
		t.Error("Triads() returned none; config/triads.json ships 20")
	}

	if _, err := client.Leaderboard(ctx); err != nil {
		t.Fatalf("Leaderboard() = %v", err)
	}
	if _, err := client.ObservabilitySummary(ctx); err != nil {
		t.Fatalf("ObservabilitySummary() = %v", err)
	}
}

// TestE2EBlockingCouncilRun checks the REST verdict path end to end, including
// that vote_breakdown decodes.
func TestE2EBlockingCouncilRun(t *testing.T) {
	client, model := e2eClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	result, err := client.Chat(ctx, CouncilRequest{
		Prompt:           "In one short sentence: is water wet?",
		Models:           []string{model},
		Mode:             ModeMultiModel,
		VotingMode:       VotingClassic,
		InferenceBackend: BackendOllama,
		Temperature:      0.7,
	})
	if err != nil {
		t.Fatalf("Chat() = %v", err)
	}
	if result.FinalAnswer == "" {
		t.Error("the service returned an empty final answer")
	}
	if len(result.VoteBreakdown) == 0 {
		t.Fatal("vote_breakdown was empty; the tally would have nothing to show")
	}
	for key, group := range result.VoteBreakdown {
		if group.Weight <= 0 {
			t.Errorf("group %q has weight %v", key, group.Weight)
		}
		if len(group.Models) == 0 {
			t.Errorf("group %q named no models, so the tally cannot label it", key)
		}
	}
}

// TestE2EStreamingFramesMatchTheClient is the one that matters for the
// interactive UI: it drives the real /ws/chat and asserts the frames decode into
// the types the TUI folds.
func TestE2EStreamingFramesMatchTheClient(t *testing.T) {
	client, model := e2eClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	stream, err := StreamCouncil(ctx, client.WSURL(ChatPath), os.Getenv("AICONGRESS_API_KEY"), CouncilRequest{
		Prompt:           "In one short sentence: is water wet?",
		Models:           []string{model},
		Mode:             ModeMultiModel,
		VotingMode:       VotingClassic,
		InferenceBackend: BackendOllama,
	})
	if err != nil {
		t.Fatalf("StreamCouncil() = %v", err)
	}
	defer stream.Close()

	seen := map[string]int{}
	sawFinal := false
	deadline := time.After(9 * time.Minute)

	for {
		select {
		case event, ok := <-stream.Events():
			if !ok {
				if err := stream.Err(); err != nil {
					t.Fatalf("stream ended with %v", err)
				}
				if !sawFinal {
					t.Fatal("the stream ended without a final_answer frame")
				}
				return
			}
			seen[event.Type]++
			switch event.Type {
			case "undecodable":
				// The client treats a bad frame as survivable, but a real
				// service sending one means the types are wrong.
				t.Errorf("the service sent a frame this client could not decode: %s", event.Message)
			case "final_answer":
				sawFinal = true
				if len(event.VoteBreakdown) == 0 {
					t.Error("final_answer carried an empty vote_breakdown")
				}
				if event.Data == nil || len(event.Data.FinalPositions) == 0 {
					t.Error("final_answer carried no final_positions, so the transcript would be empty")
				}
			case "end":
				t.Logf("frame types observed: %v", seen)
				return
			}
		case <-deadline:
			t.Fatalf("the stream produced no end frame in 9 minutes; types seen: %v", seen)
		}
	}
}
