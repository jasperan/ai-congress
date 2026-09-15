package tui

import (
	"strings"
	"testing"

	"github.com/jasperan/ai-congress/gotui/internal/api"
)

// voteGroup builds a vote_breakdown entry in the shape the service really
// sends: one record per distinct answer, carrying the models that pooled into
// it and the pooled weight. Shape captured from a live /ws/chat response and
// confirmed against voting_engine.py:weighted_majority_vote.
func voteGroup(weight float64, models ...string) api.VoteGroup {
	return api.VoteGroup{
		Original:       "answer from " + strings.Join(models, ","),
		OriginalWeight: weight,
		Weight:         weight,
		Votes:          []float64{weight},
		Models:         models,
	}
}

// TestNewCouncilSeedsEveryMemberAsPending pins that a run shows its whole
// roster before the first token arrives, so a slow model is visibly pending
// rather than simply absent.
func TestNewCouncilSeedsEveryMemberAsPending(t *testing.T) {
	council := NewCouncil("regulate?", api.ModeMultiModel, []string{"a", "b", "c"})

	if council.Question != "regulate?" {
		t.Errorf("Question = %q", council.Question)
	}
	if len(council.Members) != 3 {
		t.Fatalf("got %d members, want 3", len(council.Members))
	}
	for _, member := range council.Members {
		if member.Status != MemberPending {
			t.Errorf("member %q starts as %v, want pending", member.Name, member.Status)
		}
	}
	if council.Completed() != 0 {
		t.Errorf("Completed() = %d, want 0", council.Completed())
	}
	if !council.Running() {
		t.Error("a fresh council must be running")
	}
}

func TestApplyStartRecordsThePhase(t *testing.T) {
	council := NewCouncil("q", api.ModeMultiModel, []string{"a"})
	council.Apply(api.StreamEvent{Type: "start", Message: "Processing with 3 models..."})

	if council.Phase != "Processing with 3 models..." {
		t.Errorf("Phase = %q", council.Phase)
	}
}

func TestApplyStatusInitRegistersPersonalitiesAndTheirOrder(t *testing.T) {
	council := NewCouncil("q", api.ModePersonality, nil)
	council.Apply(api.StreamEvent{
		Type: "status_init",
		Personalities: []api.PersonalityStatus{
			{Name: "The Pragmatist", Status: "queued"},
			{Name: "The Adversary", Status: "queued"},
		},
	})

	if len(council.Members) != 2 {
		t.Fatalf("got %d members, want 2", len(council.Members))
	}
	if council.Members[0].Name != "The Pragmatist" || council.Members[1].Name != "The Adversary" {
		t.Errorf("order was not preserved: %+v", council.Members)
	}
	if council.Members[0].Detail != "queued" {
		t.Errorf("Detail = %q, want queued", council.Members[0].Detail)
	}
}

func TestApplyStatusUpdateTracksGeneratingThenComplete(t *testing.T) {
	council := NewCouncil("q", api.ModeMultiModel, []string{"qwen3.5:9b"})

	council.Apply(api.StreamEvent{Type: "status_update", Name: "qwen3.5:9b", Status: "Generating..."})
	if council.Members[0].Status != MemberGenerating {
		t.Fatalf("status = %v, want generating", council.Members[0].Status)
	}
	if council.Completed() != 0 {
		t.Error("a generating member must not count as completed")
	}

	council.Apply(api.StreamEvent{
		Type: "status_update", Name: "qwen3.5:9b", Status: "Complete", Response: "Yes, regulate.",
	})
	if council.Members[0].Status != MemberComplete {
		t.Fatalf("status = %v, want complete", council.Members[0].Status)
	}
	if council.Members[0].Response != "Yes, regulate." {
		t.Errorf("Response = %q", council.Members[0].Response)
	}
	if council.Completed() != 1 {
		t.Errorf("Completed() = %d, want 1", council.Completed())
	}
}

// TestApplyChunkAccumulatesLiveTokens is the "live transcript" behaviour: the
// answer grows as the model writes, it does not replace itself per chunk.
func TestApplyChunkAccumulatesLiveTokens(t *testing.T) {
	council := NewCouncil("q", api.ModeMultiModel, []string{"a"})
	for _, chunk := range []string{"Reg", "ulate", " it."} {
		council.Apply(api.StreamEvent{Type: "chunk", Name: "a", Content: chunk})
	}

	if council.Members[0].Response != "Regulate it." {
		t.Errorf("Response = %q, want the concatenated chunks", council.Members[0].Response)
	}
	if council.Members[0].Status != MemberGenerating {
		t.Errorf("status = %v, want generating", council.Members[0].Status)
	}
}

func TestApplyModelResponseCompletesAMember(t *testing.T) {
	council := NewCouncil("q", api.ModeMultiModel, []string{"a", "b"})
	council.Apply(api.StreamEvent{Type: "model_response", Model: "b", Content: "No."})

	if council.Members[1].Status != MemberComplete || council.Members[1].Response != "No." {
		t.Errorf("member b = %+v", council.Members[1])
	}
	if council.Members[0].Status != MemberPending {
		t.Errorf("member a = %v, want it untouched", council.Members[0].Status)
	}
}

func TestApplyFinalAnswerFoldsTheVerdict(t *testing.T) {
	council := NewCouncil("q", api.ModeDeliberation, nil)
	council.Apply(api.StreamEvent{
		Type:       "final_answer",
		Content:    "Regulate, carefully.",
		Confidence: 0.82,
		Mode:       "deliberation",
		VoteBreakdown: map[string]api.VoteGroup{
			"yes": voteGroup(2.4, "qwen3.5:9b"),
			"no":  voteGroup(1.1, "gemma4:latest"),
		},
		SemanticVote: &api.SemanticVote{
			WinningModel: "qwen3.5:9b",
			Consensus:    0.66,
			Clusters: []api.Cluster{
				{ID: 1, Label: "regulate softly", Models: []string{"qwen3.5:9b"}, KeyClaims: []string{"gradual"}},
			},
		},
		Data: &api.VerdictData{
			Rounds: []api.Round{
				{Name: "opening", Outputs: []api.DebateEntry{
					{Agent: "Pragmatist", Model: "qwen3.5:9b", Response: "ship it", Success: true},
				}},
				{Name: "rebuttal", Outputs: []api.DebateEntry{
					{Agent: "Adversary", Model: "gemma4:latest", Response: "fix it first", Success: true},
				}},
			},
			FinalPositions: []api.ModelResponse{
				{Agent: "Pragmatist", Model: "qwen3.5:9b", Response: "final: regulate", Success: true},
			},
		},
	})

	if !council.Done {
		t.Error("Done = false after a final answer")
	}
	if council.FinalAnswer != "Regulate, carefully." {
		t.Errorf("FinalAnswer = %q", council.FinalAnswer)
	}
	if council.Confidence != 0.82 {
		t.Errorf("Confidence = %v", council.Confidence)
	}
	if council.RoundCount() != 2 {
		t.Errorf("RoundCount() = %d, want 2", council.RoundCount())
	}
	if council.Winner() != "qwen3.5:9b" {
		t.Errorf("Winner() = %q", council.Winner())
	}
	if council.ConsensusShare() != 0.66 {
		t.Errorf("ConsensusShare() = %v, want the semantic consensus 0.66", council.ConsensusShare())
	}
	// The Pragmatist was introduced by final_positions, not by the request, so
	// it must have been appended rather than dropped.
	if len(council.Members) != 1 || council.Members[0].Name != "Pragmatist" {
		t.Fatalf("members = %+v, want the final position's agent appended", council.Members)
	}
	if council.Members[0].Response != "final: regulate" || council.Members[0].Status != MemberComplete {
		t.Errorf("final position was not folded in: %+v", council.Members[0])
	}
}

// TestApplyFinalAnswerFailsMembersThatNeverReported stops the transcript
// leaving a spinner running forever after a run ends.
func TestApplyFinalAnswerFailsMembersThatNeverReported(t *testing.T) {
	council := NewCouncil("q", api.ModeMultiModel, []string{"answered", "silent"})
	council.Apply(api.StreamEvent{Type: "model_response", Model: "answered", Content: "yes"})
	council.Apply(api.StreamEvent{Type: "final_answer", Content: "yes", Confidence: 1})

	if council.Members[1].Status != MemberFailed {
		t.Errorf("silent member status = %v, want failed", council.Members[1].Status)
	}
	if council.Members[0].Status != MemberComplete {
		t.Errorf("answered member status = %v, want complete", council.Members[0].Status)
	}
}

func TestApplyErrorStopsTheCouncil(t *testing.T) {
	council := NewCouncil("q", api.ModeMultiModel, []string{"a"})
	council.Apply(api.StreamEvent{Type: "error", Message: "Ollama is down"})

	if council.Error != "Ollama is down" {
		t.Errorf("Error = %q", council.Error)
	}
	if council.Running() {
		t.Error("Running() = true after an error")
	}
}

// TestApplyRegistersAnUnknownMember keeps the transcript honest when the
// service names a participant the request never mentioned.
func TestApplyRegistersAnUnknownMember(t *testing.T) {
	council := NewCouncil("q", api.ModeMultiModel, []string{"a"})
	council.Apply(api.StreamEvent{Type: "model_response", Model: "surprise-model", Content: "hi"})

	if len(council.Members) != 2 {
		t.Fatalf("got %d members, want the unseen one appended", len(council.Members))
	}
	if council.Members[1].Name != "surprise-model" {
		t.Errorf("appended member = %q", council.Members[1].Name)
	}
}

func TestApplyIgnoresAFrameWithNoName(t *testing.T) {
	council := NewCouncil("q", api.ModeMultiModel, []string{"a"})
	before := len(council.Members)
	council.Apply(api.StreamEvent{Type: "model_response", Content: "orphan"})

	if len(council.Members) != before {
		t.Errorf("a nameless frame added a member: %+v", council.Members)
	}
}

func TestProgressFractionTracksCompletion(t *testing.T) {
	council := NewCouncil("q", api.ModeMultiModel, []string{"a", "b", "c", "d"})
	if got := council.ProgressFraction(); got != 0 {
		t.Errorf("ProgressFraction() = %v, want 0", got)
	}
	council.Apply(api.StreamEvent{Type: "model_response", Model: "a", Content: "x"})
	council.Apply(api.StreamEvent{Type: "model_response", Model: "b", Content: "y"})
	if got := council.ProgressFraction(); got != 0.5 {
		t.Errorf("ProgressFraction() = %v, want 0.5", got)
	}
}

func TestTalliesRankAndShare(t *testing.T) {
	council := NewCouncil("q", api.ModeMultiModel, nil)
	council.VoteBreakdown = map[string]api.VoteGroup{
		"low":  voteGroup(1, "low"),
		"high": voteGroup(6, "high"),
		"mid":  voteGroup(3, "mid"),
	}

	rows := council.Tallies()
	if len(rows) != 3 {
		t.Fatalf("got %d rows, want 3", len(rows))
	}
	if rows[0].Name != "high" || rows[1].Name != "mid" || rows[2].Name != "low" {
		t.Errorf("order = %v", []string{rows[0].Name, rows[1].Name, rows[2].Name})
	}
	if !rows[0].Winner || rows[1].Winner {
		t.Error("only the top row should be marked the winner")
	}
	if rows[0].Share != 0.6 {
		t.Errorf("winner share = %v, want 0.6", rows[0].Share)
	}
	if council.Winner() != "high" {
		t.Errorf("Winner() = %q", council.Winner())
	}
}

// TestTalliesAreStableOnATie pins the tie-break: an unstable order would make
// the tally jump between renders.
func TestTalliesAreStableOnATie(t *testing.T) {
	council := NewCouncil("q", api.ModeMultiModel, nil)
	council.VoteBreakdown = map[string]api.VoteGroup{
		"z": voteGroup(2, "zeta"),
		"a": voteGroup(2, "alpha"),
		"m": voteGroup(2, "mid"),
	}

	first := council.Tallies()
	for i := 0; i < 20; i++ {
		again := council.Tallies()
		for j := range first {
			if first[j].Name != again[j].Name {
				t.Fatalf("tally order changed between calls: %v then %v",
					names(first), names(again))
			}
		}
	}
	if got := names(first); got[0] != "alpha" || got[1] != "mid" || got[2] != "zeta" {
		t.Errorf("tie order = %v, want alphabetical", got)
	}
}

func names(rows []Tally) []string {
	out := make([]string, 0, len(rows))
	for _, row := range rows {
		out = append(out, row.Name)
	}
	return out
}

func TestNoVotesMeansNoWinner(t *testing.T) {
	council := NewCouncil("q", api.ModeMultiModel, []string{"a"})
	if council.Winner() != "" {
		t.Errorf("Winner() = %q on an empty tally", council.Winner())
	}
	if council.ConsensusShare() != 0 {
		t.Errorf("ConsensusShare() = %v on an empty tally", council.ConsensusShare())
	}
	if rows := council.Tallies(); rows != nil {
		t.Errorf("Tallies() = %v, want nil", rows)
	}
}

// TestConsensusPrefersTheSemanticJudgement documents the deliberate choice:
// a 100% classic share (the only model that answered) must not be reported as
// unanimous agreement when semantic voting measured something lower.
func TestConsensusPrefersTheSemanticJudgement(t *testing.T) {
	council := NewCouncil("q", api.ModeMultiModel, nil)
	council.VoteBreakdown = map[string]api.VoteGroup{"only": voteGroup(1, "only")}
	council.Semantic = &api.SemanticVote{Consensus: 0.4, WinningModel: "only"}

	if got := council.ConsensusShare(); got != 0.4 {
		t.Errorf("ConsensusShare() = %v, want the semantic 0.4, not the classic 1.0", got)
	}
}

func TestNilCouncilIsSafe(t *testing.T) {
	var council *Council
	if council.Completed() != 0 || council.Running() || council.ProgressFraction() != 0 ||
		council.RoundCount() != 0 || council.Winner() != "" || council.ConsensusShare() != 0 {
		t.Error("a nil council returned a non-zero value")
	}
	if rows := council.Tallies(); rows != nil {
		t.Error("nil council returned tallies")
	}
}

// --- rendering ---------------------------------------------------------------------

func TestTallyRowsMarksTheWinnerAndRendersEveryMember(t *testing.T) {
	council := NewCouncil("q", api.ModeMultiModel, nil)
	council.VoteBreakdown = map[string]api.VoteGroup{
		"yes": voteGroup(5, "qwen3.5:9b"),
		"no":  voteGroup(2, "gemma4:latest"),
	}

	rendered := TallyRows(council, 100)
	for _, want := range []string{"qwen3.5:9b", "gemma4:latest", "votes"} {
		if !strings.Contains(rendered, want) {
			t.Errorf("tally is missing %q:\n%s", want, rendered)
		}
	}
	if !strings.Contains(rendered, "> ") {
		t.Errorf("the winner is not marked:\n%s", rendered)
	}
	if !strings.Contains(rendered, "71%") {
		t.Errorf("the winner's share is not shown:\n%s", rendered)
	}
}

func TestTallyRowsSaysSoWhenThereAreNoVotes(t *testing.T) {
	rendered := TallyRows(NewCouncil("q", api.ModeMultiModel, nil), 80)
	if !strings.Contains(rendered, "No votes") {
		t.Errorf("empty tally rendered %q", rendered)
	}
}

func TestMemberRowsShowsEveryState(t *testing.T) {
	council := NewCouncil("q", api.ModeMultiModel, []string{"done", "busy", "quiet"})
	council.Apply(api.StreamEvent{Type: "model_response", Model: "done", Content: "x"})
	council.Apply(api.StreamEvent{Type: "status_update", Name: "busy", Status: "Generating..."})

	rendered := MemberRows(council, 100)
	for _, want := range []string{"done", "busy", "quiet", "complete", "generating", "pending"} {
		if !strings.Contains(rendered, want) {
			t.Errorf("member rows are missing %q:\n%s", want, rendered)
		}
	}
}

func TestRoundRowsReportsPerRoundPositions(t *testing.T) {
	council := NewCouncil("q", api.ModeDeliberation, nil)
	council.Rounds = []api.Round{
		{Name: "opening", Outputs: []api.DebateEntry{
			{Agent: "Pragmatist", Response: "ship it", Success: true},
			{Agent: "Adversary", Response: "no", Success: true},
		}},
		{Name: "restate", Outputs: []api.DebateEntry{
			{Agent: "Pragmatist", Response: "", Success: false},
		}},
	}

	rendered := RoundRows(council, 100)
	for _, want := range []string{"opening", "restate", "2/2", "0/1", "Pragmatist"} {
		if !strings.Contains(rendered, want) {
			t.Errorf("round rows are missing %q:\n%s", want, rendered)
		}
	}
}

func TestRoundRowsSaysWhenTheProtocolDidNotRun(t *testing.T) {
	rendered := RoundRows(NewCouncil("q", api.ModeMultiModel, nil), 80)
	if !strings.Contains(rendered, "deliberation") {
		t.Errorf("rendered %q, want it to say the protocol did not run", rendered)
	}
}

func TestTranscriptRowsShowsResponsesAndSilence(t *testing.T) {
	council := NewCouncil("q", api.ModeMultiModel, []string{"talker", "silent"})
	council.Apply(api.StreamEvent{Type: "model_response", Model: "talker", Content: "a considered answer"})

	rendered := TranscriptRows(council, 100)
	if !strings.Contains(rendered, "a considered answer") {
		t.Errorf("transcript is missing the answer:\n%s", rendered)
	}
	if !strings.Contains(rendered, "waiting") {
		t.Errorf("transcript does not mark the pending member:\n%s", rendered)
	}
}

func TestVerdictRowsSummarisesTheRun(t *testing.T) {
	council := NewCouncil("Should AI be regulated?", api.ModeDeliberation, []string{"a"})
	council.Done = true
	council.Confidence = 0.9
	council.VoteBreakdown = map[string]api.VoteGroup{"a": voteGroup(1, "a")}
	council.Rounds = []api.Round{{Name: "opening"}}

	rendered := VerdictRows(council, 100)
	for _, want := range []string{"mode", "deliberation", "confidence", "90%", "winner", "rounds"} {
		if !strings.Contains(rendered, want) {
			t.Errorf("verdict is missing %q:\n%s", want, rendered)
		}
	}
}

func TestSemanticRowsRendersClustersAndDissent(t *testing.T) {
	council := NewCouncil("q", api.ModeMultiModel, nil)
	council.Semantic = &api.SemanticVote{
		Consensus:         0.5,
		WinningModel:      "a",
		DissentingSummary: "one model wanted a ban",
		Clusters: []api.Cluster{
			{ID: 1, Label: "regulate softly", Models: []string{"a", "b"}, KeyClaims: []string{"gradual"}},
		},
	}

	rendered := SemanticRows(council, 100)
	for _, want := range []string{"regulate softly", "a, b", "gradual", "one model wanted a ban"} {
		if !strings.Contains(rendered, want) {
			t.Errorf("semantic rows are missing %q:\n%s", want, rendered)
		}
	}
	if SemanticRows(NewCouncil("q", api.ModeMultiModel, nil), 80) != "" {
		t.Error("SemanticRows rendered something without a semantic result")
	}
}

func TestLeaderboardRowsRendersAHeaderAndRows(t *testing.T) {
	rows := []api.LeaderboardRow{
		{Model: "qwen3.5:9b", Weight: 0.912, WinRate: 0.5, Participations: 7},
	}
	rendered := LeaderboardRows(rows, 100)
	for _, want := range []string{"model", "weight", "win rate", "qwen3.5:9b", "0.912", "50%"} {
		if !strings.Contains(rendered, want) {
			t.Errorf("leaderboard is missing %q:\n%s", want, rendered)
		}
	}
	if got := LeaderboardRows(nil, 80); !strings.Contains(got, "No runs") {
		t.Errorf("empty leaderboard rendered %q", got)
	}
}

func TestModelRowsRendersModelsAndTheirBackend(t *testing.T) {
	models := []api.ModelInfo{
		{Name: "qwen3.5:9b", Weight: 0.87, Backend: "ollama"},
		{Name: "deepseek-v4-flash", Weight: 1, Backend: "pi"},
	}
	rendered := ModelRows(models, 100)
	for _, want := range []string{"qwen3.5:9b", "ollama", "deepseek-v4-flash", "pi"} {
		if !strings.Contains(rendered, want) {
			t.Errorf("model rows are missing %q:\n%s", want, rendered)
		}
	}
	if got := ModelRows(nil, 80); !strings.Contains(got, "Ollama") {
		t.Errorf("empty model list rendered %q, want a hint about Ollama", got)
	}
}

func TestTriadRowsRendersNameAndDescription(t *testing.T) {
	triads := []api.TriadInfo{{Name: "architecture", Description: "built to disagree"}}
	rendered := TriadRows(triads, 100)
	if !strings.Contains(rendered, "architecture") || !strings.Contains(rendered, "built to disagree") {
		t.Errorf("triad rows = %q", rendered)
	}
	if got := TriadRows(nil, 80); !strings.Contains(got, "No triads") {
		t.Errorf("empty triad list rendered %q", got)
	}
}

func TestOneLineCollapsesWhitespace(t *testing.T) {
	if got := oneLine("a\n\nb\t c "); got != "a b c" {
		t.Errorf("oneLine() = %q, want %q", got, "a b c")
	}
	if got := oneLine("   "); got != "" {
		t.Errorf("oneLine(blank) = %q, want empty", got)
	}
}

func TestWrapLinesWrapsAndPreservesWords(t *testing.T) {
	wrapped := wrapLines("one two three four five six seven", 12)
	for _, line := range strings.Split(wrapped, "\n") {
		if len([]rune(line)) > 12 {
			t.Errorf("line %q exceeds the width", line)
		}
	}
	if strings.Join(strings.Fields(wrapped), " ") != "one two three four five six seven" {
		t.Errorf("wrapping lost or reordered words: %q", wrapped)
	}
	if wrapLines("", 20) != "" {
		t.Error("wrapLines(empty) is not empty")
	}
}
