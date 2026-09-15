package tui

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/jasperan/ai-congress/gotui/internal/api"
	"github.com/jasperan/ai-congress/gotui/internal/session"
)

// PlainOptions drive the non-full-screen path.
//
// This path is used when ACCESSIBLE is set and when stdin is not a terminal.
// It reuses the SAME settings and request types as the TUI, so the two
// front-ends cannot drift apart in what they ask or what they validate.
type PlainOptions struct {
	ProjectRoot string
	Settings    session.Settings
	APIKey      string
	Triads      []api.TriadInfo
	Input       io.Reader
	Output      io.Writer
}

// RunPlainPrompts runs the whole accessible flow as ONE pass of plain prompts.
//
// One form, one read of the reader. huh's accessible Form.Run wraps the reader
// in its own scanner, which buffers ahead; a second form built over the same
// stdin therefore starts at EOF and returns empty answers for everything. That
// is why connection and council are one group here rather than two forms.
//
// The returned request is already validated, so the caller can send it without
// re-deriving anything.
func RunPlainPrompts(opts PlainOptions) (session.Settings, api.CouncilRequest, string, string, error) {
	in, out := opts.Input, opts.Output
	if in == nil {
		in = os.Stdin
	}
	if out == nil {
		out = os.Stdout
	}

	answers := PlainAnswers{
		Connect: ConnectDefaults(opts.Settings),
		Council: CouncilDefaults(),
	}
	form := PlainForm(&answers, opts.Triads, session.PasswordFromEnv() != "", finalAPIKey(opts.APIKey) != "").
		WithInput(in).
		WithOutput(out)

	if err := form.Run(); err != nil {
		return session.Settings{}, api.CouncilRequest{}, "", "", err
	}

	password := answers.Connect.OraclePassword
	if password == "" {
		password = session.PasswordFromEnv()
	}
	apiKey := strings.TrimSpace(answers.Connect.APIKey)
	if apiKey == "" {
		apiKey = finalAPIKey(opts.APIKey)
	}

	settings := answers.Connect.ToSettings()
	answers.Council.Models = ParseMembers(answers.Members)
	request := answers.Council.ToRequest()
	if err := request.Validate(); err != nil {
		return settings, api.CouncilRequest{}, password, apiKey, err
	}
	return settings, request, password, apiKey, nil
}

// finalAPIKey prefers an explicit key, then the environment.
func finalAPIKey(explicit string) string {
	if key := strings.TrimSpace(explicit); key != "" {
		return key
	}
	return session.APIKeyFromEnv()
}

// StartServiceIfRequested honours the connection form's launch answer.
func StartServiceIfRequested(ctx context.Context, settings session.Settings, password, apiKey, projectRoot string, out io.Writer) (*session.Server, error) {
	if !settings.LaunchServer {
		return nil, nil
	}
	fmt.Fprintf(out, "Starting the AI Congress API on port %d...\n", settings.Port)
	server, err := session.LaunchServer(ctx, projectRoot, settings.Port,
		session.ServerEnv(settings, password, apiKey))
	if err != nil {
		return nil, err
	}
	if err := session.WaitForPort(ctx, "127.0.0.1", settings.Port, 90*time.Second); err != nil {
		server.Stop()
		return nil, err
	}
	return server, nil
}

// ActionRequest is a scripted, non-interactive request.
type ActionRequest struct {
	Action string
	// Ask fields.
	Question string
	Mode     string
	Voting   string
	Backend  string
	Triad    string
	Models   []string
	Evidence bool
	JSON     bool
}

// RunAction executes a scripted request and writes its result.
func RunAction(ctx context.Context, client *api.Client, req ActionRequest, out io.Writer) error {
	switch req.Action {
	case ActionHealth:
		health, err := client.Health(ctx)
		if err != nil {
			return err
		}
		if req.JSON {
			return writeJSON(out, health)
		}
		fmt.Fprintf(out, "status       %s\n", health.Status)
		fmt.Fprintf(out, "service      %s\n", client.BaseURL())
		return nil

	case ActionListModels:
		models, err := client.Models(ctx)
		if err != nil {
			return err
		}
		if req.JSON {
			return writeJSON(out, models)
		}
		if len(models) == 0 {
			fmt.Fprintln(out, "The service reported no models. Is Ollama running?")
			return nil
		}
		for _, model := range models {
			fmt.Fprintf(out, "  %-32s weight %.3f  %s\n", model.Name, model.Weight, model.Backend)
		}
		return nil

	case ActionListTriads:
		triads, err := client.Triads(ctx)
		if err != nil {
			return err
		}
		if req.JSON {
			return writeJSON(out, triads)
		}
		for _, triad := range triads {
			fmt.Fprintf(out, "  %-24s %s\n", triad.Name, triad.Description)
		}
		return nil

	case ActionStandings:
		rows, err := client.Leaderboard(ctx)
		if err != nil {
			return err
		}
		if req.JSON {
			return writeJSON(out, rows)
		}
		if len(rows) == 0 {
			fmt.Fprintln(out, "No runs have been recorded yet.")
			return nil
		}
		for _, row := range rows {
			fmt.Fprintf(out, "  %-28s weight %.3f  win rate %s  %d runs\n",
				row.Model, row.Weight, Percent(row.WinRate), row.Participations)
		}
		return nil

	case ActionControlRoom:
		summary, err := client.ObservabilitySummary(ctx)
		if err != nil {
			return err
		}
		if req.JSON {
			return writeJSON(out, summary)
		}
		return writeControlRoom(out, summary)

	case ActionAsk:
		return runAsk(ctx, client, req, out)
	}

	return fmt.Errorf("unknown action %q", req.Action)
}

// runAsk runs one council through the blocking REST endpoint.
//
// The interactive UI streams over /ws/chat instead; this path exists so a
// script or a screen reader gets the same result as a plain document.
func runAsk(ctx context.Context, client *api.Client, req ActionRequest, out io.Writer) error {
	answers := CouncilAnswers{
		Question:    req.Question,
		Mode:        firstNonEmpty(req.Mode, api.ModeMultiModel),
		VotingMode:  firstNonEmpty(req.Voting, api.VotingClassic),
		Backend:     firstNonEmpty(req.Backend, api.BackendOllama),
		Triad:       req.Triad,
		Models:      req.Models,
		Evidence:    req.Evidence,
		Temperature: "0.7",
	}
	request := answers.ToRequest()
	if err := request.Validate(); err != nil {
		return err
	}

	var (
		result *api.CouncilResult
		err    error
	)
	if request.Mode == api.ModeDeliberation {
		result, err = client.Deliberate(ctx, api.DeliberationRequest{
			Question:         request.Prompt,
			Triad:            request.Triad,
			Models:           request.Models,
			Temperature:      request.Temperature,
			InferenceBackend: request.InferenceBackend,
			Evidence:         request.Evidence,
		})
	} else {
		result, err = client.Chat(ctx, request)
	}
	if err != nil {
		return err
	}

	if req.JSON {
		return writeJSON(out, result)
	}

	fmt.Fprintln(out, result.FinalAnswer)
	fmt.Fprintln(out)

	council := NewCouncil(request.Prompt, result.Mode, request.Models)
	council.Done = true
	council.FinalAnswer = result.FinalAnswer
	council.Confidence = result.Confidence
	council.VoteBreakdown = result.VoteBreakdown
	council.Semantic = result.SemanticVote
	council.Rounds = result.Rounds
	for _, response := range result.Responses {
		marker := "  + "
		if !response.Success {
			marker = "  ! "
		}
		fmt.Fprintf(out, "\n%s%s\n", marker, response.Model)
		fmt.Fprintf(out, "%s\n", indent(strings.TrimSpace(response.Response), "    "))
	}
	for _, round := range result.Rounds {
		fmt.Fprintf(out, "\nround: %s\n", round.Name)
		for _, entry := range round.Outputs {
			speaker := entry.Agent
			if speaker == "" {
				speaker = entry.Model
			}
			fmt.Fprintf(out, "    %s\n", speaker)
		}
	}
	fmt.Fprintln(out)
	fmt.Fprintf(out, "confidence   %s\n", Percent(result.Confidence))
	fmt.Fprintf(out, "consensus    %s\n", Percent(council.ConsensusShare()))
	if winner := council.Winner(); winner != "" {
		fmt.Fprintf(out, "winner       %s\n", winner)
	}
	if len(result.VoteBreakdown) > 0 {
		fmt.Fprintln(out, "votes")
		for _, row := range council.Tallies() {
			fmt.Fprintf(out, "  %-28s %6.1f  %s\n", row.Name, row.Votes, Percent(row.Share))
		}
	}
	return nil
}

// writeControlRoom renders the observability summary as text.
//
// The summary is an untyped map, so unknown keys are reported rather than
// guessed at.
func writeControlRoom(out io.Writer, summary map[string]any) error {
	if len(summary) == 0 {
		fmt.Fprintln(out, "The service returned no observability data.")
		return nil
	}
	keys := make([]string, 0, len(summary))
	for key := range summary {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		switch value := summary[key].(type) {
		case []any:
			fmt.Fprintf(out, "%-22s %d entries\n", key, len(value))
		case map[string]any:
			fmt.Fprintf(out, "%-22s %d keys\n", key, len(value))
		default:
			fmt.Fprintf(out, "%-22s %v\n", key, value)
		}
	}
	return nil
}

// indent prefixes every non-empty line, for the scripted transcript.
func indent(text, prefix string) string {
	if text == "" {
		return prefix
	}
	lines := strings.Split(text, "\n")
	for i, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}
		lines[i] = prefix + line
	}
	return strings.Join(lines, "\n")
}

// firstNonEmpty returns the first non-blank value.
func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

// writeJSON emits an indented JSON document for scripting.
func writeJSON(out io.Writer, value any) error {
	encoded, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Errorf("encode json: %w", err)
	}
	_, err = out.Write(append(encoded, '\n'))
	return err
}

// ParseActionFlags decides which scripted action a flag set requests.
//
// Order matters: the most specific flag wins, so a read can never silently
// mask an ask.
func ParseActionFlags(question string, healthFlag, modelsFlag, triadsFlag, standingsFlag, controlRoomFlag bool) (string, bool) {
	switch {
	case strings.TrimSpace(question) != "":
		return ActionAsk, true
	case healthFlag:
		return ActionHealth, true
	case modelsFlag:
		return ActionListModels, true
	case triadsFlag:
		return ActionListTriads, true
	case standingsFlag:
		return ActionStandings, true
	case controlRoomFlag:
		return ActionControlRoom, true
	}
	return "", false
}
