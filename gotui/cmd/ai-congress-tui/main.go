// Command ai-congress-tui is an additional way to run AI Congress: a Go
// front-end in the charm v2 + huh stack that talks to the same FastAPI service
// the Python CLI, the Svelte frontend and the Rust TUI already use.
//
// It never reimplements orchestration, voting or deliberation. Every answer
// comes from the service, so a Go user, a Rust user and a Python user get
// identical results.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"charm.land/bubbletea/v2"
	"github.com/jasperan/ai-congress/gotui/internal/api"
	"github.com/jasperan/ai-congress/gotui/internal/huhstyle"
	"github.com/jasperan/ai-congress/gotui/internal/session"
	"github.com/jasperan/ai-congress/gotui/internal/tui"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "ai-congress-tui: "+err.Error())
		os.Exit(1)
	}
}

// stringList collects a repeatable flag.
type stringList []string

func (l *stringList) String() string { return strings.Join(*l, ",") }

func (l *stringList) Set(value string) error {
	for _, part := range strings.Split(value, ",") {
		if trimmed := strings.TrimSpace(part); trimmed != "" {
			*l = append(*l, trimmed)
		}
	}
	return nil
}

func run() error {
	var (
		server      = flag.String("server", "", "AI Congress service URL (default from saved settings, then "+api.DefaultBaseURL+")")
		baseURL     = flag.String("base-url", "", "alias for --server")
		apiKey      = flag.String("api-key", "", "shared secret sent as X-API-Key (env: "+session.EnvAPIKey+")")
		port        = flag.Int("port", 0, "port for the service started with --start-service")
		startSvc    = flag.Bool("start-service", false, "start the repo's own API (python -m uvicorn src.ai_congress.api.main:app) before connecting")
		projectRoot = flag.String("project-root", "", "AI Congress checkout used to start the service (default: this binary's repo)")

		oracleUser    = flag.String("oracle-user", "", "Oracle user for the started service (env: "+session.EnvOracleUsername+")")
		oraclePass    = flag.String("oracle-password", "", "Oracle password for the started service (env: "+session.EnvOraclePassword+")")
		oracleHost    = flag.String("oracle-host", "", "Oracle host for the started service")
		oraclePort    = flag.String("oracle-port", "", "Oracle port for the started service")
		oracleService = flag.String("oracle-service", "", "Oracle service name for the started service")

		// Scripted actions.
		askFlag       = flag.String("ask", "", "ask one question, print the verdict, then exit")
		modeFlag      = flag.String("mode", "", "swarm mode for --ask: "+strings.Join(api.Modes, ", "))
		votingFlag    = flag.String("voting", "", "voting strategy for --ask: classic or semantic")
		backendFlag   = flag.String("backend", "", "inference backend for --ask: ollama, pi or openai")
		triadFlag     = flag.String("triad", "", "deliberation triad for --ask")
		evidenceFlag  = flag.Bool("evidence", false, "ground a deliberation --ask in web-search evidence")
		healthFlag    = flag.Bool("health", false, "print service health, then exit")
		modelsFlag    = flag.Bool("list-models", false, "list available models, then exit")
		triadsFlag    = flag.Bool("list-triads", false, "list deliberation triads, then exit")
		standingsFlag = flag.Bool("standings", false, "print the model leaderboard, then exit")
		controlFlag   = flag.Bool("control-room", false, "print the observability summary, then exit")
		jsonFlag      = flag.Bool("json", false, "emit machine-readable JSON for scripted actions")
		noInputFlag   = flag.Bool("no-input", false, "never prompt; fail instead if input is required")
	)
	var models stringList
	flag.Var(&models, "model", "council member for --ask (repeatable, or comma-separated)")
	flag.Parse()

	settings, err := session.Load()
	if err != nil {
		fmt.Fprintln(os.Stderr, "note: "+err.Error())
	}

	// --server wins over --base-url when both are given.
	chosenURL := *server
	if strings.TrimSpace(chosenURL) == "" {
		chosenURL = *baseURL
	}
	if strings.TrimSpace(chosenURL) != "" {
		if err := api.ValidateBaseURL(chosenURL); err != nil {
			return fmt.Errorf("--server: %w", err)
		}
		settings.BaseURL = strings.TrimRight(strings.TrimSpace(chosenURL), "/")
	}
	if *port != 0 {
		if err := session.ValidatePort(fmt.Sprint(*port)); err != nil {
			return fmt.Errorf("--port: %w", err)
		}
		settings.Port = *port
	}
	if *oracleUser != "" {
		settings.OracleUser = *oracleUser
	}
	if *oracleHost != "" {
		settings.OracleHost = *oracleHost
	}
	if *oraclePort != "" {
		settings.OraclePort = *oraclePort
	}
	if *oracleService != "" {
		settings.OracleService = *oracleService
	}
	if *startSvc {
		settings.LaunchServer = true
	}

	key := strings.TrimSpace(*apiKey)
	if key == "" {
		key = session.APIKeyFromEnv()
	}
	password := *oraclePass
	if password == "" {
		password = session.PasswordFromEnv()
	}

	root := *projectRoot
	if root == "" {
		root = defaultProjectRoot()
	}

	// --- scripted path -----------------------------------------------------------
	// This runs before any prompt is considered, so a pipeline never blocks on
	// a question.
	actionFlag, statusAction := tui.ParseActionFlags(*askFlag,
		*healthFlag, *modelsFlag, *triadsFlag, *standingsFlag, *controlFlag)
	if statusAction {
		ctx := context.Background()
		client := api.NewClient(settings.BaseURL, key)
		var serverProcess *session.Server
		if settings.LaunchServer {
			serverProcess, err = tui.StartServiceIfRequested(ctx, settings, password, key, root, os.Stdout)
			if err != nil {
				return err
			}
			defer serverProcess.Stop()
		}
		return tui.RunAction(ctx, client, tui.ActionRequest{
			Action:   actionFlag,
			Question: *askFlag,
			Mode:     *modeFlag,
			Voting:   *votingFlag,
			Backend:  *backendFlag,
			Triad:    *triadFlag,
			Models:   models,
			Evidence: *evidenceFlag,
			JSON:     *jsonFlag,
		}, os.Stdout)
	}
	// A council needs a question: a bare --triad or --model cannot mean "ask".
	if strings.TrimSpace(*triadFlag) != "" || len(models) > 0 {
		return errors.New("a council needs a question: add --ask \"your question\", or run the interactive TUI")
	}

	// --- screen-reader / piped path ---------------------------------------------
	// huh's accessible rendering only exists in its standalone Run path, so the
	// embedded full-screen UI is skipped entirely here.
	if huhstyle.Accessible() {
		fmt.Fprintln(os.Stdout, tui.AccessibleNotice)
		return runPlain(settings, key, root, password, *noInputFlag)
	}
	if !huhstyle.Interactive() || *noInputFlag {
		return errors.New("no terminal on stdin: pass an action flag such as --health, --list-models, " +
			"--list-triads, --standings, --control-room or --ask \"...\" (or set ACCESSIBLE for plain prompts)")
	}

	// --- full-screen path --------------------------------------------------------
	model := tui.New(tui.Options{ProjectRoot: root, Settings: settings, APIKey: key})
	defer model.Close()

	program := tea.NewProgram(model)
	if _, err := program.Run(); err != nil {
		return err
	}
	// A stream that failed after the screen closed still deserves reporting.
	if err := model.StreamError(); err != nil {
		return err
	}
	return nil
}

// runPlain drives the same forms as plain prompts.
func runPlain(settings session.Settings, key, root, password string, noInput bool) error {
	if noInput {
		return errors.New("-no-input cannot be combined with ACCESSIBLE plain prompts")
	}

	ctx := context.Background()

	// The council form's triad list is nice to have but not required; a failure
	// to fetch it must not stop an accessible user from asking anything.
	client := api.NewClient(settings.BaseURL, key)
	triads, err := client.Triads(ctx)
	if err != nil {
		fmt.Fprintln(os.Stderr, "note: could not list triads: "+err.Error())
	}

	chosen, request, passwordOut, keyOut, err := tui.RunPlainPrompts(tui.PlainOptions{
		ProjectRoot: root,
		Settings:    settings,
		APIKey:      key,
		Triads:      triads,
		Input:       os.Stdin,
		Output:      os.Stdout,
	})
	if err != nil {
		return err
	}
	if passwordOut == "" {
		passwordOut = password
	}
	if err := session.Save(chosen); err != nil {
		fmt.Fprintln(os.Stderr, "note: "+err.Error())
	}

	serverProcess, err := tui.StartServiceIfRequested(ctx, chosen, passwordOut, keyOut, root, os.Stdout)
	if err != nil {
		return err
	}
	if serverProcess != nil {
		defer serverProcess.Stop()
	}

	return tui.RunAction(ctx, api.NewClient(chosen.BaseURL, keyOut), tui.ActionRequest{
		Action:   tui.ActionAsk,
		Question: request.Prompt,
		Mode:     request.Mode,
		Voting:   request.VotingMode,
		Backend:  request.InferenceBackend,
		Triad:    request.Triad,
		Models:   request.Models,
	}, os.Stdout)
}

// defaultProjectRoot finds the checkout so the service can be started from it.
// The binary may be built anywhere, so this walks up from the executable and
// falls back to the working directory.
func defaultProjectRoot() string {
	if wd, err := os.Getwd(); err == nil {
		if looksLikeAICongress(wd) {
			return wd
		}
	}
	executable, err := os.Executable()
	if err == nil {
		dir := filepath.Dir(executable)
		for i := 0; i < 6; i++ {
			if looksLikeAICongress(dir) {
				return dir
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}
	wd, _ := os.Getwd()
	return wd
}

// looksLikeAICongress reports whether dir is the AI Congress checkout by
// looking for the two things starting the service needs.
func looksLikeAICongress(dir string) bool {
	if dir == "" {
		return false
	}
	if _, err := os.Stat(filepath.Join(dir, "run_server.py")); err != nil {
		return false
	}
	if _, err := os.Stat(filepath.Join(dir, "src", "ai_congress", "api", "main.py")); err != nil {
		return false
	}
	return true
}
