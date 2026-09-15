// Package tui is the Go front-end's terminal UI: a peer client of the AI
// Congress FastAPI service, built on charm.land/bubbletea/v2 and
// charm.land/huh/v2.
package tui

import (
	"context"
	"errors"
	"image/color"
	"sort"
	"strconv"
	"strings"
	"time"

	"charm.land/bubbles/v2/progress"
	"charm.land/bubbles/v2/spinner"
	"charm.land/bubbletea/v2"
	"charm.land/huh/v2"
	"github.com/jasperan/ai-congress/gotui/internal/api"
	"github.com/jasperan/ai-congress/gotui/internal/session"
)

// requestTimeout bounds every blocking REST call so a hung service shows an
// error state instead of freezing the UI.
//
// It is deliberately generous: a council run is one long request, and the
// deliberation protocol runs three rounds of several models.
const requestTimeout = 30 * time.Minute

// Screen identifies the active view.
type Screen int

// Screens.
const (
	ScreenConnect Screen = iota
	ScreenMenu
	ScreenCouncil
	ScreenRunning
	ScreenResult
	ScreenStandings
	ScreenControlRoom
	ScreenTriads
	ScreenModels
)

// Options configure a TUI run.
type Options struct {
	ProjectRoot string
	Settings    session.Settings
	APIKey      string
}

// Model is the root bubbletea model.
type Model struct {
	width  int
	height int
	opts   Options

	screen Screen
	client *api.Client
	form   *huh.Form

	connect ConnectAnswers
	council CouncilAnswers
	result  ResultAnswers
	readout ReadoutAnswers
	menu    string

	models      []api.ModelInfo
	triads      []api.TriadInfo
	leaderboard []api.LeaderboardRow
	summary     map[string]any

	// councilState is the live run, folded from the stream frames.
	councilState *Council
	stream       *api.Stream

	spinner  spinner.Model
	progress progress.Model

	// server is non-nil when this front-end started the Python service, so it
	// can stop it again on exit.
	server *session.Server
	// password and apiKey live in memory only. Neither is written to disk and
	// neither is passed as a command-line argument.
	password string
	apiKey   string

	busy    string
	failure error
	notice  string

	// streamFailure is the last stream error, kept after the alternate screen is
	// restored so the command can print it to a terminal the user can still
	// read. m.failure is cleared on every screen transition.
	streamFailure error
}

// New builds the root model and shows the connection form.
func New(opts Options) *Model {
	apiKey := strings.TrimSpace(opts.APIKey)
	if apiKey == "" {
		apiKey = session.APIKeyFromEnv()
	}

	m := &Model{
		opts:    opts,
		width:   100,
		height:  32,
		client:  api.NewClient(opts.Settings.BaseURL, apiKey),
		connect: ConnectDefaults(opts.Settings),
		council: CouncilDefaults(),
		screen:  ScreenConnect,
		spinner: spinner.New(),
		progress: progress.New(
			progress.WithColors(progressColor()),
			progress.WithoutPercentage(),
		),
		apiKey: apiKey,
	}
	m.form = ConnectForm(&m.connect, session.PasswordFromEnv() != "", apiKey != "")
	m.resize()
	return m
}

// progressColor reads the accent out of the shared theme so the bar cannot
// drift from the palette.
func progressColor() color.Color {
	_, _, _, _, _, _, warning := palette()
	return warning.GetForeground()
}

// setForm installs a freshly built form, sizes it, and returns its first
// command. Every screen transition goes through here so no form can ever be
// left at huh's default zero width (which renders as blank lines).
func (m *Model) setForm(form *huh.Form) tea.Cmd {
	m.form = form
	m.resize()
	return m.form.Init()
}

// resize applies the current window size to the form and the progress bar.
//
// huh.NewForm leaves the form at width 0 until something sets it, and a
// zero-width field renders as blank lines. Calling this from New means the form
// is visible even before bubbletea delivers its first WindowSizeMsg.
//
// The width is clamped to a positive floor first: bubbletea can report a width
// of 0, and a negative width panics inside lipgloss.
func (m *Model) resize() {
	width := m.width
	if width < MinPaneWidth {
		width = MinPaneWidth
	}
	if m.form != nil {
		m.form = m.form.WithWidth(width - 4)
	}
	m.progress.SetWidth(width - 8)
}

// Init implements tea.Model.
func (m *Model) Init() tea.Cmd { return m.form.Init() }

// Close stops any service this front-end started, and any stream it opened.
func (m *Model) Close() {
	if m.stream != nil {
		m.stream.Close()
		m.stream = nil
	}
	if m.server != nil {
		m.server.Stop()
		m.server = nil
	}
}

// StreamError reports the last stream failure, if any.
//
// It exists for the command: a failure that ended a run is also shown in the
// result screen, but that screen is torn down with the alternate screen, so a
// non-interactive caller needs a way to surface it after the program exits.
func (m *Model) StreamError() error { return m.streamFailure }

// --- async results -----------------------------------------------------------------

type modelsLoadedMsg struct {
	models []api.ModelInfo
	err    error
}

type triadsLoadedMsg struct {
	triads []api.TriadInfo
	err    error
}

type standingsLoadedMsg struct {
	rows []api.LeaderboardRow
	err  error
}

type controlRoomLoadedMsg struct {
	summary map[string]any
	err     error
}

type streamEventMsg struct{ event api.StreamEvent }

type streamClosedMsg struct{ err error }

// --- commands ----------------------------------------------------------------------

func loadModelsCmd(client *api.Client) tea.Cmd {
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
		defer cancel()
		models, err := client.Models(ctx)
		return modelsLoadedMsg{models: models, err: err}
	}
}

func loadTriadsCmd(client *api.Client) tea.Cmd {
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()
		triads, err := client.Triads(ctx)
		return triadsLoadedMsg{triads: triads, err: err}
	}
}

func loadStandingsCmd(client *api.Client) tea.Cmd {
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), requestTimeout)
		defer cancel()
		rows, err := client.Leaderboard(ctx)
		return standingsLoadedMsg{rows: rows, err: err}
	}
}

func loadControlRoomCmd(client *api.Client) tea.Cmd {
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), requestTimeout)
		defer cancel()
		summary, err := client.ObservabilitySummary(ctx)
		return controlRoomLoadedMsg{summary: summary, err: err}
	}
}

// startStreamCmd opens /ws/chat and hands back the first frame.
//
// The returned stream is captured by the caller through the model, not through
// this message: a tea.Cmd cannot mutate the model, so the stream is created in
// the Update goroutine and only its frames travel through messages.
func waitForEventCmd(stream *api.Stream) tea.Cmd {
	return func() tea.Msg {
		event, ok := <-stream.Events()
		if !ok {
			return streamClosedMsg{err: stream.Err()}
		}
		return streamEventMsg{event: event}
	}
}

// --- update ------------------------------------------------------------------------

// Update implements tea.Model.
func (m *Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.resize()
		return m, nil

	case spinner.TickMsg:
		// Only re-arm the spinner while something is actually running, so a
		// finished screen stops scheduling frames.
		if m.busy == "" && m.screen != ScreenRunning {
			return m, nil
		}
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		return m, cmd

	case modelsLoadedMsg:
		m.busy = ""
		m.failure = explain(msg.err)
		if msg.err == nil {
			m.models = msg.models
		}
		return m, nil

	case triadsLoadedMsg:
		m.busy = ""
		m.failure = explain(msg.err)
		if msg.err == nil {
			m.triads = msg.triads
		}
		return m, nil

	case standingsLoadedMsg:
		m.busy = ""
		m.failure = explain(msg.err)
		if msg.err == nil {
			m.leaderboard = msg.rows
		}
		m.readout = ReadoutAnswers{}
		return m, m.setForm(ReadoutForm(&m.readout, "Standings"))

	case controlRoomLoadedMsg:
		m.busy = ""
		m.failure = explain(msg.err)
		if msg.err == nil {
			m.summary = msg.summary
		}
		m.readout = ReadoutAnswers{}
		return m, m.setForm(ReadoutForm(&m.readout, "Control room"))

	case streamEventMsg:
		if m.councilState != nil {
			m.councilState.Apply(msg.event)
		}
		if m.stream != nil {
			return m, waitForEventCmd(m.stream)
		}
		return m, nil

	case streamClosedMsg:
		m.busy = ""
		// A cancelled stream is a user action, not a failure to report.
		if msg.err != nil && !errors.Is(msg.err, context.Canceled) &&
			!errors.Is(msg.err, context.DeadlineExceeded) {
			m.failure = msg.err
			m.streamFailure = msg.err
		}
		if m.stream != nil {
			m.stream.Close()
			m.stream = nil
		}
		m.screen = ScreenResult
		m.result = ResultAnswers{}
		if m.councilState != nil {
			// Fill in any member that never reported, so the transcript does
			// not leave a spinner running forever.
			m.councilState.Done = true
			m.councilState.Apply(api.StreamEvent{Type: "end"})
		}
		return m, m.setForm(ResultForm(&m.result))

	case tea.KeyPressMsg:
		// Only key PRESSES are handled. bubbletea v2 also delivers key
		// releases, and acting on both would fire every binding twice.
		switch msg.String() {
		case "ctrl+c":
			m.Close()
			return m, tea.Quit
		case "esc":
			// esc cancels the open form and returns to the menu.
			//
			// It must be intercepted here rather than delegated: huh's default
			// keymap binds Quit to ctrl+c only, so escape never reaches huh's
			// abort path and the form would stay installed. Because no key is
			// handled while a form is open (see the bail below), an orphaned form
			// then swallows every later keystroke and the user has no way back.
			//
			// The connect screen is excluded on purpose. It is the entry point,
			// and the menu it would cancel to cannot do anything until a
			// connection has been configured, so cancelling there would strand
			// the user somewhere useless rather than somewhere they can act.
			// toMenu installs the menu's own form, so input keeps working.
			if m.form != nil && m.screen != ScreenConnect {
				m.notice = "Cancelled."
				return m, m.toMenu()
			}
		}
		m.notice = ""
	}

	if m.form == nil {
		return m, nil
	}

	updated, cmd := m.form.Update(msg)
	if form, ok := updated.(*huh.Form); ok {
		m.form = form
	}
	if m.form.State != huh.StateNormal {
		return m, m.advance()
	}
	return m, cmd
}

// advance reacts to a finished form. The screen decides what the answers mean.
func (m *Model) advance() tea.Cmd {
	state := m.form.State
	m.form = nil
	m.failure = nil

	if state == huh.StateAborted {
		// Not reachable through escape. huh sets StateAborted only from
		// keymap.Quit, which is bound to ctrl+c, and Update handles ctrl+c
		// before the form is ever updated. Kept so that a form aborted by any
		// future path still stops the service it started instead of leaving it
		// running behind a quit.
		m.Close()
		return tea.Quit
	}

	switch m.screen {
	case ScreenConnect:
		return m.finishConnect()

	case ScreenMenu:
		return m.finishMenu()

	case ScreenCouncil:
		return m.finishCouncil()

	case ScreenResult:
		if m.result.Again {
			return m.toCouncil()
		}
		return m.toMenu()

	case ScreenStandings:
		if m.readout.Action == ActionRefresh {
			m.busy = "Refreshing standings"
			return loadStandingsCmd(m.client)
		}
		return m.toMenu()

	case ScreenControlRoom:
		if m.readout.Action == ActionRefresh {
			m.busy = "Refreshing the control room"
			return loadControlRoomCmd(m.client)
		}
		return m.toMenu()

	case ScreenTriads:
		if m.readout.Action == ActionRefresh {
			m.busy = "Refreshing triads"
			return loadTriadsCmd(m.client)
		}
		return m.toMenu()

	case ScreenModels:
		if m.readout.Action == ActionRefresh {
			m.busy = "Refreshing models"
			return loadModelsCmd(m.client)
		}
		return m.toMenu()

	case ScreenRunning:
		// The run is driven by the stream, not by a form; a stray submit here
		// means the user confirmed the abort prompt.
		m.Close()
		return tea.Quit
	}
	return nil
}

// finishMenu routes the top-level menu selection.
func (m *Model) finishMenu() tea.Cmd {
	switch m.menu {
	case ActionCouncil:
		return m.toCouncil()

	case ActionStandings:
		m.busy = "Loading standings"
		m.screen = ScreenStandings
		return tea.Batch(loadStandingsCmd(m.client), m.spinner.Tick)

	case ActionControlRoom:
		m.busy = "Loading the control room"
		m.screen = ScreenControlRoom
		return tea.Batch(loadControlRoomCmd(m.client), m.spinner.Tick)

	case ActionListTriads:
		m.busy = "Loading triads"
		m.screen = ScreenTriads
		return tea.Batch(loadTriadsCmd(m.client), m.spinner.Tick)

	case ActionListModels:
		m.busy = "Loading models"
		m.screen = ScreenModels
		return tea.Batch(loadModelsCmd(m.client), m.spinner.Tick)

	case ActionReconnect:
		m.connect = ConnectDefaults(m.persistedSettings())
		m.screen = ScreenConnect
		return m.setForm(ConnectForm(&m.connect, session.PasswordFromEnv() != "", m.apiKey != ""))

	default:
		m.Close()
		return tea.Quit
	}
}

// finishConnect applies the connection answers and moves to the menu.
func (m *Model) finishConnect() tea.Cmd {
	settings := m.connect.ToSettings()

	// A blank secret field means "use whatever is already in the environment",
	// which is how a scripted user avoids typing it at all.
	m.password = m.connect.OraclePassword
	if m.password == "" {
		m.password = session.PasswordFromEnv()
	}
	m.apiKey = strings.TrimSpace(m.connect.APIKey)
	if m.apiKey == "" {
		m.apiKey = session.APIKeyFromEnv()
	}

	m.opts.Settings = settings
	m.client = api.NewClient(settings.BaseURL, m.apiKey)

	// Only non-secret settings are persisted, and a save failure is not fatal.
	if err := session.Save(settings); err != nil {
		m.notice = "Could not save settings: " + err.Error()
	}

	cmds := []tea.Cmd{m.spinner.Tick}

	if settings.LaunchServer {
		m.busy = "Starting the service on port " + strconv.Itoa(settings.Port)
		server, err := session.LaunchServer(context.Background(), m.opts.ProjectRoot, settings.Port,
			session.ServerEnv(settings, m.password, m.apiKey))
		if err != nil {
			m.busy = ""
			m.failure = err
			return m.toMenu()
		}
		m.server = server
		if err := session.WaitForPort(context.Background(), "127.0.0.1", settings.Port, 90*time.Second); err != nil {
			m.failure = err
		} else {
			m.notice = "Service started on port " + strconv.Itoa(settings.Port)
		}
		m.busy = ""
	}

	// The council form needs the live model and triad lists, so they are
	// fetched before the user can convene anything.
	cmds = append(cmds, loadModelsCmd(m.client), loadTriadsCmd(m.client))
	cmds = append(cmds, m.toMenu())
	return tea.Batch(cmds...)
}

// toCouncil builds the council form from the live model and triad lists.
func (m *Model) toCouncil() tea.Cmd {
	m.screen = ScreenCouncil
	m.council = CouncilDefaults()
	return m.setForm(CouncilForm(&m.council, m.models, m.triads))
}

// finishCouncil validates the answers and opens the stream.
func (m *Model) finishCouncil() tea.Cmd {
	request := m.council.ToRequest()
	if err := request.Validate(); err != nil {
		// The form itself cannot enforce a cross-field rule (see
		// councilFields), so the failure is reported here and the user is sent
		// straight back to the answers rather than into a broken run.
		m.failure = err
		m.screen = ScreenCouncil
		return m.setForm(CouncilForm(&m.council, m.models, m.triads))
	}

	members := request.Models
	if request.Mode == api.ModeDeliberation && request.Triad != "" {
		members = []string{request.Triad + " (triad)"}
	}
	m.councilState = NewCouncil(request.Prompt, request.Mode, members)

	stream, err := api.StreamCouncil(context.Background(), m.client.WSURL(api.ChatPath), m.apiKey, request)
	if err != nil {
		m.failure = explain(err)
		m.screen = ScreenCouncil
		return m.setForm(CouncilForm(&m.council, m.models, m.triads))
	}

	m.stream = stream
	m.screen = ScreenRunning
	m.busy = "Convening the council"
	m.failure = nil
	return tea.Batch(waitForEventCmd(stream), m.spinner.Tick)
}

func (m *Model) toMenu() tea.Cmd {
	m.screen = ScreenMenu
	m.menu = ""
	return m.setForm(MenuForm(&m.menu, m.client.BaseURL()))
}

func (m *Model) persistedSettings() session.Settings {
	settings := m.opts.Settings
	if settings.BaseURL == "" {
		settings = session.Defaults()
	}
	return settings
}

// explain turns a transport failure into something actionable. A stopped
// service is the common case here, not an exceptional one.
func explain(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, api.ErrUnreachable) {
		return errors.New(err.Error() + " - start the service (python run_server.py), then choose Reconnect")
	}
	return err
}

// --- view --------------------------------------------------------------------------

// View implements tea.Model.
func (m *Model) View() tea.View {
	var body strings.Builder

	body.WriteString(Header(m.client.BaseURL(), m.council.Mode, m.width))
	body.WriteString("\n\n")

	if m.busy != "" {
		body.WriteString(Pane("Working", m.spinner.View()+" "+m.busy, m.width))
		body.WriteString("\n\n")
	}
	if m.failure != nil {
		body.WriteString(PaneError("Problem", m.failure, m.width))
		body.WriteString("\n\n")
	}
	if m.notice != "" {
		body.WriteString(Pane("Done", m.notice, m.width))
		body.WriteString("\n\n")
	}

	switch m.screen {
	case ScreenRunning:
		body.WriteString(m.viewRunning())
	case ScreenResult:
		body.WriteString(m.viewResult())
	case ScreenStandings:
		body.WriteString(Pane("Model standings", LeaderboardRows(m.leaderboard, m.width), m.width))
		body.WriteString("\n\n")
	case ScreenControlRoom:
		body.WriteString(Pane("Control room", m.viewControlRoom(), m.width))
		body.WriteString("\n\n")
	case ScreenTriads:
		body.WriteString(Pane("Deliberation triads", TriadRows(m.triads, m.width), m.width))
		body.WriteString("\n\n")
	case ScreenModels:
		body.WriteString(Pane("Available models", ModelRows(m.models, m.width), m.width))
		body.WriteString("\n\n")
	}

	if m.form != nil {
		body.WriteString(m.form.View())
		body.WriteString("\n\n")
	}

	body.WriteString(Footer(m.hint(), m.width))

	view := tea.NewView(body.String())
	view.AltScreen = true
	return view
}

// viewRunning is the live council screen: who is speaking, and how far along.
func (m *Model) viewRunning() string {
	council := m.councilState
	if council == nil {
		return ""
	}

	var out strings.Builder
	out.WriteString(Pane("Council",
		Stat("mode", council.Mode)+"\n"+
			Stat("question", Truncate(oneLine(council.Question), contentWidth(m.width)-12)), m.width))
	out.WriteString("\n\n")

	bar := m.progress.ViewAs(council.ProgressFraction())
	out.WriteString(Pane("Progress",
		strconv.Itoa(council.Completed())+" of "+strconv.Itoa(len(council.Members))+" responded\n"+bar,
		m.width))
	out.WriteString("\n\n")

	out.WriteString(Pane("Members", MemberRows(council, m.width), m.width))
	out.WriteString("\n\n")

	if len(council.VoteBreakdown) > 0 || council.Semantic != nil {
		out.WriteString(Pane("Vote tally", TallyRows(council, m.width), m.width))
		out.WriteString("\n\n")
	}
	if len(council.Rounds) > 0 {
		out.WriteString(Pane("Rounds", RoundRows(council, m.width), m.width))
		out.WriteString("\n\n")
	}
	return out.String()
}

// viewResult is the finished council: verdict, tally, rounds and transcript.
func (m *Model) viewResult() string {
	council := m.councilState
	if council == nil {
		return ""
	}

	var out strings.Builder
	out.WriteString(Pane("Verdict", VerdictRows(council, m.width), m.width))
	out.WriteString("\n\n")

	if council.FinalAnswer != "" {
		out.WriteString(Pane("Winning answer", wrapLines(council.FinalAnswer, contentWidth(m.width)), m.width))
		out.WriteString("\n\n")
	}

	out.WriteString(Pane("Vote tally", TallyRows(council, m.width), m.width))
	out.WriteString("\n\n")

	if semantic := SemanticRows(council, m.width); semantic != "" {
		out.WriteString(Pane("Semantic vote", semantic, m.width))
		out.WriteString("\n\n")
	}

	if len(council.Rounds) > 0 {
		out.WriteString(Pane("Rounds", RoundRows(council, m.width), m.width))
		out.WriteString("\n\n")
	}

	out.WriteString(Pane("Transcript", TranscriptRows(council, m.width), m.width))
	out.WriteString("\n\n")

	if council.Error != "" {
		out.WriteString(Pane("Service reported", council.Error, m.width))
		out.WriteString("\n\n")
	}
	return out.String()
}

// viewControlRoom renders the observability summary the service returns.
//
// The summary is an untyped map, so only the sections this UI understands are
// formatted; anything else is listed as a count rather than guessed at.
func (m *Model) viewControlRoom() string {
	if len(m.summary) == 0 {
		return Fog("The service returned no observability data.")
	}

	var out strings.Builder

	if open, ok := number(m.summary["breaker_open_count"]); ok {
		line := "0"
		if open > 0 {
			line = strconv.FormatFloat(open, 'f', 0, 64)
		}
		out.WriteString(Stat("open circuit breakers", line))
		out.WriteString("\n")
	}

	if breakers, ok := m.summary["circuit_breakers"].(map[string]any); ok && len(breakers) > 0 {
		out.WriteString("\n")
		out.WriteString(Stat("tracked breakers", strconv.Itoa(len(breakers))))
		out.WriteString("\n")
		names := make([]string, 0, len(breakers))
		for name := range breakers {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			state := ""
			if entry, ok := breakers[name].(map[string]any); ok {
				if s, ok := entry["state"].(string); ok {
					state = s
				}
			}
			style := ""
			if state == "OPEN" {
				style = Caution("OPEN")
				out.WriteString("  " + Pad(Truncate(name, 26), 26) + " " + style + "\n")
				continue
			}
			out.WriteString("  " + Pad(Truncate(name, 26), 26) + " " + state + "\n")
		}
	}

	if runs, ok := m.summary["recent_runs"].([]any); ok && len(runs) > 0 {
		out.WriteString("\n")
		out.WriteString(Stat("recent runs", strconv.Itoa(len(runs))))
		out.WriteString("\n")
		for _, raw := range runs {
			run, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			query, _ := run["query"].(string)
			status, _ := run["status"].(string)
			out.WriteString("  " + Pad(Truncate(oneLine(query), 34), 34) + " " + status + "\n")
		}
	}

	if domains, ok := m.summary["domain_win_rates"].([]any); ok && len(domains) > 0 {
		out.WriteString("\n")
		out.WriteString(Stat("domains tracked", strconv.Itoa(len(domains))))
		out.WriteString("\n")
		for _, raw := range domains {
			domain, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			name, _ := domain["domain"].(string)
			rate, _ := number(domain["win_rate"])
			count, _ := number(domain["feedback_count"])
			out.WriteString("  " + Pad(Truncate(name, 22), 22) + " " + Pad(Percent(rate), 5) + " " +
				strconv.FormatFloat(count, 'f', 0, 64) + " feedback\n")
		}
	}

	if out.Len() == 0 {
		out.WriteString(Fog("The control room is empty: no runs or feedback have been recorded yet."))
	}
	return out.String()
}

// number reads a JSON number, which decodes to float64, and tolerates the
// integer form too.
func number(value any) (float64, bool) {
	switch typed := value.(type) {
	case float64:
		return typed, true
	case int:
		return float64(typed), true
	case int64:
		return float64(typed), true
	default:
		return 0, false
	}
}

func (m *Model) hint() string {
	switch m.screen {
	case ScreenConnect:
		return "tab next - shift+tab back - enter submit - ctrl+c quit"
	case ScreenRunning:
		return "the council is running - ctrl+c aborts and stops any service this front-end started"
	case ScreenResult:
		return "answer the prompt below to continue - ctrl+c quit"
	case ScreenStandings, ScreenControlRoom, ScreenTriads, ScreenModels:
		return "choose an action below - ctrl+c quit"
	default:
		return "arrows move - / filters - space selects - enter confirms - ctrl+c quits"
	}
}

// AccessibleNotice explains the mode switch a screen-reader user gets.
const AccessibleNotice = "ACCESSIBLE is set: using plain prompts instead of the full-screen UI."

// compile-time guard that the root model satisfies tea.Model.
var _ tea.Model = (*Model)(nil)
