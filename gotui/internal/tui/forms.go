package tui

import (
	"errors"
	"strconv"
	"strings"

	"charm.land/huh/v2"
	"github.com/jasperan/ai-congress/gotui/internal/api"
	"github.com/jasperan/ai-congress/gotui/internal/huhstyle"
	"github.com/jasperan/ai-congress/gotui/internal/session"
)

// Action identifiers shared by the menu, the readouts and the scripted flags.
//
// There is one constant per destination so a screen and a flag cannot disagree
// about which action they mean.
const (
	// Menu destinations and readout actions.
	ActionCouncil     = "council"
	ActionStandings   = "standings"
	ActionControlRoom = "control_room"
	ActionListTriads  = "list_triads"
	ActionListModels  = "list_models"
	ActionReconnect   = "reconnect"
	ActionBack        = "back"
	ActionRefresh     = "refresh"
	ActionQuit        = "quit"

	// Scripted, non-interactive actions.
	ActionHealth = "health"
	ActionAsk    = "ask"
)

// ConnectAnswers are the connection form's values.
type ConnectAnswers struct {
	BaseURL        string
	APIKey         string
	LaunchServer   bool
	Port           string
	OracleHost     string
	OraclePort     string
	OracleService  string
	OracleUser     string
	OraclePassword string
}

// ConnectDefaults seeds the form from persisted, non-secret settings.
func ConnectDefaults(settings session.Settings) ConnectAnswers {
	return ConnectAnswers{
		BaseURL:       settings.BaseURL,
		LaunchServer:  settings.LaunchServer,
		Port:          strconv.Itoa(settings.Port),
		OracleHost:    settings.OracleHost,
		OraclePort:    settings.OraclePort,
		OracleService: settings.OracleService,
		OracleUser:    settings.OracleUser,
	}
}

// ToSettings converts the connection answers into persistable settings,
// filling in the documented defaults. The password and API key are
// deliberately absent: neither is ever written to disk.
func (a ConnectAnswers) ToSettings() session.Settings {
	settings := session.Settings{
		BaseURL:       strings.TrimSpace(a.BaseURL),
		LaunchServer:  a.LaunchServer,
		Port:          session.ParsePort(a.Port),
		OracleHost:    strings.TrimSpace(a.OracleHost),
		OraclePort:    strings.TrimSpace(a.OraclePort),
		OracleService: strings.TrimSpace(a.OracleService),
		OracleUser:    strings.TrimSpace(a.OracleUser),
	}
	defaults := session.Defaults()
	if settings.BaseURL == "" {
		settings.BaseURL = api.DefaultBaseURL
	}
	if settings.OracleHost == "" {
		settings.OracleHost = defaults.OracleHost
	}
	if settings.OraclePort == "" {
		settings.OraclePort = defaults.OraclePort
	}
	if settings.OracleService == "" {
		settings.OracleService = defaults.OracleService
	}
	if settings.OracleUser == "" {
		settings.OracleUser = defaults.OracleUser
	}
	return settings
}

// connectFields are the connection questions.
//
// They are separated from the form so the accessibility path can place them in
// a single form together with the council questions: huh's accessible Run
// buffers its reader, so two sequential forms cannot share stdin (the second
// one sees EOF and every answer comes back empty).
func connectFields(answers *ConnectAnswers, passwordFromEnv, keyFromEnv bool) []huh.Field {
	passwordNote := "Stored only in this process, passed to the service through the environment."
	if passwordFromEnv {
		passwordNote = "Already set in " + session.EnvOraclePassword + "; leave blank to use it."
	}
	keyNote := "Optional. Sent as X-API-Key; only needed when the service enables auth."
	if keyFromEnv {
		keyNote = "Already set in " + session.EnvAPIKey + "; leave blank to use it."
	}

	return []huh.Field{
		huh.NewInput().
			Title("Service URL").
			Description("Where the AI Congress FastAPI service is listening.").
			Placeholder(api.DefaultBaseURL).
			Value(&answers.BaseURL).
			Validate(ValidateDefaultedValue(answers.BaseURL, api.ValidateBaseURL)),

		huh.NewInput().
			Title("API key (optional)").
			Description(keyNote).
			EchoMode(huh.EchoModePassword).
			Value(&answers.APIKey),

		huh.NewConfirm().
			Title("Start a local service for me?").
			Description("Runs the repo's own API: python -m uvicorn src.ai_congress.api.main:app.").
			Value(&answers.LaunchServer),

		huh.NewInput().
			Title("Service port").
			Description("Used only when starting a local service.").
			Placeholder("8000").
			Value(&answers.Port).
			Validate(ValidateDefaultedValue(answers.Port, session.ValidatePort)),

		huh.NewInput().
			Title("Oracle host (optional)").
			Description("Sets " + session.EnvOracleHost + " for the started service.").
			Placeholder(session.DefaultOracleHost).
			Value(&answers.OracleHost),

		huh.NewInput().
			Title("Oracle port (optional)").
			Description("Sets " + session.EnvOraclePort + " for the started service.").
			Placeholder(session.DefaultOraclePort).
			Value(&answers.OraclePort),

		huh.NewInput().
			Title("Oracle service (optional)").
			Description("Sets " + session.EnvOracleService + " for the started service.").
			Placeholder(session.DefaultOracleSvc).
			Value(&answers.OracleService),

		huh.NewInput().
			Title("Oracle user (optional)").
			Description("Sets " + session.EnvOracleUsername + " for the started service.").
			Placeholder(session.DefaultOracleUser).
			Value(&answers.OracleUser),

		huh.NewInput().
			Title("Oracle password (optional)").
			Description(passwordNote).
			EchoMode(huh.EchoModePassword).
			Value(&answers.OraclePassword),
	}
}

// ConnectForm builds the connection form used by the full-screen UI.
//
// Every field lives in ONE group with no WithHideFunc: huh does not skip hidden
// groups in accessible mode, so a conditionally hidden required field would
// trap a screen-reader user on a question they cannot answer. The Oracle fields
// are all optional and port/URL carry valid defaults, so the form is always
// completable as rendered.
func ConnectForm(answers *ConnectAnswers, passwordFromEnv, keyFromEnv bool) *huh.Form {
	return huh.NewForm(
		huh.NewGroup(connectFields(answers, passwordFromEnv, keyFromEnv)...).Title("Connect"),
	).WithTheme(huh.ThemeFunc(huhstyle.Theme)).WithAccessible(huhstyle.Accessible())
}

// MenuActions are the top-level destinations.
func MenuActions() []huh.Option[string] {
	return []huh.Option[string]{
		huh.NewOption("Convene a council", ActionCouncil),
		huh.NewOption("Model standings", ActionStandings),
		huh.NewOption("Control room", ActionControlRoom),
		huh.NewOption("Deliberation triads", ActionListTriads),
		huh.NewOption("Available models", ActionListModels),
		huh.NewOption("Reconnect", ActionReconnect),
		huh.NewOption("Quit", ActionQuit),
	}
}

// MenuForm builds the main menu.
func MenuForm(choice *string, serviceURL string) *huh.Form {
	return huh.NewForm(
		huh.NewGroup(
			huh.NewSelect[string]().
				Title("What next?").
				Description("Connected to " + serviceURL + ".").
				Options(MenuActions()...).
				Value(choice),
		).Title("AI Congress"),
	).WithTheme(huh.ThemeFunc(huhstyle.Theme)).WithAccessible(huhstyle.Accessible())
}

// ModeOptions are the swarm modes, with the CLI's one-line explanation.
var ModeOptions = func() []huh.Option[string] {
	options := make([]huh.Option[string], 0, len(api.SelectableModes))
	for _, mode := range api.SelectableModes {
		options = append(options, huh.NewOption(mode+" - "+api.ModeDescription(mode), mode))
	}
	return options
}()

// VotingOptions are the voting strategies.
var VotingOptions = []huh.Option[string]{
	huh.NewOption("classic - weighted majority", api.VotingClassic),
	huh.NewOption("semantic - cluster by meaning", api.VotingSemantic),
}

// BackendOptions are the inference backends.
var BackendOptions = []huh.Option[string]{
	huh.NewOption("ollama - local", api.BackendOllama),
	huh.NewOption("pi - cloud (needs OPENCODE_GO_API_KEY)", api.BackendPi),
	huh.NewOption("openai - OpenAI-compatible", api.BackendOpenAI),
}

// ModelOptions builds the model multi-select list.
//
// A fresh slice is built per load: huh silently ignores an empty Options call,
// so a refresh that returned nothing would otherwise keep showing the previous
// models.
func ModelOptions(models []api.ModelInfo) []huh.Option[string] {
	if len(models) == 0 {
		return []huh.Option[string]{huh.NewOption("(the service reported no models)", "")}
	}
	options := make([]huh.Option[string], 0, len(models))
	for _, model := range models {
		label := model.Name + "  (" + model.Backend + ", weight " +
			strconv.FormatFloat(model.Weight, 'f', 3, 64) + ")"
		options = append(options, huh.NewOption(label, model.Name))
	}
	return options
}

// TriadOptions builds the triad select list.
func TriadOptions(triads []api.TriadInfo) []huh.Option[string] {
	options := []huh.Option[string]{huh.NewOption("none - use the models I choose", "")}
	for _, triad := range triads {
		label := triad.Name
		if triad.Description != "" {
			label = triad.Name + " - " + triad.Description
		}
		options = append(options, huh.NewOption(label, triad.Name))
	}
	return options
}

// CouncilAnswers are the council form's values.
type CouncilAnswers struct {
	Question    string
	Mode        string
	VotingMode  string
	Backend     string
	Triad       string
	Models      []string
	Temperature string
	UseRAG      bool
	SearchWeb   bool
	Evidence    bool
}

// CouncilDefaults mirrors the service's own defaults: multi_model mode,
// classic voting, Ollama, temperature 0.7 (schemas.py:ChatRequest).
func CouncilDefaults() CouncilAnswers {
	return CouncilAnswers{
		Mode:        api.ModeMultiModel,
		VotingMode:  api.VotingClassic,
		Backend:     api.BackendOllama,
		Temperature: "0.7",
	}
}

// ToRequest converts the form answers into POST /api/chat body.
//
// Deliberation is sent as triad + evidence: the service builds the council from
// the named triad, and passing a raw model list alongside it would silently
// override the triad's archetypes.
func (a CouncilAnswers) ToRequest() api.CouncilRequest {
	request := api.CouncilRequest{
		Prompt:           strings.TrimSpace(a.Question),
		Mode:             a.Mode,
		VotingMode:       a.VotingMode,
		InferenceBackend: a.Backend,
		Temperature:      ParseTemperature(a.Temperature),
	}
	if a.Mode == api.ModeDeliberation {
		request.Triad = strings.TrimSpace(a.Triad)
		evidence := a.Evidence
		request.Evidence = &evidence
		if request.Triad == "" {
			request.Models = a.Models
		}
		return request
	}
	if a.Mode != api.ModePersonality {
		request.Models = a.Models
	}
	request.UseRAG = a.UseRAG
	request.SearchWeb = a.SearchWeb
	return request
}

// councilFields are the council questions.
//
// They are ONE group with no WithHideFunc, for two reasons. First, huh v2.0.3
// only offers WithHideFunc on *Group, not on individual fields, so a
// per-field condition is not expressible. Second, hiding a group is not
// honoured in accessible mode, so a conditionally hidden required field would
// trap a screen-reader user on a question they cannot reach.
//
// The conditional questions (triad, evidence) are therefore always shown and
// always optional; the service ignores the ones that do not apply, and each
// Description says when it is used. Nothing here is required, so the form is
// completable in every mode as rendered.
func councilFields(answers *CouncilAnswers, models []api.ModelInfo, triads []api.TriadInfo) []huh.Field {
	return []huh.Field{
		huh.NewText().
			Title("Question").
			Description("What should the congress decide? Ctrl+J adds a line.").
			Placeholder("Should AI systems be regulated by federal law?").
			CharLimit(8000).
			Lines(4).
			Value(&answers.Question).
			Validate(huh.ValidateNotEmpty()),

		huh.NewSelect[string]().
			Title("Mode").
			Options(ModeOptions...).
			Value(&answers.Mode),

		huh.NewMultiSelect[string]().
			Title("Council members").
			Description("Used by multi_model, multi_request and hybrid. Ignored when a triad or personalities decide.").
			Options(ModelOptions(models)...).
			Height(10).
			Value(&answers.Models),

		huh.NewSelect[string]().
			Title("Triad").
			Description("Used by deliberation mode: three archetypes chosen to disagree. Leave as none to use the models above.").
			Options(TriadOptions(triads)...).
			Value(&answers.Triad),

		huh.NewSelect[string]().
			Title("Voting").
			Options(VotingOptions...).
			Value(&answers.VotingMode),

		huh.NewSelect[string]().
			Title("Inference backend").
			Options(BackendOptions...).
			Value(&answers.Backend),

		huh.NewInput().
			Title("Temperature").
			Description("0.0 to 2.0.").
			Placeholder("0.7").
			Value(&answers.Temperature).
			Validate(ValidateDefaultedValue(answers.Temperature, ValidateTemperature)),

		huh.NewConfirm().
			Title("Use RAG?").
			Description("Adds stored-document context to the prompt.").
			Value(&answers.UseRAG),

		huh.NewConfirm().
			Title("Search the web?").
			Description("Adds live web context to the prompt.").
			Value(&answers.SearchWeb),

		huh.NewConfirm().
			Title("Ground deliberation in evidence?").
			Description("Used by deliberation mode: adds a web-search evidence round.").
			Value(&answers.Evidence),
	}
}

// CouncilForm builds the council form used by the full-screen UI.
func CouncilForm(answers *CouncilAnswers, models []api.ModelInfo, triads []api.TriadInfo) *huh.Form {
	return huh.NewForm(
		huh.NewGroup(councilFields(answers, models, triads)...).Title("Convene"),
	).WithTheme(huh.ThemeFunc(huhstyle.Theme)).WithAccessible(huhstyle.Accessible())
}

// PlainAnswers holds both pages of the single accessible form.
type PlainAnswers struct {
	Connect ConnectAnswers
	Council CouncilAnswers
	// Members is the plain-path model list as typed text. The accessible form
	// cannot use a multi-select: its options come from the live service, and a
	// screen-reader user on a stopped service would face an empty list they
	// cannot answer. A comma-separated line has no such dependency.
	Members string
}

// plainCouncilFields are the council questions as the accessible form asks
// them.
//
// Two shapes differ from the interactive form on purpose:
//   - the question is a single-line Input, because huh's accessible textarea
//     prompt keeps reading until it is satisfied and then consumes lines
//     belonging to the next question;
//   - the council members are typed, not selected, for the reason on
//     PlainAnswers.Members.
func plainCouncilFields(answers *CouncilAnswers, members *string, triads []api.TriadInfo) []huh.Field {
	return []huh.Field{
		huh.NewInput().
			Title("Question").
			Description("What should the congress decide?").
			Placeholder("Should AI systems be regulated by federal law?").
			CharLimit(8000).
			Value(&answers.Question).
			Validate(huh.ValidateNotEmpty()),

		huh.NewSelect[string]().
			Title("Mode").
			Options(ModeOptions...).
			Value(&answers.Mode),

		huh.NewInput().
			Title("Council members").
			Description("Comma-separated model names. Used by the swarm modes; leave blank for personality or a triad.").
			Placeholder("qwen3.5:9b, gemma4:latest").
			Value(members),

		huh.NewSelect[string]().
			Title("Triad").
			Description("Used by deliberation mode. Leave as none to use the members above.").
			Options(TriadOptions(triads)...).
			Value(&answers.Triad),

		huh.NewSelect[string]().
			Title("Voting").
			Options(VotingOptions...).
			Value(&answers.VotingMode),

		huh.NewSelect[string]().
			Title("Inference backend").
			Options(BackendOptions...).
			Value(&answers.Backend),

		huh.NewInput().
			Title("Temperature").
			Description("0.0 to 2.0.").
			Placeholder("0.7").
			Value(&answers.Temperature).
			Validate(ValidateDefaultedValue(answers.Temperature, ValidateTemperature)),

		huh.NewConfirm().
			Title("Use RAG?").
			Description("Adds stored-document context to the prompt.").
			Value(&answers.UseRAG),

		huh.NewConfirm().
			Title("Search the web?").
			Description("Adds live web context to the prompt.").
			Value(&answers.SearchWeb),

		huh.NewConfirm().
			Title("Ground deliberation in evidence?").
			Description("Used by deliberation mode.").
			Value(&answers.Evidence),
	}
}

// PlainForm is the one-pass form used when ACCESSIBLE is set or stdin is not a
// terminal.
//
// It is deliberately a SINGLE group, not two. In accessible mode huh runs only
// the first group of a form and then reports the form complete, so a second
// group's questions would be silently left at their defaults. One group means
// every question is actually asked, exactly once, one line each.
func PlainForm(answers *PlainAnswers, triads []api.TriadInfo, passwordFromEnv, keyFromEnv bool) *huh.Form {
	fields := connectFields(&answers.Connect, passwordFromEnv, keyFromEnv)
	fields = append(fields, plainCouncilFields(&answers.Council, &answers.Members, triads)...)

	return huh.NewForm(
		huh.NewGroup(fields...).Title("Connect and convene"),
	).WithTheme(huh.ThemeFunc(huhstyle.Theme)).WithAccessible(huhstyle.Accessible())
}

// ParseMembers splits the typed council-member line on commas, dropping blanks.
func ParseMembers(raw string) []string {
	var members []string
	for _, part := range strings.Split(raw, ",") {
		trimmed := strings.TrimSpace(part)
		if trimmed != "" {
			members = append(members, trimmed)
		}
	}
	return members
}

// ResultAnswers is the post-run continue prompt.
type ResultAnswers struct{ Again bool }

// ResultForm builds the "convene another?" prompt.
func ResultForm(answers *ResultAnswers) *huh.Form {
	return huh.NewForm(
		huh.NewGroup(
			huh.NewConfirm().
				Title("Convene another council?").
				Affirmative("Yes").
				Negative("Back to the menu").
				Value(&answers.Again),
		),
	).WithTheme(huh.ThemeFunc(huhstyle.Theme)).WithAccessible(huhstyle.Accessible())
}

// ReadoutAnswers is a read-only screen's action choice.
//
// The readouts need a form to hand control back: with no form the root model
// would have nothing to continue from.
type ReadoutAnswers struct{ Action string }

// ReadoutForm builds a readout's continue prompt.
func ReadoutForm(answers *ReadoutAnswers, title string) *huh.Form {
	return huh.NewForm(
		huh.NewGroup(
			huh.NewSelect[string]().
				Title("What next?").
				Options(
					huh.NewOption("Refresh", ActionRefresh),
					huh.NewOption("Back to the menu", ActionBack),
				).
				Value(&answers.Action),
		).Title(title),
	).WithTheme(huh.ThemeFunc(huhstyle.Theme)).WithAccessible(huhstyle.Accessible())
}

// ValidateDefaulted accepts an empty answer as "keep the value already in the
// field".
//
// It exists for accessible mode. huh's screen-reader path runs a field's
// validator on the raw line and only afterwards substitutes the field's
// default, and it never prints that default. A pre-filled field whose validator
// rejects "" therefore re-prompts on every bare Enter, so a screen-reader user
// cannot accept a value they cannot see.
func ValidateDefaulted(inner func(string) error) func(string) error {
	return func(s string) error {
		if strings.TrimSpace(s) == "" {
			return nil
		}
		return inner(s)
	}
}

// ValidateDefaultedValue is ValidateDefaulted for a field whose pre-filled
// value may itself be empty (the connect defaults pass through whatever was
// persisted). Blank stays invalid when there is nothing to keep.
func ValidateDefaultedValue(prefilled string, inner func(string) error) func(string) error {
	if strings.TrimSpace(prefilled) == "" {
		return inner
	}
	return ValidateDefaulted(inner)
}

// ValidateTemperature enforces the service's accepted range so a bad value is
// caught in the form rather than as an HTTP 422.
func ValidateTemperature(raw string) error {
	value, err := strconv.ParseFloat(strings.TrimSpace(raw), 64)
	if err != nil {
		return errors.New("use a number between 0.0 and 2.0")
	}
	if value < 0 || value > 2 {
		return errors.New("use a number between 0.0 and 2.0")
	}
	return nil
}

// ParseTemperature converts a validated temperature string.
func ParseTemperature(raw string) float64 {
	value, err := strconv.ParseFloat(strings.TrimSpace(raw), 64)
	if err != nil || value < 0 || value > 2 {
		return 0.7
	}
	return value
}
