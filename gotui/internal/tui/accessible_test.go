package tui

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	"charm.land/huh/v2"
	"github.com/jasperan/ai-congress/gotui/internal/api"
	"github.com/jasperan/ai-congress/gotui/internal/session"
)

// This file covers the accessible (screen-reader) path.
//
// Why it exists: huh's accessible prompts run a field's validator on the raw
// line and only afterwards substitute the field's default, and they never print
// that default. A pre-filled field whose validator rejects "" therefore
// re-prompts on every bare Enter, so a screen-reader user cannot accept a value
// they cannot see.
//
// These tests drive the module's own field builders through huh's real
// accessible entry point (huh.Field.RunAccessible) rather than re-declaring the
// fields, so they pin the actual wiring. Per-field driving is used instead of
// Form.Run because each accessible prompt builds its own buffered scanner over
// the reader, so only the first prompt of a multi-field form can be fed a real
// answer.

var errBlank = errors.New("blank answer rejected")

// runFieldAccessible drives one real field, built by the module's own field
// builder, through huh's accessible path.
func runFieldAccessible(t *testing.T, f huh.Field, input string) string {
	t.Helper()
	var out bytes.Buffer
	if err := f.RunAccessible(&out, strings.NewReader(input)); err != nil {
		// A field whose echo mode needs a tty reports here; the form ignores it too.
		t.Logf("RunAccessible returned %v (ignored, as huh.Form does)", err)
	}
	return out.String()
}

func TestValidateDefaultedAcceptsBlank(t *testing.T) {
	inner := func(s string) error {
		if strings.TrimSpace(s) == "" {
			return errBlank
		}
		if s == "bad" {
			return errors.New("not usable")
		}
		return nil
	}
	wrapped := ValidateDefaulted(inner)

	for _, in := range []string{"", "   ", "\t"} {
		if err := wrapped(in); err != nil {
			t.Errorf("ValidateDefaulted(inner)(%q) = %v, want nil", in, err)
		}
	}
	if err := wrapped("bad"); err == nil {
		t.Error(`ValidateDefaulted(inner)("bad") = nil, want the inner validator's error`)
	}
	if err := wrapped("0.7"); err != nil {
		t.Errorf(`ValidateDefaulted(inner)("0.7") = %v, want nil`, err)
	}
}

// TestValidateDefaultedValueOnlyRelaxesWhenSomethingIsKept is the guard:
// ConnectDefaults passes through whatever was persisted, so a field may have
// nothing to keep.
func TestValidateDefaultedValueOnlyRelaxesWhenSomethingIsKept(t *testing.T) {
	inner := func(s string) error {
		if strings.TrimSpace(s) == "" {
			return errBlank
		}
		return nil
	}

	if err := ValidateDefaultedValue("", inner)(""); err == nil {
		t.Error("empty pre-fill accepted a blank answer; a required field was weakened")
	}
	if err := ValidateDefaultedValue("   ", inner)(""); err == nil {
		t.Error("whitespace-only pre-fill accepted a blank answer")
	}
	if err := ValidateDefaultedValue("http://127.0.0.1:8000", inner)(""); err != nil {
		t.Errorf("pre-filled field rejected blank: %v", err)
	}
}

// TestAccessibleBlankAnswerKeepsThePrefilledValue drives the real connect and
// council fields.
func TestAccessibleBlankAnswerKeepsThePrefilledValue(t *testing.T) {
	t.Run("seeded service url", func(t *testing.T) {
		answers := ConnectDefaults(session.Defaults())
		fields := connectFields(&answers, false, false)
		seeded := answers.BaseURL
		if seeded == "" {
			t.Fatal("service URL is not pre-filled; the test would prove nothing")
		}

		out := runFieldAccessible(t, fields[0], "\n")
		if strings.Contains(out, "a service URL is required") {
			t.Errorf("a blank answer was rejected, so a screen-reader user cannot keep the "+
				"pre-filled service URL.\noutput:\n%s", out)
		}
		if answers.BaseURL != seeded {
			t.Errorf("service URL = %q after a blank answer, want the pre-filled %q", answers.BaseURL, seeded)
		}
	})

	t.Run("seeded port", func(t *testing.T) {
		answers := ConnectDefaults(session.Defaults())
		fields := connectFields(&answers, false, false)
		seeded := answers.Port
		if seeded == "" {
			t.Fatal("port is not pre-filled; the test would prove nothing")
		}

		out := runFieldAccessible(t, fields[3], "\n")
		if strings.Contains(out, "between 1 and 65535") {
			t.Errorf("a blank answer was rejected, so a screen-reader user cannot keep the "+
				"pre-filled port.\noutput:\n%s", out)
		}
		if answers.Port != seeded {
			t.Errorf("port = %q after a blank answer, want the pre-filled %q", answers.Port, seeded)
		}
	})

	t.Run("seeded temperature", func(t *testing.T) {
		answers := CouncilDefaults()
		fields := plainCouncilFields(&answers, new(string), nil)
		// Temperature is the seventh plain field, after question, mode,
		// members, triad, voting and backend.
		temperature := fields[6]
		seeded := answers.Temperature
		if seeded == "" {
			t.Fatal("temperature is not pre-filled; the test would prove nothing")
		}

		out := runFieldAccessible(t, temperature, "\n")
		if strings.Contains(out, "between 0.0 and 2.0") {
			t.Errorf("a blank answer was rejected, so a screen-reader user cannot keep the "+
				"pre-filled temperature.\noutput:\n%s", out)
		}
		if answers.Temperature != seeded {
			t.Errorf("temperature = %q after a blank answer, want the pre-filled %q",
				answers.Temperature, seeded)
		}
	})
}

// TestAccessibleStillRejectsBlankWithoutAPrefill is the negative half, on real
// fields: the question has nothing to keep, so blank must still fail.
func TestAccessibleStillRejectsBlankWithoutAPrefill(t *testing.T) {
	t.Run("question has no default", func(t *testing.T) {
		answers := CouncilDefaults()
		fields := plainCouncilFields(&answers, new(string), nil)

		out := runFieldAccessible(t, fields[0], "\n")
		if !strings.Contains(out, "input cannot be empty") {
			t.Errorf("a required field with no default accepted a blank answer, so the "+
				"blank-pass wrapper leaked.\noutput:\n%s", out)
		}
	})

	t.Run("unseeded service url", func(t *testing.T) {
		answers := ConnectAnswers{} // nothing persisted yet
		fields := connectFields(&answers, false, false)

		out := runFieldAccessible(t, fields[0], "\n")
		if !strings.Contains(out, "a service URL is required") {
			t.Errorf("an unseeded service URL accepted a blank answer.\noutput:\n%s", out)
		}
		if answers.BaseURL != "" {
			t.Errorf("service URL = %q, want it left empty", answers.BaseURL)
		}
	})
}

// TestPlainCouncilFormTypesMembersInsteadOfSelectingThem pins the documented
// reason the accessible path differs: a multi-select's options come from the
// live service, and a screen-reader user on a stopped service would face an
// empty list they cannot answer.
func TestPlainCouncilFormTypesMembersInsteadOfSelectingThem(t *testing.T) {
	var members string
	answers := CouncilDefaults()
	fields := plainCouncilFields(&answers, &members, nil)

	if _, ok := fields[2].(*huh.Input); !ok {
		t.Errorf("plain council field 2 is %T, want *huh.Input so members can be typed", fields[2])
	}
	// The question must be single-line too: huh's accessible textarea prompt
	// keeps reading and consumes the next question's line.
	if _, ok := fields[0].(*huh.Input); !ok {
		t.Errorf("plain council field 0 is %T, want *huh.Input", fields[0])
	}
	// Switching modes and voting must stay closed lists, which have no such
	// dependency.
	if _, ok := fields[1].(*huh.Select[string]); !ok {
		t.Errorf("plain council field 1 is %T, want a Select", fields[1])
	}
}

func TestConnectDefaultsSeedsEveryNonSecretSetting(t *testing.T) {
	answers := ConnectDefaults(session.Defaults())
	for name, got := range map[string]string{
		"service url": answers.BaseURL,
		"port":        answers.Port,
		"oracle host": answers.OracleHost,
		"oracle port": answers.OraclePort,
	} {
		if strings.TrimSpace(got) == "" {
			t.Errorf("%s is not seeded from settings", name)
		}
	}
	if answers.OraclePassword != "" {
		t.Error("the Oracle password must never be seeded from persisted settings")
	}
	if answers.APIKey != "" {
		t.Error("the API key must never be seeded from persisted settings")
	}
	if strings.TrimSpace(api.DefaultBaseURL) == "" {
		t.Error("api.DefaultBaseURL is empty; the seeded-URL assertions above rest on it")
	}
}

// TestToSettingsIsTotal covers the accessible path's blank-setting surprises: a
// skipped optional field must not become an empty environment variable.
func TestToSettingsIsTotal(t *testing.T) {
	settings := ConnectAnswers{BaseURL: "   ", Port: "not a port"}.ToSettings()

	if settings.BaseURL != api.DefaultBaseURL {
		t.Errorf("BaseURL = %q, want the service default", settings.BaseURL)
	}
	if settings.Port != session.DefaultPort {
		t.Errorf("Port = %d, want the default", settings.Port)
	}
	defaults := session.Defaults()
	if settings.OracleHost != defaults.OracleHost {
		t.Errorf("OracleHost = %q, want %q", settings.OracleHost, defaults.OracleHost)
	}
	if settings.OracleService != defaults.OracleService {
		t.Errorf("OracleService = %q, want %q", settings.OracleService, defaults.OracleService)
	}
	if settings.OracleUser != defaults.OracleUser {
		t.Errorf("OracleUser = %q, want %q", settings.OracleUser, defaults.OracleUser)
	}
}

func TestParseMembersSplitsAndTrims(t *testing.T) {
	got := ParseMembers(" qwen3.5:9b , gemma4:latest ,,  , mistral:7b ")
	want := []string{"qwen3.5:9b", "gemma4:latest", "mistral:7b"}
	if len(got) != len(want) {
		t.Fatalf("ParseMembers() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("ParseMembers()[%d] = %q, want %q", i, got[i], want[i])
		}
	}
	if members := ParseMembers("   "); members != nil {
		t.Errorf("ParseMembers(blank) = %v, want nil", members)
	}
}

func TestAccessibleNoticeExplainsTheModeSwitch(t *testing.T) {
	if !strings.Contains(AccessibleNotice, "ACCESSIBLE") {
		t.Errorf("AccessibleNotice = %q, want it to name the variable", AccessibleNotice)
	}
}
