package session

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// isolateConfig points the user config dir at a temp directory so a test never
// reads or writes the real settings file.
func isolateConfig(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	t.Setenv("HOME", dir)
	return dir
}

func TestDefaultsMatchTheDocumentedServiceDefaults(t *testing.T) {
	defaults := Defaults()
	if defaults.BaseURL != "http://127.0.0.1:8000" {
		t.Errorf("BaseURL = %q, want http://127.0.0.1:8000 (run_server.py's default)", defaults.BaseURL)
	}
	if defaults.Port != 8000 {
		t.Errorf("Port = %d, want 8000", defaults.Port)
	}
	if defaults.LaunchServer {
		t.Error("LaunchServer defaults to true; starting a service must be opt-in")
	}
	// .env.example documents these exact Oracle values.
	if defaults.OracleService != "FREEPDB1" {
		t.Errorf("OracleService = %q, want FREEPDB1", defaults.OracleService)
	}
	if defaults.OraclePort != "1521" {
		t.Errorf("OraclePort = %q, want 1521", defaults.OraclePort)
	}
}

func TestSaveLoadRoundTripAndPermissions(t *testing.T) {
	dir := isolateConfig(t)

	if loaded, err := Load(); err != nil || loaded.BaseURL != Defaults().BaseURL {
		t.Fatalf("Load() before any save = %+v, %v; want the defaults", loaded, err)
	}

	saved := Settings{
		BaseURL:       "http://127.0.0.1:9100",
		LaunchServer:  true,
		Port:          9100,
		OracleHost:    "db.example",
		OraclePort:    "1522",
		OracleUser:    "SOMEUSER",
		OracleService: "OTHERDB",
	}
	if err := Save(saved); err != nil {
		t.Fatalf("Save() = %v", err)
	}

	path := filepath.Join(dir, "ai-congress", "gotui.json")
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat %s: %v", path, err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Errorf("config permissions = %o, want 600", perm)
	}

	loaded, err := Load()
	if err != nil {
		t.Fatalf("Load() = %v", err)
	}
	if loaded != saved {
		t.Errorf("Load() = %+v, want %+v", loaded, saved)
	}
}

// TestSettingsNeverHoldASecret pins the package's central promise: the file is
// allowed to describe where to connect, never how to authenticate.
func TestSettingsNeverHoldASecret(t *testing.T) {
	source, err := os.ReadFile("session.go")
	if err != nil {
		t.Fatalf("read session.go: %v", err)
	}
	// The Settings struct must not carry a password or API key field, and the
	// JSON tags must not either.
	structStart := strings.Index(string(source), "type Settings struct {")
	if structStart < 0 {
		t.Fatal("Settings struct not found; this test needs updating")
	}
	structEnd := strings.Index(string(source)[structStart:], "}")
	body := string(source)[structStart : structStart+structEnd]
	for _, forbidden := range []string{"Password", "password", "APIKey", "api_key"} {
		if strings.Contains(body, forbidden) {
			t.Errorf("Settings contains %q; a persisted file must hold no secret", forbidden)
		}
	}
}

func TestPortValidation(t *testing.T) {
	valid := []string{"8000", "1", "65535", "  8100  "}
	for _, raw := range valid {
		if err := ValidatePort(raw); err != nil {
			t.Errorf("ValidatePort(%q) = %v, want nil", raw, err)
		}
	}

	invalid := []string{"", "0", "-1", "65536", "http", "80o0"}
	for _, raw := range invalid {
		if err := ValidatePort(raw); err == nil {
			t.Errorf("ValidatePort(%q) = nil, want an error", raw)
		}
		if got := ParsePort(raw); got != DefaultPort {
			t.Errorf("ParsePort(%q) = %d, want the default %d", raw, got, DefaultPort)
		}
	}
}

// TestServerArgsCarryNoCredential is the security-critical assertion: argv is
// world-readable through /proc/<pid>/cmdline.
func TestServerArgsCarryNoCredential(t *testing.T) {
	argv := ServerArgs("/repo/.venv/bin/python", 8000)
	joined := strings.Join(argv, " ")

	for _, secret := range []string{"hunter2", "sk-live-abc", "ORACLE_PASSWORD"} {
		if strings.Contains(joined, secret) {
			t.Errorf("argv %q contains %q; credentials must never be arguments", joined, secret)
		}
	}

	// The documented target must be the module run_server.py serves.
	if !strings.Contains(joined, "src.ai_congress.api.main:app") {
		t.Errorf("argv %q does not point at the app module", joined)
	}
	// reload must stay off: the reloader forks a process that outlives Stop().
	if strings.Contains(joined, "--reload") {
		t.Errorf("argv %q enables reload, which leaves an orphan holding the port", joined)
	}
	if !strings.Contains(joined, "--port 8000") {
		t.Errorf("argv %q does not pass the port", joined)
	}
	if !strings.Contains(joined, "127.0.0.1") {
		t.Errorf("argv %q does not bind loopback", joined)
	}
}

func TestServerEnvPassesSecretsThroughTheEnvironment(t *testing.T) {
	settings := Defaults()
	settings.OracleUser = "ADMIN"
	settings.OracleHost = "db.example"

	env := ServerEnv(settings, "s3cret", "api-key-123")
	joined := strings.Join(env, "\n")

	for _, want := range []string{
		EnvOraclePassword + "=s3cret",
		EnvAPIKey + "=api-key-123",
		EnvOracleUsername + "=ADMIN",
		EnvOracleHost + "=db.example",
		EnvHost + "=127.0.0.1",
		"AI_CONGRESS_PORT=8000",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("environment is missing %q", want)
		}
	}

	// A blank secret must not be exported: appending an empty ORACLE_PASSWORD
	// would override a real one the user's shell already provided. The count is
	// compared rather than the substring, because os.Environ() may legitimately
	// already contain an empty value that this function did not add.
	inherited := countEntries(os.Environ(), EnvOraclePassword)
	blank := ServerEnv(settings, "", "")
	if got := countEntries(blank, EnvOraclePassword); got != inherited {
		t.Errorf("ServerEnv appended a blank %s (count %d, inherited %d)", EnvOraclePassword, got, inherited)
	}
	if got := countEntries(blank, EnvAPIKey); got != countEntries(os.Environ(), EnvAPIKey) {
		t.Errorf("ServerEnv appended a blank %s", EnvAPIKey)
	}
}

// countEntries counts environment entries with the given key.
func countEntries(env []string, key string) int {
	count := 0
	for _, entry := range env {
		if strings.HasPrefix(entry, key+"=") {
			count++
		}
	}
	return count
}

func TestPythonForPrefersTheRepoVirtualenv(t *testing.T) {
	root := t.TempDir()

	// No venv yet: fall back to the system interpreter name.
	if got := PythonFor(root); got != "python3" {
		t.Errorf("PythonFor(empty) = %q, want python3", got)
	}

	venvPython := filepath.Join(root, ".venv", "bin", "python")
	if err := os.MkdirAll(filepath.Dir(venvPython), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(venvPython, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatalf("write: %v", err)
	}
	if got := PythonFor(root); got != venvPython {
		t.Errorf("PythonFor(venv) = %q, want %q", got, venvPython)
	}

	// A directory named python must not be mistaken for an interpreter.
	if err := os.Remove(venvPython); err != nil {
		t.Fatalf("remove: %v", err)
	}
	if err := os.MkdirAll(venvPython, 0o755); err != nil {
		t.Fatalf("mkdir python: %v", err)
	}
	if got := PythonFor(root); got != "python3" {
		t.Errorf("PythonFor(dir named python) = %q, want python3", got)
	}
}

func TestWaitForPortTimesOutWithoutHanging(t *testing.T) {
	start := time.Now()
	// Port 1 on loopback refuses connections immediately.
	err := WaitForPort(context.Background(), "127.0.0.1", 1, 900*time.Millisecond)
	if err == nil {
		t.Fatal("WaitForPort() = nil for a port nothing listens on")
	}
	if !strings.Contains(err.Error(), "did not accept connections") {
		t.Errorf("error = %q, want it to explain the timeout", err)
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Errorf("WaitForPort took %s; it should give up at the deadline", elapsed)
	}
}

func TestWaitForPortHonoursCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := WaitForPort(ctx, "127.0.0.1", 1, 30*time.Second); err == nil {
		t.Fatal("WaitForPort() = nil with a cancelled context")
	}
}
