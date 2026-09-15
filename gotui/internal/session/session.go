// Package session holds the non-secret connection settings for the Go
// front-end, plus the one mechanism used to hand an Oracle password to the
// Python service: environment variables.
//
// Why environment and never argv: a command line is world-readable via
// /proc/<pid>/cmdline, and shell history records it. The repo's own
// .env.example already documents ORACLE_PASSWORD as the channel the service
// reads, so this is the codebase's supported interface rather than an
// invention.
package session

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Environment variables the service reads, matched to .env.example and
// run_server.py.
const (
	EnvOraclePassword = "ORACLE_PASSWORD"
	EnvOracleUsername = "ORACLE_USER"
	EnvOracleHost     = "ORACLE_HOST"
	EnvOraclePort     = "ORACLE_PORT"
	EnvOracleService  = "ORACLE_SERVICE"
	EnvAPIKey         = "AI_CONGRESS_API_KEY"
	EnvHost           = "AI_CONGRESS_HOST"
	EnvPort           = "AI_CONGRESS_PORT"

	DefaultPort       = 8000
	DefaultOracleHost = "localhost"
	DefaultOraclePort = "1521"
	DefaultOracleSvc  = "FREEPDB1"
	DefaultOracleUser = "ADMIN"
)

// Settings are the persisted, NON-SECRET connection settings. There is
// deliberately no password field: see the package comment.
type Settings struct {
	BaseURL       string `json:"base_url"`
	LaunchServer  bool   `json:"launch_server"`
	Port          int    `json:"port"`
	OracleHost    string `json:"oracle_host"`
	OraclePort    string `json:"oracle_port"`
	OracleUser    string `json:"oracle_user"`
	OracleService string `json:"oracle_service"`
}

// Defaults returns the settings a first run should start from.
func Defaults() Settings {
	return Settings{
		BaseURL:       "http://127.0.0.1:8000",
		LaunchServer:  false,
		Port:          DefaultPort,
		OracleHost:    DefaultOracleHost,
		OraclePort:    DefaultOraclePort,
		OracleUser:    DefaultOracleUser,
		OracleService: DefaultOracleSvc,
	}
}

// ConfigPath is where Settings live. The file is written 0600 and holds no
// secret, so a leaked copy reveals only a URL and a hostname.
func ConfigPath() (string, error) {
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", fmt.Errorf("locate config dir: %w", err)
	}
	return filepath.Join(dir, "ai-congress", "gotui.json"), nil
}

// Load reads Settings, falling back to Defaults when there is no file yet.
func Load() (Settings, error) {
	path, err := ConfigPath()
	if err != nil {
		return Defaults(), err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return Defaults(), nil
		}
		return Defaults(), fmt.Errorf("read %s: %w", path, err)
	}
	settings := Defaults()
	if err := json.Unmarshal(raw, &settings); err != nil {
		return Defaults(), fmt.Errorf("parse %s: %w", path, err)
	}
	return settings, nil
}

// Save writes Settings with 0600 permissions.
func Save(settings Settings) error {
	path, err := ConfigPath()
	if err != nil {
		return err
	}
	encoded, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return fmt.Errorf("encode settings: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return fmt.Errorf("create %s: %w", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, append(encoded, '\n'), 0o600); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	return nil
}

// PasswordFromEnv returns the Oracle password from the environment, if any.
//
// This is how a user avoids typing it at all: export ORACLE_PASSWORD once and
// every flow below picks it up.
func PasswordFromEnv() string { return os.Getenv(EnvOraclePassword) }

// APIKeyFromEnv returns the shared-secret API key from the environment, if any.
func APIKeyFromEnv() string { return os.Getenv(EnvAPIKey) }

// ValidatePort rejects a port huh could have accepted as text.
func ValidatePort(raw string) error {
	value, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil {
		return errors.New("use a number between 1 and 65535")
	}
	if value < 1 || value > 65535 {
		return errors.New("use a number between 1 and 65535")
	}
	return nil
}

// ParsePort converts a validated port string.
func ParsePort(raw string) int {
	value, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil || value < 1 || value > 65535 {
		return DefaultPort
	}
	return value
}

// ServerEnv builds the environment for the spawned Python service.
//
// The password is placed in the child's environment only. It is never an
// argument, never written to disk, and never logged.
func ServerEnv(s Settings, oraclePassword, apiKey string) []string {
	env := os.Environ()
	set := func(key, value string) {
		if strings.TrimSpace(value) != "" {
			env = append(env, key+"="+value)
		}
	}
	set(EnvHost, "127.0.0.1")
	set(EnvPort, strconv.Itoa(s.Port))
	set(EnvOracleHost, s.OracleHost)
	set(EnvOraclePort, s.OraclePort)
	set(EnvOracleService, s.OracleService)
	set(EnvOracleUsername, s.OracleUser)
	set(EnvOraclePassword, oraclePassword)
	set(EnvAPIKey, apiKey)
	return env
}

// PythonFor returns the interpreter the service should be started with.
//
// The repo ships a .venv with FastAPI, uvicorn and websockets installed, and
// run_server.py documents the module path; preferring the venv means the Go
// front-end works on a checkout that was set up with ./startup.sh and does not
// silently pick up a system interpreter that lacks the dependencies.
func PythonFor(projectRoot string) string {
	candidates := []string{
		filepath.Join(projectRoot, ".venv", "bin", "python"),
		filepath.Join(projectRoot, "venv", "bin", "python"),
	}
	for _, candidate := range candidates {
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return candidate
		}
	}
	return "python3"
}

// ServerArgs is the exact argv used to start the service.
//
// It is exported so a test can assert that no credential can ever appear here:
// /proc/<pid>/cmdline is world readable, so a secret in argv would leak it to
// every user on the box regardless of file permissions.
//
// The module path is run_server.py's own target. reload is deliberately NOT
// enabled: the reloader forks a second process, which would survive Stop() and
// hold the port.
func ServerArgs(python string, port int) []string {
	return []string{
		python, "-m", "uvicorn", "src.ai_congress.api.main:app",
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
	}
}

// Server is a spawned AI Congress API process.
type Server struct {
	cmd    *exec.Cmd
	stderr strings.Builder
}

// Stop terminates the spawned service and any child it left behind.
func (s *Server) Stop() {
	if s == nil || s.cmd == nil || s.cmd.Process == nil {
		return
	}
	_ = s.cmd.Process.Kill()
	_, _ = s.cmd.Process.Wait()
}

// Stderr returns whatever the service logged, for error reporting.
func (s *Server) Stderr() string { return s.stderr.String() }

// LaunchServer starts the repo's own API, using the interface the repo already
// documents (src.ai_congress.api.main:app via uvicorn) rather than
// reimplementing it.
func LaunchServer(ctx context.Context, projectRoot string, port int, env []string) (*Server, error) {
	argv := ServerArgs(PythonFor(projectRoot), port)
	cmd := exec.Command(argv[0], argv[1:]...) //nolint:gosec // argv is built above from constants plus an int port.
	cmd.Dir = projectRoot
	cmd.Env = env

	server := &Server{cmd: cmd}
	cmd.Stdout = discardWriter{}
	cmd.Stderr = &server.stderr
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start AI Congress API: %w", err)
	}
	return server, nil
}

// discardWriter swallows the service's stdout; the TUI owns the screen.
type discardWriter struct{}

func (discardWriter) Write(p []byte) (int, error) { return len(p), nil }

// WaitForPort polls until the service accepts TCP connections, so the TUI
// shows a deterministic "starting" state instead of a connection error.
func WaitForPort(ctx context.Context, host string, port int, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	address := net.JoinHostPort(host, strconv.Itoa(port))
	for {
		conn, err := net.DialTimeout("tcp", address, 500*time.Millisecond)
		if err == nil {
			_ = conn.Close()
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("service did not accept connections on %s within %s", address, timeout)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(250 * time.Millisecond):
		}
	}
}
