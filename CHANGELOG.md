# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added
- Backend CI workflow (`.github/workflows/backend.yml`): lint/compile check + hermetic unit suite on Python 3.10–3.12.
- `.env.example` documenting all optional environment configuration (Oracle datalake, CORS, API key, voice, search).
- `.dockerignore`, compose healthcheck, and standardized API port (8000 everywhere).
- `api/state.py` + `api/schemas.py` + route modules under `api/routes/` — `api/main.py` is now a composition root (was a 1,200-line monolith).
- Test auto-classification: any test not marked `integration` is treated as `unit` (hermetic CI-safe).
- `/api/models` short-TTL cache.
- Upload validation for `/api/documents/upload` (extension allowlist + 25 MB cap).

### Changed
- Dependencies: heavy ML deps (torch, sentence-transformers, faster-whisper, transformers) moved to optional `full` extra; core requirements trimmed accordingly. Startup uses `uv` when available.
- Default models updated to the live Ollama catalog (qwen3.5:9b, gemma3:4b, qwen2.5:1.5b) across config, CLI, API, and WebSocket.
- `startup.sh` uses `uv` with pip fallback and pulls current recommended models.
- Lazy-import + graceful degradation for `web_search.py` (DuckDuckGo) and `voice.py` (faster-whisper): missing optional deps no longer break app/tests; transcription returns 501 when unavailable.
- `swarm_orchestrator.query_model` now marks failed/empty model calls as `success=False` (was swallowing 404s as empty successes); `semantic_confidence` early-returns on blank responses.
- README port table, model names, and docs links corrected; stale `AGENT_HARNESS_ANNEX.md` reference replaced.

### Fixed
- Test collection error caused by hard `duckduckgo_search` / `faster_whisper` imports.
- `adaptive_chunking` test assertion inconsistent with the code default.

## [0.2.0] - previous
- See git history (`git log`) for the pre-changelog period.
