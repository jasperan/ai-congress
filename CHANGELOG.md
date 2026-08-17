# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Phase 3 — Learning persistence + model-agnostic bootstrap
- **3.5.1 live-catalog bootstrap**: `ModelRegistry.list_available_models` now seeds a neutral 0.5 weight for every model actually in Ollama and purges stale benchmark keys (no more learning against model ghosts); `load_benchmark_weights` applies only to installed models. Catalog outages no longer wipe weights.
- **3.5.2 persisted learning**: dynamic weights, confidence calibration, feedback log, and circuit-breaker state survive restarts via atomic JSON writes under `data/` (`LearningConfig` overrides paths). New shared `utils/persistence.py` (tmp+rename atomic writes).
- **3.5.3 feedback→weight bridge**: `record_feedback` now moves the model's dynamic weight with a small EMA delta (positive up / negative down) — the feedback loop finally does something. Feedback entries also carry a `domain` tag (3.5.4) and the `/api/feedback` route accepts it.
- **3.4.3 circuit-breaker persistence**: OPEN breakers (with failure timestamps) are restored on restart, so a model that hung all day is not retried immediately after every process start; recovery timing is preserved.

### Phase 2 — Wire dormant intelligence into the enhanced pipeline
- **3.1.1 reasoning_mode actually changes prompts**: `MODE_INSTRUCTIONS` in `role_prompts.py`; cot/react instructions are appended to the Wave-1 user prompt — routing finally has an effect on output.
- **3.1.4 AgentMemory into the pipeline**: recall before Wave 1 (top relevant exchanges injected as continuity context), `add_exchange(prompt, winner)` after each run. Gated by `intelligence.memory_enabled`.
- **3.1.6 self-consistency gating**: when Wave-1 `agreement_ratio < min_agreement`, each model is re-sampled N× at temperature 0.9 and ensemble-voted; the resampled winner is adopted only if confidence rises. Config: `intelligence.self_consistency.{enabled,samples,min_agreement,temperature}` — cost-gated by design.
- **3.3.3 evidence-grounded deliberation**: `deliberation_swarm(..., evidence=True)` (or `config.deliberation.evidence_grounded`) searches the web, injects evidence into Round 1 as constrained context and Round 3 as a cross-check, and attaches `evidence_alignment` per final position. Degrades gracefully when web search is unavailable.
- **3.5.5 prompt-evolution A/B**: debate (critique/pressure) instructions and deliberation Round-2 cross-examination route through `PromptEvolution.select_template` with `record_outcome` per run.

### Added
- **pi inference backend**: run the congress against `deepseek-v4-flash` (and other cloud models) via opencode-go alongside local Ollama — selectable per request via `inference_backend: pi | ollama | openai` (API/WS/CLI). Enabled by setting `OPENCODE_GO_API_KEY`.
- `PiBackendConfig` in the config loader with env overrides (`PI_BASE_URL`, `PI_MODEL`); env overrides now apply even when `config.yaml` is absent.
- OpenAI client: `max_tokens` cap (unbounded reasoning budgets on DeepSeek-style models no longer stall) + `reasoning_content` capture for reasoning models.
- Backend-aware summarizer selection in `semantic_confidence` (pi/openai runs no longer fall back to local Ollama for scoring).
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
- Model listing with ollama>=0.6 (pydantic model dumps carry `model`, not `name`).

## [0.2.0] - previous
- See git history (`git log`) for the pre-changelog period.
