# Cline Workspace Guide - AI Congress Project

> **Note:** This file predates the project's current state. The authoritative
> project documentation is **[CLAUDE.md](../CLAUDE.md)** at the repo root.
> Read that first; the highlights below are preserved for historical context.

## 🎯 Project Overview

**ai-congress** — an LLM swarm system using Python, Ollama, and ensemble
decision-making where multiple models vote on responses using weighted
ensemble algorithms.

## Current Stack (see CLAUDE.md for details)

- Backend: Python 3.10+, FastAPI, async Ollama SDK, asyncio
- Frontend: Svelte + Vite + Tailwind CSS
- TUI: Rust + ratatui (`tui-rs/`)
- Database: Oracle 26ai Free (optional data lake; app degrades gracefully)
- Deployment: Docker + Docker Compose, uv-managed environment

## Quick Start

```bash
./startup.sh          # deps + services
./run_cli.py chat "question" -m qwen3.5:9b
python run_server.py  # API at :8000
cd frontend && npm run dev  # UI at :3000
```
