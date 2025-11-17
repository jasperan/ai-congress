# Cline Workspace Guide - AI Congress Project

## 🎯 Project Overview for Cline + Grok4

This document provides Cline (AI coding agent) with all necessary information to build the **AI Congress** project - an LLM swarm system using Python, Ollama, and ensemble decision-making.

## 📋 Project Requirements

**Project Name**: ai-congress

**Core Purpose**: Create a "congress" of LLM agents (using Ollama) that vote on responses using weighted ensemble algorithms based on model accuracy.

**Tech Stack**:
- Backend: Python 3.11+, FastAPI, Ollama Python SDK, asyncio
- Frontend: Svelte + Vite + Tailwind CSS
- CLI: Rich library + Click/Typer
- Database: SQLite with SQLAlchemy
- Deployment: Docker + Docker Compose

## 🎨 Key Features to Implement

1. **Swarm Orchestration**
   - Multi-model mode: Query different models concurrently
   - Multi-request mode: Query same model with temperature variation
   - Hybrid mode: Combination of both
   
2. **Voting Mechanisms**
   - Weighted majority vote (based on model accuracy)
   - Simple majority vote
   - Confidence-based voting
   - Temperature ensemble

3. **User Interfaces**
   - Beautiful web UI (Svelte) inspired by Open WebUI
   - Rich-formatted CLI with high verbosity
   - FastAPI backend with REST + WebSocket endpoints

4. **Model Management**
   - Track available Ollama models
   - Store and use model accuracy weights
   - Pull/manage models via API

## 📁 Complete Project Structure

```
ai-congress/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── setup.py
├── docker-compose.yml
├── Dockerfile
├── .env.example
│
├── config/
│   ├── config.yaml                    # Main configuration
│   └── models_benchmark.json          # Model accuracy scores
│
├── src/
│   └── ai_congress/
│       ├── __init__.py
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── model_registry.py      # Track Ollama models
│       │   ├── swarm_orchestrator.py  # Coordinate multi-model requests
│       │   ├── voting_engine.py       # Ensemble voting algorithms
│       │   └── ollama_client.py       # Ollama API wrapper
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py                # FastAPI app
│       │   ├── routes/
│       │   │   ├── __init__.py
│       │   │   ├── chat.py           # Chat endpoints
│       │   │   ├── models.py         # Model management
│       │   │   └── health.py         # Health checks
│       │   └── websockets/
│       │       ├── __init__.py
│       │       └── chat_ws.py        # WebSocket streaming
│       │
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── main.py               # CLI entry point
│       │   ├── commands/
│       │   │   ├── __init__.py
│       │   │   ├── chat.py           # Interactive chat
│       │   │   ├── models.py         # Model commands
│       │   │   └── config.py         # Configuration
│       │   └── formatters/
│       │       ├── __init__.py
│       │       └── rich_output.py    # Rich formatting
│       │
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── config_loader.py      # YAML/JSON config
│       │   ├── logger.py             # Logging setup
│       │   └── benchmarks.py         # Model benchmarking
│       │
│       └── db/
│           ├── __init__.py
│           ├── models.py             # SQLAlchemy models
│           └── database.py           # Database connection
│
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   │
│   ├── src/
│   │   ├── App.svelte                # Main app component
│   │   ├── main.js                   # Entry point
│   │   │
│   │   ├── components/
│   │   │   ├── Chat/
│   │   │   │   ├── ChatInterface.svelte
│   │   │   │   ├── MessageList.svelte
│   │   │   │   ├── MessageInput.svelte
│   │   │   │   └── StreamingMessage.svelte
│   │   │   │
│   │   │   ├── Models/
│   │   │   │   ├── ModelSelector.svelte
│   │   │   │   ├── ModelCard.svelte
│   │   │   │   └── ModelMetrics.svelte
│   │   │   │
│   │   │   ├── Voting/
│   │   │   │   ├── VotingVisualization.svelte
│   │   │   │   ├── ResponseCard.svelte
│   │   │   │   └── VoteBreakdown.svelte
│   │   │   │
│   │   │   └── Common/
│   │   │       ├── Header.svelte
│   │   │       ├── Sidebar.svelte
│   │   │       └── LoadingSpinner.svelte
│   │   │
│   │   ├── lib/
│   │   │   ├── api.js                # API client
│   │   │   ├── websocket.js          # WebSocket handler
│   │   │   └── stores.js             # Svelte stores
│   │   │
│   │   └── styles/
│   │       └── global.css
│   │
│   └── public/
│       ├── index.html
│       └── favicon.ico
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_model_registry.py
│   ├── test_voting_engine.py
│   ├── test_swarm_orchestrator.py
│   ├── test_api.py
│   └── test_cli.py
│
├── docs/
│   ├── architecture.md
│   ├── api_reference.md
│   ├── cli_guide.md
│   ├── voting_algorithms.md
│   └── deployment.md
│
└── scripts/
    ├── benchmark_models.py           # Collect model accuracy
    ├── setup_dev.sh                  # Development setup
    └── run_local.sh                  # Run locally
```

## 🔑 Implementation Priorities

### Phase 1: Core Infrastructure (Week 1)
1. Project structure setup
2. `model_registry.py` - List Ollama models, track weights
3. `voting_engine.py` - Weighted majority vote algorithm
4. Basic CLI with Rich for testing
5. Configuration loading (config.yaml)

### Phase 2: Swarm Logic (Week 2)
1. `swarm_orchestrator.py` - Concurrent model querying with asyncio
2. Temperature sampling for multi-request mode
3. Response aggregation and ranking
4. Integration of voting engine

### Phase 3: FastAPI Backend (Week 3)
1. `api/main.py` - FastAPI setup with CORS
2. REST endpoints: `/api/chat`, `/api/models`
3. WebSocket endpoint: `/ws/chat` for streaming
4. Error handling and logging

### Phase 4: Svelte Frontend (Week 4)
1. Svelte project setup with Vite + Tailwind
2. `ChatInterface.svelte` - Main chat component
3. `ModelSelector.svelte` - Checkbox list of models
4. WebSocket client for streaming responses
5. `VotingVisualization.svelte` - Show vote breakdown

### Phase 5: CLI Enhancement (Week 5)
1. Rich panels and tables for output
2. Verbose logging modes (0-3)
3. Interactive model selection
4. Progress bars for swarm requests

### Phase 6: Testing & Deployment (Week 6)
1. Unit tests (pytest)
2. Integration tests
3. Dockerfile and docker-compose.yml
4. Documentation

## 🔗 Essential URLs for Reference

### Ollama Integration
- https://github.com/ollama/ollama-python - Official Python SDK
- https://ollama.com/library - Model library
- https://docs.ollama.com/ - Documentation

### Open WebUI (for UI inspiration)
- https://github.com/open-webui/open-webui - Main repo
- https://github.com/open-webui/open-webui/tree/main/src - Frontend source
- https://docs.openwebui.com/ - Docs

### FastAPI
- https://fastapi.tiangolo.com/ - Official docs
- https://fastapi.tiangolo.com/advanced/websockets/ - WebSocket guide
- https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse - Streaming

### Svelte
- https://svelte.dev/docs - Svelte documentation
- https://kit.svelte.dev/docs - SvelteKit docs
- https://tailwindcss.com/docs - Tailwind CSS

### Rich CLI
- https://github.com/Textualize/rich - Rich library
- https://rich.readthedocs.io/ - Documentation
- https://rich.readthedocs.io/en/stable/console.html - Console API

### Ensemble Learning
- https://arxiv.org/abs/2312.12036 - LLM ensemble paper
- https://scikit-learn.org/stable/modules/ensemble.html - Ensemble methods
- https://docs.swarms.world/en/latest/swarms/concept/multi_agent_architectures/

### Model Benchmarks
- https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
- https://collabnix.com/best-ollama-models-2025/

## 📊 Model Weight Examples (MMLU Accuracy)

```json
{
  "phi3:3.8b": 0.69,
  "mistral:7b": 0.82,
  "llama3.1:8b": 0.845,
  "llama3.2:3b": 0.78,
  "gemma2:2b": 0.75,
  "qwen2.5:7b": 0.84,
  "codellama:13b": 0.80,
  "vicuna:13b": 0.76
}
```

## 🧮 Key Algorithms to Implement

### 1. Weighted Majority Vote

```python
def weighted_majority_vote(responses, weights):
    # Group identical responses
    # Sum weights for each unique response
    # Return response with highest weight
    response_weights = {}
    for response, weight in zip(responses, weights):
        normalized = response.strip().lower()
        response_weights[normalized] = response_weights.get(normalized, 0) + weight
    
    winner = max(response_weights.items(), key=lambda x: x[1])
    confidence = winner[1] / sum(weights)
    return winner[0], confidence
```

### 2. Temperature Ensemble

```python
async def temperature_ensemble(model, prompt, temperatures=[0.3, 0.7, 1.0]):
    tasks = [
        ollama_client.chat(model, prompt, temperature=t)
        for t in temperatures
    ]
    responses = await asyncio.gather(*tasks)
    
    # Weight by inverse temperature (lower = more confident)
    weights = [1.0 / (t + 0.1) for t in temperatures]
    
    return weighted_majority_vote(responses, weights)
```

### 3. Multi-Model Concurrent Query

```python
async def multi_model_swarm(models, prompt):
    # Query all models concurrently
    tasks = [
        ollama_client.chat(model, prompt)
        for model in models
    ]
    
    responses = await asyncio.gather(*tasks)
    
    # Get model weights from registry
    weights = [model_registry.get_weight(m) for m in models]
    
    return weighted_majority_vote(responses, weights)
```

## 🎨 UI Component Breakdown

### ChatInterface.svelte
- Message list with auto-scroll
- Input field with send button
- Model selector sidebar
- Voting visualization panel
- Streaming response indicators

### ModelSelector.svelte
- Checkbox list of available models
- Show model size and weight
- "Select All" / "Clear All" buttons
- Filter by size/type

### VotingVisualization.svelte
- Bar chart of vote weights
- Cards for each unique response
- Model names that voted for each
- Confidence meter (0-100%)

## 🔧 Configuration Schema (config.yaml)

```yaml
ollama:
  base_url: "http://localhost:11434"
  timeout: 120

swarm:
  default_mode: "multi_model"  # multi_model | multi_request | hybrid
  max_concurrent_requests: 10
  multi_request:
    temperatures: [0.3, 0.7, 1.0, 1.2]

voting:
  algorithm: "weighted_majority"  # weighted_majority | majority | confidence
  consensus_threshold: 0.6

models:
  preferred:
    - "phi3:3.8b"
    - "mistral:7b"
    - "llama3.2:3b"
  weights:
    "phi3:3.8b": 0.69
    "mistral:7b": 0.82

api:
  host: "0.0.0.0"
  port: 8000

cli:
  verbosity: 2  # 0=minimal, 1=normal, 2=verbose, 3=debug
  use_colors: true
```

## 🧪 Testing Requirements

### Unit Tests
- `test_voting_engine.py` - All voting algorithms
- `test_model_registry.py` - Model tracking, weights
- `test_swarm_orchestrator.py` - Concurrent queries

### Integration Tests
- `test_api.py` - FastAPI endpoints with TestClient
- End-to-end swarm workflow

### Test Coverage
- Aim for 80%+ code coverage
- All critical paths tested

## 📦 Dependencies (requirements.txt)

```txt
# Core
fastapi==0.115.0
uvicorn[standard]==0.30.6
ollama==0.3.3

# Async
aiohttp==3.10.5
asyncio-mqtt==0.16.2

# CLI
rich==13.8.1
click==8.1.7

# Database
sqlalchemy==2.0.35
aiosqlite==0.20.0

# Config
pyyaml==6.0.2
python-dotenv==1.0.1

# Testing
pytest==8.3.3
pytest-asyncio==0.24.0
httpx==0.27.2
```

## 🐳 Docker Setup

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.ai_congress.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
    volumes:
      - ./config:/app/config
  
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - api
```

## ✅ Acceptance Criteria

A fully working AI Congress system should:

1. ✅ List available Ollama models
2. ✅ Query 3+ models concurrently
3. ✅ Implement weighted majority voting
4. ✅ Display results in beautiful CLI with Rich
5. ✅ Provide REST API endpoints
6. ✅ Stream responses via WebSocket
7. ✅ Show voting visualization in web UI
8. ✅ Allow model selection via checkboxes
9. ✅ Support multi-request mode with temperature
10. ✅ Handle errors gracefully

## 🚀 Getting Started Commands for Cline

```bash
# 1. Create project structure
mkdir -p ai-congress/{src/ai_congress/{core,api/routes,api/websockets,cli/commands,cli/formatters,utils,db},frontend/src/{components/{Chat,Models,Voting,Common},lib,styles},tests,docs,scripts,config}

# 2. Create __init__.py files
find src -type d -exec touch {}/__init__.py \;

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull lightweight models
ollama pull phi3:3.8b
ollama pull mistral:7b
ollama pull llama3.2:3b

# 5. Run backend
uvicorn src.ai_congress.api.main:app --reload

# 6. Test CLI
python -m src.ai_congress.cli.main chat
```

## 💡 Code Style Guidelines

- Use **async/await** for all I/O operations
- Type hints for all function signatures
- Docstrings for public methods (Google style)
- Black formatting (line length 100)
- Import order: stdlib, third-party, local

## 🎯 Success Metrics

- Response time: < 5 seconds for 3-model swarm
- UI responsiveness: Real-time streaming
- CLI clarity: Rich formatted, easy to read
- Code quality: 80%+ test coverage
- Documentation: Complete README and API docs

---

**This workspace guide provides everything Cline needs to build AI Congress from scratch. Start with Phase 1 and progress systematically through each component.**
