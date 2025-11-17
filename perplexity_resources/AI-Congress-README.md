# AI Congress - LLM Swarm with Ensemble Decision Making

> **A Python + Ollama project that creates a "congress" of autonomous LLM agents that vote on responses using ensemble decision-making algorithms**

## 🎯 Project Vision

AI Congress is an innovative LLM swarm system where multiple Ollama models work together like a legislative body - each model contributes its perspective, and the final answer is determined through weighted voting based on model accuracy and performance metrics.

## ✨ Key Features

- **🗳️ Ensemble Decision Making**: Multiple LLMs vote on responses with weights based on their benchmark accuracy scores
- **🔄 Dual Swarm Modes**:
  - **Multi-Model**: Query different models simultaneously (e.g., Mistral, Llama, Phi-3)
  - **Multi-Request**: Query same model multiple times with temperature variation
  - **Hybrid**: Combination of both approaches
- **🎨 Beautiful Web UI**: Modern Svelte-based chat interface inspired by Open WebUI
- **💻 Rich CLI**: Command-line interface with stunning Rich library formatting
- **⚡ Async & Concurrent**: Built on asyncio for maximum performance
- **🪶 Lightweight Focus**: Prioritizes efficient models (3B-13B parameters)
- **📊 Voting Algorithms**: Weighted majority, confidence-based, boosting, and temperature ensemble
- **🔍 High Verbosity**: Debug-friendly with detailed logging options

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  User Interface                      │
│  ┌──────────────┐              ┌─────────────────┐  │
│  │   Web UI     │              │   CLI (Rich)    │  │
│  │  (Svelte)    │              │                 │  │
│  └──────┬───────┘              └────────┬────────┘  │
│         │                               │           │
│         └───────────┬───────────────────┘           │
│                     │                               │
│         ┌───────────▼────────────┐                  │
│         │   FastAPI Backend      │                  │
│         │   (WebSocket + REST)   │                  │
│         └───────────┬────────────┘                  │
│                     │                               │
│         ┌───────────▼────────────┐                  │
│         │  Swarm Orchestrator    │                  │
│         │  (Async Coordination)  │                  │
│         └───────────┬────────────┘                  │
│                     │                               │
│    ┌────────────────┼────────────────┐              │
│    │                │                │              │
│    ▼                ▼                ▼              │
│ ┌──────┐        ┌──────┐        ┌──────┐           │
│ │Model │        │Model │        │Model │           │
│ │  A   │        │  B   │        │  C   │           │
│ └──┬───┘        └──┬───┘        └──┬───┘           │
│    │               │               │               │
│    └───────────────┼───────────────┘               │
│                    │                               │
│         ┌──────────▼───────────┐                   │
│         │   Voting Engine      │                   │
│         │ (Weighted Ensemble)  │                   │
│         └──────────┬───────────┘                   │
│                    │                               │
│         ┌──────────▼───────────┐                   │
│         │   Final Decision     │                   │
│         └──────────────────────┘                   │
└─────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Ollama installed and running
- Node.js 18+ (for frontend)
- At least 8GB RAM (for lightweight models)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ai-congress.git
cd ai-congress

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Pull some lightweight Ollama models
ollama pull phi3:3.8b
ollama pull mistral:7b
ollama pull llama3.2:3b

# 4. Install frontend dependencies
cd frontend
npm install
cd ..

# 5. Run the application
# Terminal 1 - Backend
uvicorn src.ai_congress.api.main:app --reload

# Terminal 2 - Frontend
cd frontend
npm run dev

# Terminal 3 - CLI (optional)
python -m src.ai_congress.cli.main chat
```

## 💡 Usage Examples

### CLI Chat

```bash
# Interactive chat with multi-model swarm
python -m src.ai_congress.cli.main chat --mode multi_model -vv

# List available models
python -m src.ai_congress.cli.main models

# Pull a new model
python -m src.ai_congress.cli.main pull gemma2:2b
```

### Python API

```python
import asyncio
from ai_congress.core import ModelRegistry, VotingEngine, SwarmOrchestrator

async def main():
    # Initialize components
    registry = ModelRegistry()
    voting = VotingEngine()
    swarm = SwarmOrchestrator(registry, voting)
    
    # Query multiple models
    result = await swarm.multi_model_swarm(
        models=['phi3:3.8b', 'mistral:7b', 'llama3.2:3b'],
        prompt="Explain quantum computing in simple terms"
    )
    
    print(f"Final Answer: {result['final_answer']}")
    print(f"Confidence: {result['confidence']:.2%}")

asyncio.run(main())
```

### REST API

```bash
# Chat request
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of France?",
    "models": ["phi3:3.8b", "mistral:7b"],
    "mode": "multi_model"
  }'

# List models
curl http://localhost:8000/api/models
```

## 🗳️ Voting Mechanisms

### 1. Weighted Majority Vote (Default)

Each model's vote is weighted by its accuracy score from benchmarks (MMLU, HumanEval, etc.)

```python
weighted_score = Σ(model_accuracy_i × vote_i)
```

**Example**: Phi-3 (85% accuracy) + Mistral (82% accuracy) + Llama (78% accuracy)

### 2. Temperature Ensemble

Same model queried multiple times with different temperatures (0.3, 0.7, 1.0, 1.2) to generate diverse responses. Lower temperatures get higher weights.

### 3. Confidence-Based Voting

Uses model's internal confidence scores when available.

### 4. Hybrid Mode

Combines multiple models AND multiple temperatures for maximum diversity.

## 📦 Project Structure

```
ai-congress/
├── config/
│   ├── config.yaml                    # Main configuration
│   └── models_benchmark.json          # Model accuracy weights
│
├── src/ai_congress/
│   ├── core/
│   │   ├── model_registry.py          # Track Ollama models
│   │   ├── swarm_orchestrator.py      # Coordinate requests
│   │   ├── voting_engine.py           # Ensemble algorithms
│   │   └── ollama_client.py           # Ollama wrapper
│   │
│   ├── api/
│   │   ├── main.py                    # FastAPI app
│   │   ├── routes/                    # API endpoints
│   │   └── websockets/                # WebSocket streaming
│   │
│   ├── cli/
│   │   ├── main.py                    # CLI entry point
│   │   ├── commands/                  # CLI commands
│   │   └── formatters/                # Rich formatting
│   │
│   └── utils/
│       ├── config_loader.py
│       ├── logger.py
│       └── benchmarks.py
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Chat/                  # Chat interface
│   │   │   ├── Models/                # Model selector
│   │   │   ├── Voting/                # Vote visualization
│   │   │   └── Common/                # Shared components
│   │   ├── lib/
│   │   │   ├── api.js                 # API client
│   │   │   └── websocket.js           # WebSocket handler
│   │   └── App.svelte
│   └── package.json
│
├── tests/
├── docs/
└── scripts/
```

## 🎨 Web UI Features

- **Real-time Streaming**: See responses as they come in from each model
- **Model Selection**: Check/uncheck models to include in the swarm
- **Vote Visualization**: See which models voted for which response
- **Confidence Meter**: Visual representation of ensemble confidence
- **Response Comparison**: Side-by-side view of different model outputs
- **Dark/Light Theme**: Beautiful UI in both modes

## 🔧 Configuration

Edit `config/config.yaml`:

```yaml
swarm:
  default_mode: "multi_model"
  max_concurrent_requests: 10
  
voting:
  default_algorithm: "weighted_majority"
  consensus_threshold: 0.6

models:
  preferred:
    - "phi3:3.8b"
    - "mistral:7b"
    - "llama3.2:3b"
  
  weights:
    "phi3:3.8b": 0.85
    "mistral:7b": 0.82
    "llama3.2:3b": 0.78
```

## 🪶 Recommended Lightweight Models

| Model | Size | RAM | MMLU | Use Case |
|-------|------|-----|------|----------|
| phi3:3.8b | 3.8B | ~2.5GB | 69% | Best quality/size ratio |
| gemma2:2b | 2B | ~2GB | - | Multilingual support |
| mistral:7b | 7B | ~4GB | 82% | Balanced performance |
| llama3.2:3b | 3B | ~3GB | - | General purpose |
| qwen2.5:7b | 7B | ~5GB | - | Strong reasoning |

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/ai_congress

# Run specific test
pytest tests/test_voting_engine.py -v
```

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up -d

# Access
# Web UI: http://localhost:3000
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

## 📊 Benchmarking

Collect model accuracy data:

```bash
python scripts/benchmark_models.py --models phi3:3.8b,mistral:7b --dataset mmlu
```

## 🛠️ Development Workflow

1. **Setup Development Environment**
   ```bash
   ./scripts/setup_dev.sh
   ```

2. **Run in Development Mode**
   ```bash
   # Backend with auto-reload
   uvicorn src.ai_congress.api.main:app --reload
   
   # Frontend with hot reload
   cd frontend && npm run dev
   ```

3. **Format Code**
   ```bash
   black src/
   isort src/
   ```

4. **Type Checking**
   ```bash
   mypy src/
   ```

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## 📝 License

MIT License - see [LICENSE](LICENSE)

## 🙏 Acknowledgments

- **Ollama** - For making local LLMs accessible
- **Open WebUI** - UI design inspiration
- **Rich** - Beautiful CLI formatting
- **FastAPI** - Modern Python web framework
- **Svelte** - Reactive frontend framework

## 🔗 Resources

- [Ollama Documentation](https://ollama.com/docs)
- [Open WebUI GitHub](https://github.com/open-webui/open-webui)
- [FastAPI Guide](https://fastapi.tiangolo.com/)
- [Rich Documentation](https://rich.readthedocs.io/)
- [LLM Ensemble Learning Paper](https://arxiv.org/abs/2312.12036)

## 📮 Support

- GitHub Issues: [Report bugs](https://github.com/yourusername/ai-congress/issues)
- Discussions: [Ask questions](https://github.com/yourusername/ai-congress/discussions)

---

**Built with ❤️ for the open-source AI community**
