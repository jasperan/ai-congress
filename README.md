![Logo](img/logo.jpg)

# ai-congress
AI Congress is an innovative system where autonomous LLM agents (powered by Ollama) collaborate and vote on responses using weighted ensemble decision-making algorithms.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Ollama installed and running
- Node.js 18+ (for frontend)

### Configuration
The system loads configuration from `config/config.yaml`. For remote Ollama deployments (e.g., on a GPU server), update the `ollama.base_url` in this file.

#### Example for Remote Ollama
Edit `config/config.yaml`:
```yaml
ollama:
  base_url: "http://your-gpu-server:11434"  # Replace with your Ollama URL
```

#### Environment Variable Override
You can override the Ollama URL at runtime by setting the `OLLAMA_BASE_URL` environment variable:
```bash
export OLLAMA_BASE_URL="http://192.168.1.100:11434"
```

This is useful for Docker deployments or different environments.

### Server Setup (for Remote Ollama)
If you're using a remote Ollama server (e.g., with GPU), set up models on the server first:

1. **On your Ollama server**: Run the model management script to pull preferred models:
   ```bash
   ./scripts/ollama_server.sh
   ```
   This will pull models listed in `config/config.yaml` under `models.preferred` (e.g., phi3:3.8b, mistral:7b, llama3.2:3b).

2. **Verify models**: Check that models are available:
   ```bash
   ollama list
   ```

3. **Configure client**: Ensure `config/config.yaml` has the correct `ollama.base_url` pointing to your server.

### Local Setup (3 Steps)
1. **Install Dependencies**: `./startup.sh` (installs Python deps, sets up frontend, pulls models if local Ollama)
2. **Start Backend**: `uvicorn src.ai_congress.api.main:app --reload` (runs API on :8000)
3. **Start Frontend**: `cd frontend && npm run dev` (runs UI on :3000)

> **Note**: The frontend now features a beautiful Open WebUI-inspired interface with:
> - 🗳️ Real-time vote breakdown visualization
> - 🤖 Individual model response cards
> - 🌓 Dark mode support
> - 📊 Confidence meters
> - ✨ Smooth animations

### Docker Setup (1 Step)
1. **Run Everything**: `docker-compose up -d` (starts API :8000, frontend :3000)

## 💡 Usage
- **CLI**: `python -m src.ai_congress.cli.main chat "Your prompt"`
- **Web UI**: Visit http://localhost:3000 (beautiful interface with swarm visualizations!)
- **API**: POST to http://localhost:8000/api/chat
- **API Docs**: Interactive API docs at http://localhost:8000/docs

### Web UI Features
The modern web interface provides:
- **Two Chat Modes**: Choose between "Model Swarm" and "Personality Swarm"
- **Model Selection**: Choose which LLMs participate in the swarm
- **Personality Creation**: Create custom AI personalities with unique system prompts
- **Swarm Modes**: Multi-Model, Multi-Request, or Hybrid
- **Vote Visualization**: See how each model contributes to the final answer
- **Response Details**: Expandable view showing individual model responses
- **Dual Confidence Scoring**: Both string-based and semantic agreement confidence
- **Semantic Warnings**: Alerts when responses agree in voting but differ semantically
- **Dark Mode**: Easy on the eyes for late-night AI conversations

### New Features

#### 🎭 Personality Swarm Mode
- **Create Custom Personalities**: Define AI agents with specific personalities (e.g., "Donald Trump", "Albert Einstein")
- **Base Model Configuration**: Configure which model to use as the base for all personalities (default: deepseek-r1)
- **Predefined Personalities**: Comes with 6 built-in personalities including Trump, Biden, Einstein, and more
- **Personality Voting**: All personalities vote using their unique perspectives
- **Pure Roleplay**: Enhanced prompt engineering ensures responses are purely in-character, with no internal reasoning or meta-commentary leaking through

#### 🧠 Semantic Confidence Threshold
- **Dual Confidence Metrics**: Shows both string-based confidence (exact matching) and semantic confidence (meaning agreement)
- **LLM-Powered Evaluation**: Uses a separate summarizer model (phi3) to assess semantic agreement
- **Low Agreement Warnings**: Yellow warning banner when semantic confidence is below 60%
- **Better Accuracy**: Catches cases where models agree on words but not meaning (e.g., "2+2=4" responses)

#### 🏛️ Enhanced Voting System
- **Weighted Majority Vote**: Models vote based on their MMLU benchmark performance scores
- **Semantic Validation**: Ensures voted-upon answers actually mean the same thing
- **Response Aggregation**: Intelligent combination of diverse model outputs

## 🧪 Testing
Run tests: `python -m pytest tests/ -v`
