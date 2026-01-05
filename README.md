![Logo](img/logo.jpg)

# AI Congress

![Status](https://img.shields.io/badge/Status-Active-success)
![Reasoning](https://img.shields.io/badge/Reasoning-CoT%20%7C%20ReAct-blue)
![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green)

AI Congress is an innovative system where autonomous LLM agents (powered by **Ollama**) collaborate and vote on responses using weighted ensemble decision-making algorithms.

Now featuring **Advanced Reasoning Layers** (Chain-of-Thought, ReAct) to solve complex problems with higher accuracy.

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11+
- Ollama installed and running (`ollama serve`)
- Node.js 18+ (for frontend)

### 2. Installation

```bash
# Clone and setup
./startup.sh
```

### 3. Usage

#### 🖥️ CLI (Command Line Interface)
The quickest way to interact with the swarm. Use the newly added `run_cli.py` script:

```bash
# Basic Chat
./run_cli.py chat "Should we regulate AI?"

# Enable Reasoning (Chain-of-Thought)
./run_cli.py chat "Solve this logic puzzle..." --reasoning cot

# Enable Reasoning (ReAct with Tools)
./run_cli.py chat "Calculate 25 * 48" --reasoning react

# Use specific models
./run_cli.py chat "Hello" -m gemma2 -m llama3
```

#### 🌐 Web Interface
Launch the full stack with beautiful UI:

```bash
# Start backend
python run_server.py
# In another terminal, start frontend
cd frontend && npm run dev
```
Visit `http://localhost:3000`

## 🧠 Features

### Swarm Intelligence
- **Multi-Model**: Query multiple different models concurrently.
- **Weighted Voting**: Models vote based on their benchmark performance.
- **Semantic Consensus**: Ensures voted-upon answers actually mean the same thing.

### Advanced Reasoning
- **Chain of Thought (CoT)**: Agents think step-by-step before answering.
- **ReAct**: Agents can use tools (Calculation, Search) to answer queries.

### Personalities
- **Roleplay**: interact with predefined personalities like Donald Trump, Joe Biden, Einstein, and more.
- **Custom**: Create your own personalities via the API or Config.

## ⚙️ Configuration
The system loads configuration from `config/config.yaml`. For remote Ollama deployments (e.g., on a GPU server), update the `ollama.base_url` in this file.

### Example for Remote Ollama
Edit `config/config.yaml`:
```yaml
ollama:
  base_url: "http://your-gpu-server:11434"  # Replace with your Ollama URL
```

## 🐳 Docker Setup
1. **Run Everything**: `docker-compose up -d` (starts API :8000, frontend :3000)

## 🧪 Testing
Run tests: `python -m pytest tests/ -v`
