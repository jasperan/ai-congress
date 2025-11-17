# Quick Start Guide - New UI

## Getting Started in 3 Minutes ⚡

### Prerequisites Check
```bash
# Check if Ollama is running
ollama list

# Check if Python backend is running
curl http://localhost:8000/health
```

### Step 1: Start Backend (if not running)
```bash
# From project root
uvicorn src.ai_congress.api.main:app --reload
```

Wait for: `Application startup complete`

### Step 2: Start Frontend
```bash
# From project root
cd frontend
npm run dev
```

Wait for: `Local: http://localhost:3000/`

### Step 3: Open Browser
Navigate to: **http://localhost:3000**

---

## First Time Setup

If you see "No Models Available":

```bash
# Pull recommended models (takes 5-10 minutes)
ollama pull phi3:3.8b      # ~2.5GB
ollama pull mistral:7b     # ~4GB
ollama pull llama3.2:3b    # ~3GB
```

Then refresh the page!

---

## Interface Tour 🎯

### Navigation Bar (Top)
- **🏛️ AI Congress**: Logo and title
- **Stats Badges**: Shows available and selected models
- **🌓 Dark Mode Toggle**: Switch themes
- **GitHub Icon**: Link to repository

### Model Selection Panel
1. **Model Checkboxes**: Click to select/deselect models
   - Shows model name and accuracy weight (e.g., 85%)
   - Selected models have blue border
   - At least 1 model required

2. **Swarm Mode Dropdown**: Choose voting strategy
   - 🔄 **Multi-Model**: Different models vote (recommended)
   - 🌡️ **Multi-Request**: Same model, different temperatures
   - ⚡ **Hybrid**: Both approaches combined

### Chat Area
- **Messages**: User messages (blue, right) | AI responses (gray, left)
- **Timestamps**: When each message was sent
- **Confidence Badge**: Shows consensus strength (e.g., 85.3%)
- **"View Details" Button**: Click to see vote breakdown

### Input Box (Bottom)
- **Type Message**: Enter your question
- **Shift+Enter**: New line
- **Enter**: Send message
- **Send Button**: Click or press Enter

---

## Using the Swarm 🐝

### Example 1: Multi-Model Vote
1. Select: `phi3:3.8b`, `mistral:7b`, `llama3.2:3b`
2. Mode: `Multi-Model`
3. Ask: "Explain quantum entanglement"
4. **Result**: All 3 models answer, system picks best response
5. Click **"View Details"** to see:
   - Vote breakdown (which model won)
   - Each model's individual response
   - Confidence score

### Example 2: Temperature Variation
1. Select: `mistral:7b`
2. Mode: `Multi-Request`
3. Ask: "Write a creative story opening"
4. **Result**: Same model responds with different creativity levels
   - Low temp (0.3): Conservative, factual
   - Med temp (0.7): Balanced
   - High temp (1.0): Creative, diverse

### Example 3: Maximum Diversity
1. Select: 2-3 models
2. Mode: `Hybrid`
3. Ask: "What's the meaning of life?"
4. **Result**: Multiple models × multiple temperatures = most diverse answers

---

## Understanding Vote Breakdown 🗳️

When you click "View Details", you see:

### Confidence Meter
```
████████░░ 85.3% High Confidence
```
- **High (80%+)**: Strong consensus, reliable answer
- **Medium (60-80%)**: Good agreement
- **Low (<60%)**: Models disagree, answer uncertain

### Model Votes
```
🤖 phi3:3.8b     [████████░░] 85%
🤖 mistral:7b    [███████░░░] 82%
🤖 llama3.2:3b   [██████░░░░] 78%
```
Each bar shows:
- Model name
- Vote weight (based on accuracy)
- Visual bar (how much it influenced final answer)

### Individual Responses
Each model's full response in expandable cards:
- ✅ Success/Error badge
- 🌡️ Temperature used
- 📝 Full response text
- 📊 Metadata (length, weight)

---

## Tips & Tricks 💡

### For Best Results
- ✅ Select 2-3 models for balanced speed/accuracy
- ✅ Use Multi-Model for factual questions
- ✅ Use Multi-Request for creative tasks
- ✅ Use Hybrid when you want maximum confidence

### Keyboard Shortcuts
- `Enter`: Send message
- `Shift+Enter`: New line in message
- `Esc`: Close details panel (when open)

### Dark Mode
- Click 🌓 icon in navigation
- Automatically detects system preference
- Persists across sessions

### Model Selection
- **Minimum**: 1 model
- **Recommended**: 2-3 models
- **Maximum**: All available (slower but more confident)

---

## Troubleshooting 🔧

### "Failed to Connect"
**Problem**: Can't reach API
**Solution**:
```bash
# Check backend is running
curl http://localhost:8000/health

# Restart backend
uvicorn src.ai_congress.api.main:app --reload
```

### "No Models Available"
**Problem**: Ollama has no models
**Solution**:
```bash
# Pull at least one model
ollama pull phi3:3.8b

# Refresh browser
```

### Styles Look Broken
**Problem**: CSS not loading
**Solution**:
```bash
cd frontend
rm -rf node_modules
npm install
npm run dev
```

### Response Takes Too Long
**Causes**:
- Too many models selected (try 2-3 max)
- Large models running on CPU (use smaller models)
- Hybrid mode (most comprehensive, slowest)

**Solutions**:
- Use fewer, smaller models
- Switch to Multi-Model mode
- Consider GPU acceleration for Ollama

---

## Common Questions ❓

**Q: How does voting work?**
A: Each model responds, votes are weighted by accuracy (from benchmarks), highest-weighted response wins.

**Q: Can I add my own models?**
A: Yes! Pull any Ollama model, refresh page, it will appear in selector.

**Q: What's the difference between modes?**
- **Multi-Model**: Different LLMs → diverse perspectives
- **Multi-Request**: Same LLM, varied temp → consistency vs creativity
- **Hybrid**: Both → maximum diversity & confidence

**Q: Why are some models weighted higher?**
A: Weights come from benchmark scores (MMLU, HumanEval). Higher accuracy = higher weight.

**Q: Can I customize model weights?**
A: Yes! Edit `config/models_benchmark.json` and restart backend.

---

## Next Steps 🚀

### Explore
- Try different model combinations
- Compare responses in different modes
- Test with various question types

### Customize
- Edit `config/config.yaml` for preferred models
- Adjust weights in `config/models_benchmark.json`
- Modify temperatures in backend code

### Extend
- Check `docs/INTERFACE_UPGRADE.md` for technical details
- Read API docs at http://localhost:8000/docs
- Explore CLI with `python -m src.ai_congress.cli.main --help`

---

## Visual Guide 📸

### Main Interface
```
┌─────────────────────────────────────────────────┐
│ 🏛️ AI Congress      [2 Available] [2 Selected] │
│                           🌓  [GitHub]         │
├─────────────────────────────────────────────────┤
│ Models: [✓ phi3] [✓ mistral] [ llama3.2]      │
│ Mode: [Multi-Model ▼]                          │
├─────────────────────────────────────────────────┤
│                                                 │
│              Welcome to AI Congress             │
│                      🏛️                         │
│   Select models and start chatting...          │
│                                                 │
├─────────────────────────────────────────────────┤
│ [Type message...                    ] [Send]   │
└─────────────────────────────────────────────────┘
```

### Chat in Progress
```
┌─────────────────────────────────────────────────┐
│                              ┌────────────────┐ │
│                              │ Your question? │ │
│                              └────────────────┘ │
│                                        3:45 PM  │
│                                                 │
│ ┌─────────────────────────────────────────┐    │
│ │ 🏛️ Here's the consensus answer...       │    │
│ │                                          │    │
│ │ [Response text]                          │    │
│ └─────────────────────────────────────────┘    │
│ 3:45 PM  [View Details]  85.3% confidence      │
└─────────────────────────────────────────────────┘
```

### Details Panel
```
┌─────────────────────────────────────┐
│ Response Details              [×]   │
├─────────────────────────────────────┤
│ 🗳️ Vote Breakdown                  │
│ Confidence: High (85.3%)            │
│ ████████░░                          │
│                                     │
│ phi3:3.8b     [████████░░] 85%     │
│ mistral:7b    [███████░░░] 82%     │
├─────────────────────────────────────┤
│ Individual Model Responses          │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │ 🤖 phi3:3.8b        ✓ Success  │ │
│ │ [Full response text...]         │ │
│ └─────────────────────────────────┘ │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │ 🤖 mistral:7b       ✓ Success  │ │
│ │ [Full response text...]         │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

---

**Happy Chatting! 🎉**

For more details, see:
- Technical docs: `docs/INTERFACE_UPGRADE.md`
- API docs: http://localhost:8000/docs
- Original design: https://github.com/open-webui/open-webui

