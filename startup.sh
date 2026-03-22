#!/bin/bash

echo "🚀 Starting AI Congress Setup..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt
pip install -e .

# Install frontend dependencies
echo "🎨 Installing frontend dependencies..."
cd frontend && npm install && cd ..

# Function to pull a model if not already available
pull_model() {
    local model=$1
    # Check if model is already available
    if ollama list | grep -q "$model"; then
        echo "✅ Model $model is already available"
    else
        echo "📥 Pulling model: $model"
        ollama pull "$model"
        if [ $? -eq 0 ]; then
            echo "✅ Successfully pulled $model"
        else
            echo "❌ Failed to pull $model"
        fi
    fi
}

# Function to download Whisper model for faster-whisper
setup_whisper() {
    echo "🎤 Setting up Whisper model for voice transcription..."
    # The model will be auto-downloaded on first use by faster-whisper
    # We'll trigger a test download here
    python3 -c "
try:
    from faster_whisper import WhisperModel
    print('Downloading Whisper base model...')
    model = WhisperModel('base', device='cpu', compute_type='int8')
    print('✅ Whisper model ready')
except Exception as e:
    print(f'⚠️  Whisper model will be downloaded on first use: {e}')
" || echo "⚠️  Whisper model will be downloaded on first use"
}

# Check Ollama configuration
OLLAMA_URL=$(python -c "import yaml; config = yaml.safe_load(open('config/config.yaml')); print(config['ollama']['base_url'])")
if [[ $OLLAMA_URL == *"localhost"* ]] || [[ $OLLAMA_URL == *"127.0.0.1"* ]]; then
    echo "🏠 Local Ollama detected. Pulling models..."
    pull_model "phi3:3.8b"
    pull_model "mistral:7b"
    pull_model "llama3.2:3b"
    echo "🆕 Pulling lightweight versions of new models..."
    pull_model "deepseek-r1:1.5b"
    pull_model "qwen3:0.6b"

    echo "🎨 Stable Diffusion will be downloaded from Hugging Face on first use..."

    # Setup Whisper for voice input
    setup_whisper
else
    echo "🌐 Remote Ollama detected at $OLLAMA_URL"
    echo "⚠️  If this is your first time, ensure models are pulled on the server:"
    echo "    Run scripts/ollama_server.sh on your Ollama server."
fi

echo "✅ Setup complete! Run the following:"
echo "  Backend: python run_server.py   (verbose uvicorn at :8100)"
echo "  Backend (simple): uvicorn src.ai_congress.api.main:app --reload --port 8100"
echo "  Frontend: cd frontend && npm run dev"
