#!/bin/bash

echo "🚀 Starting AI Congress Setup..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Install frontend dependencies
echo "🎨 Installing frontend dependencies..."
cd frontend && npm install && cd ..

# Check Ollama configuration
OLLAMA_URL=$(python -c "import yaml; config = yaml.safe_load(open('config/config.yaml')); print(config['ollama']['base_url'])")
if [[ $OLLAMA_URL == *"localhost"* ]] || [[ $OLLAMA_URL == *"127.0.0.1"* ]]; then
    echo "🏠 Local Ollama detected. Pulling models..."
    ollama pull phi3:3.8b
    ollama pull mistral:7b
    ollama pull llama3.2:3b
else
    echo "🌐 Remote Ollama detected at $OLLAMA_URL"
    echo "⚠️  If this is your first time, ensure models are pulled on the server:"
    echo "    Run scripts/ollama_server.sh on your Ollama server."
fi

echo "✅ Setup complete! Run the following:"
echo "  Backend: uvicorn src.ai_congress.api.main:app --reload"
echo "  Frontend: cd frontend && npm run dev"
