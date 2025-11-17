#!/bin/bash

echo "🤖 Ollama Server Model Management Script"
echo "========================================="

# Function to pull a model if not already available
pull_model() {
    local model=$1
    echo "Checking model: $model"

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

# Read preferred models from config
echo "📖 Reading preferred models from config/config.yaml..."

# Extract models using Python (more reliable than bash YAML parsing)
PREFERRED_MODELS=$(python -c "
import yaml
try:
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    models = config['models']['preferred']
    print(' '.join(models))
except Exception as e:
    print('phi3:3.8b mistral:7b llama3.2:3b')  # Fallback
")

echo "🎯 Preferred models: $PREFERRED_MODELS"

# Pull each preferred model
for model in $PREFERRED_MODELS; do
    pull_model "$model"
done

echo ""
echo "✅ Model management complete!"
echo "📋 Available models:"
ollama list

echo ""
echo "💡 To add more models, edit config/config.yaml under models.preferred"
echo "💡 To pull specific models: ollama pull <model_name>"
