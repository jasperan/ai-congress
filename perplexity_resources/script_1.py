
# Create detailed Cline workspace files with code examples and templates

# 1. Create requirements.txt content
requirements_txt = """# Core Dependencies
fastapi==0.115.0
uvicorn[standard]==0.30.6
websockets==13.1
pydantic==2.9.2
pydantic-settings==2.5.2

# Ollama Integration
ollama==0.3.3

# Async & Concurrency
asyncio-mqtt==0.16.2
aiohttp==3.10.5
aiocache==0.12.2

# CLI & Formatting
rich==13.8.1
click==8.1.7
typer==0.12.5

# Database
sqlalchemy==2.0.35
alembic==1.13.3
aiosqlite==0.20.0

# Configuration
pyyaml==6.0.2
python-dotenv==1.0.1

# Testing
pytest==8.3.3
pytest-asyncio==0.24.0
httpx==0.27.2
pytest-cov==5.0.0

# Utilities
python-multipart==0.0.12
"""

with open('requirements.txt', 'w') as f:
    f.write(requirements_txt)

# 2. Create config.yaml template
config_yaml = """# AI Congress Configuration

# Ollama Settings
ollama:
  base_url: "http://localhost:11434"
  timeout: 120
  max_retries: 3

# Swarm Settings
swarm:
  # Swarm modes: "multi_model", "multi_request", "hybrid"
  default_mode: "multi_model"
  
  # Multi-request settings (temperature sampling)
  multi_request:
    num_requests: 5
    temperature_range: [0.3, 0.7, 1.0, 1.2]
  
  # Concurrency settings
  max_concurrent_requests: 10
  request_timeout: 60

# Voting Configuration
voting:
  # Voting algorithms: "weighted_majority", "majority", "confidence_based", "boosting"
  default_algorithm: "weighted_majority"
  
  # Minimum agreement threshold (0.0 - 1.0)
  consensus_threshold: 0.6
  
  # Enable similarity clustering for responses
  enable_clustering: true

# Model Configuration
models:
  # Preferred lightweight models (priority order)
  preferred:
    - "phi3:3.8b"
    - "mistral:7b"
    - "llama3.2:3b"
    - "gemma2:2b"
    - "qwen2.5:7b"
  
  # Model performance weights (can be auto-updated from benchmarks)
  weights:
    "phi3:3.8b": 0.85
    "mistral:7b": 0.82
    "llama3.2:3b": 0.78
    "gemma2:2b": 0.75
    "qwen2.5:7b": 0.84

# API Settings
api:
  host: "0.0.0.0"
  port: 8000
  reload: true
  workers: 4
  cors_origins:
    - "http://localhost:3000"
    - "http://localhost:5173"

# CLI Settings
cli:
  # Verbosity levels: 0=minimal, 1=normal, 2=verbose, 3=debug
  default_verbosity: 2
  
  # Rich formatting
  use_colors: true
  panel_style: "bold cyan"
  
  # Auto-save conversations
  save_history: true
  history_file: "~/.ai_congress_history"

# Database
database:
  url: "sqlite+aiosqlite:///./ai_congress.db"
  echo: false

# Logging
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/ai_congress.log"
"""

with open('config.yaml', 'w') as f:
    f.write(config_yaml)

# 3. Create sample model_registry.py with detailed implementation
model_registry_code = '''"""
Model Registry - Manages Ollama models and their performance metrics
"""
import asyncio
from typing import List, Dict, Optional
import ollama
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for managing Ollama models and their performance data"""
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self.ollama_client = ollama.AsyncClient(host=ollama_base_url)
        self.models_cache: Dict[str, Dict] = {}
        self.weights: Dict[str, float] = {}
    
    async def list_available_models(self) -> List[Dict]:
        """List all available Ollama models"""
        try:
            models_response = await self.ollama_client.list()
            models = []
            
            for model in models_response.get('models', []):
                model_info = {
                    'name': model['name'],
                    'size': model.get('size', 0),
                    'modified_at': model.get('modified_at'),
                    'digest': model.get('digest')
                }
                models.append(model_info)
                self.models_cache[model['name']] = model_info
            
            logger.info(f"Found {len(models)} available models")
            return models
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get detailed information about a specific model"""
        try:
            if model_name in self.models_cache:
                return self.models_cache[model_name]
            
            # Fetch from Ollama
            await self.list_available_models()
            return self.models_cache.get(model_name)
            
        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            return None
    
    def set_model_weight(self, model_name: str, weight: float):
        """Set performance weight for a model (0.0 - 1.0)"""
        self.weights[model_name] = max(0.0, min(1.0, weight))
        logger.info(f"Set weight for {model_name}: {weight:.2f}")
    
    def get_model_weight(self, model_name: str) -> float:
        """Get performance weight for a model"""
        return self.weights.get(model_name, 0.5)  # Default to 0.5
    
    async def load_benchmark_weights(self, benchmark_file: str):
        """Load model weights from benchmark file"""
        import json
        try:
            with open(benchmark_file, 'r') as f:
                benchmarks = json.load(f)
                
            for model_name, data in benchmarks.items():
                # Normalize accuracy to 0-1 range
                accuracy = data.get('accuracy', 0.5)
                self.set_model_weight(model_name, accuracy)
                
            logger.info(f"Loaded benchmark weights for {len(self.weights)} models")
            
        except Exception as e:
            logger.warning(f"Could not load benchmark file: {e}")
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama library"""
        try:
            logger.info(f"Pulling model: {model_name}")
            
            async for progress in await self.ollama_client.pull(model_name, stream=True):
                status = progress.get('status', '')
                if 'completed' in status.lower():
                    logger.info(f"Model {model_name} pulled successfully")
                    
            await self.list_available_models()  # Refresh cache
            return True
            
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    def filter_lightweight_models(self, models: List[Dict], max_size_gb: float = 10.0) -> List[Dict]:
        """Filter models by size to get lightweight options"""
        max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        return [m for m in models if m.get('size', 0) <= max_size_bytes]
    
    def get_top_models(self, n: int = 5) -> List[str]:
        """Get top N models by weight"""
        sorted_models = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        return [model for model, _ in sorted_models[:n]]
'''

with open('model_registry_example.py', 'w') as f:
    f.write(model_registry_code)

# 4. Create sample voting_engine.py
voting_engine_code = '''"""
Voting Engine - Implements ensemble decision-making algorithms
"""
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VotingEngine:
    """Ensemble voting algorithms for LLM swarm decisions"""
    
    def __init__(self):
        self.voting_history = []
    
    def weighted_majority_vote(
        self, 
        responses: List[str], 
        weights: List[float],
        model_names: List[str] = None
    ) -> Tuple[str, float, Dict]:
        """
        Weighted majority voting - each response weighted by model performance
        
        Args:
            responses: List of model responses
            weights: List of model weights (same length as responses)
            model_names: Optional list of model names for tracking
            
        Returns:
            (winning_response, confidence_score, vote_breakdown)
        """
        if len(responses) != len(weights):
            raise ValueError("Responses and weights must have same length")
        
        # Group responses and sum weights
        response_weights = {}
        vote_details = {}
        
        for i, response in enumerate(responses):
            # Normalize response for comparison
            normalized = response.strip().lower()
            
            if normalized not in response_weights:
                response_weights[normalized] = 0
                vote_details[normalized] = {
                    'original': response,
                    'weight': 0,
                    'votes': [],
                    'models': []
                }
            
            response_weights[normalized] += weights[i]
            vote_details[normalized]['weight'] += weights[i]
            vote_details[normalized]['votes'].append(weights[i])
            
            if model_names and i < len(model_names):
                vote_details[normalized]['models'].append(model_names[i])
        
        # Find winner
        winner = max(response_weights.items(), key=lambda x: x[1])
        winning_response = vote_details[winner[0]]['original']
        total_weight = sum(weights)
        confidence = winner[1] / total_weight if total_weight > 0 else 0
        
        logger.info(f"Weighted vote winner: {winning_response[:50]}... (confidence: {confidence:.2f})")
        
        return winning_response, confidence, vote_details
    
    def majority_vote(
        self, 
        responses: List[str],
        model_names: List[str] = None
    ) -> Tuple[str, float, Dict]:
        """Simple majority voting - all models have equal weight"""
        equal_weights = [1.0] * len(responses)
        return self.weighted_majority_vote(responses, equal_weights, model_names)
    
    def confidence_based_vote(
        self,
        responses: List[Dict],  # [{'text': str, 'confidence': float, 'model': str}]
    ) -> Tuple[str, float, Dict]:
        """Vote based on model confidence scores"""
        texts = [r['text'] for r in responses]
        confidences = [r.get('confidence', 0.5) for r in responses]
        models = [r.get('model', f'model_{i}') for i, r in enumerate(responses)]
        
        return self.weighted_majority_vote(texts, confidences, models)
    
    def rank_responses(
        self,
        responses: List[str],
        weights: List[float],
        model_names: List[str] = None
    ) -> List[Dict]:
        """
        Rank all unique responses by their weighted votes
        
        Returns:
            List of dicts with response, total_weight, models, rank
        """
        _, _, vote_details = self.weighted_majority_vote(responses, weights, model_names)
        
        ranked = sorted(
            vote_details.values(),
            key=lambda x: x['weight'],
            reverse=True
        )
        
        for i, item in enumerate(ranked):
            item['rank'] = i + 1
        
        return ranked
    
    def calculate_consensus_score(
        self,
        responses: List[str],
        weights: List[float]
    ) -> float:
        """
        Calculate consensus score (0-1) based on agreement among models
        
        Higher score = more agreement
        """
        _, confidence, _ = self.weighted_majority_vote(responses, weights)
        return confidence
    
    def temperature_ensemble(
        self,
        responses: List[str],
        temperatures: List[float]
    ) -> str:
        """
        Ensemble responses from same model at different temperatures
        
        Lower temperatures get higher weight
        """
        # Inverse temperature as weight (lower temp = more confident)
        weights = [1.0 / (t + 0.1) for t in temperatures]
        
        winner, _, _ = self.weighted_majority_vote(responses, weights)
        return winner
'''

with open('voting_engine_example.py', 'w') as f:
    f.write(voting_engine_code)

print("✅ Created requirements.txt")
print("✅ Created config.yaml")
print("✅ Created model_registry_example.py")
print("✅ Created voting_engine_example.py")
print("\n📝 Files ready for Cline to implement!")
