"""
Ollama Client - Wrapper for Ollama API interactions
"""
import asyncio
from typing import Dict, List, Optional, Any
import ollama
import logging

logger = logging.getLogger(__name__)


class OllamaClient:
    """Wrapper for Ollama API client with error handling and retries"""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 120, max_retries: int = 3):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = ollama.AsyncClient(host=base_url)

    async def list_models(self) -> List[Dict]:
        """List available models"""
        try:
            response = await self.client.list()
            return response.get('models', [])
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama library"""
        try:
            async for progress in await self.client.pull(model_name, stream=True):
                status = progress.get('status', '')
                logger.debug(f"Pull progress for {model_name}: {status}")
                if 'completed' in status.lower():
                    return True
            return True
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False

    async def chat(
        self,
        model: str,
        messages: List[Dict],
        options: Optional[Dict] = None,
        stream: bool = False
    ) -> Dict:
        """Send chat request to model"""
        try:
            response = await self.client.chat(
                model=model,
                messages=messages,
                options=options or {},
                stream=stream
            )
            return response
        except Exception as e:
            logger.error(f"Error in chat with {model}: {e}")
            return {'message': {'content': ''}, 'error': str(e)}

    async def generate(
        self,
        model: str,
        prompt: str,
        options: Optional[Dict] = None,
        stream: bool = False
    ) -> Dict:
        """Generate text with model"""
        try:
            response = await self.client.generate(
                model=model,
                prompt=prompt,
                options=options or {},
                stream=stream
            )
            return response
        except Exception as e:
            logger.error(f"Error generating with {model}: {e}")
            return {'response': '', 'error': str(e)}

    async def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get detailed model information"""
        try:
            models = await self.list_models()
            for model in models:
                if model['name'] == model_name:
                    return model
            return None
        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            return None

    async def health_check(self) -> bool:
        """Check if Ollama service is running"""
        try:
            await self.client.list()
            return True
        except Exception:
            return False
