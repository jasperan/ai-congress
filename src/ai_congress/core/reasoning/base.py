from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, List, Optional
from ..ollama_client import OllamaClient

class BaseReasoningAgent(ABC):
    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model
        self.name = "BaseReasoningAgent"

    @abstractmethod
    async def run(self, query: str) -> str:
        """Run the full reasoning process and return the final answer"""
        pass

    @abstractmethod
    async def stream(self, query: str) -> AsyncGenerator[str, None]:
        """Stream the reasoning process"""
        pass
