from .base import BaseReasoningAgent
from typing import AsyncGenerator

class CoTAgent(BaseReasoningAgent):
    def __init__(self, client, model):
        super().__init__(client, model)
        self.name = "CoTAgent"

    async def run(self, query: str) -> str:
        prompt = f"Answer the following question. Think step-by-step. Break down the reasoning process before giving the final answer.\n\nQuestion: {query}"
        response = await self.client.chat(self.model, [{'role': 'user', 'content': prompt}])
        return response['message']['content']

    async def stream(self, query: str) -> AsyncGenerator[str, None]:
        prompt = f"Answer the following question. Think step-by-step. Break down the reasoning process before giving the final answer.\n\nQuestion: {query}"
        async for chunk in await self.client.chat(self.model, [{'role': 'user', 'content': prompt}], stream=True):
             yield chunk['message']['content']
