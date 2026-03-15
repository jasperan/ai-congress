"""
OpenAI Client - Wrapper for OpenAI-compatible API interactions.

Mirrors the OllamaClient interface so the SwarmOrchestrator can swap
between Ollama and OpenAI backends transparently.
"""
import asyncio
from typing import Dict, List, Optional, Any
import logging

from ..utils.logger import debug_action, error_message, truncate_text
from ..utils.config_loader import load_config

logger = logging.getLogger(__name__)
config = load_config()


class OpenAIClient:
    """Wrapper for OpenAI-compatible APIs (OCA / LiteLLM / OpenAI direct)."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str = "gpt-5.4",
        timeout: int = 120,
        max_retries: int = 3,
    ):
        from openai import AsyncOpenAI

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=api_key or "not-needed",
            timeout=timeout,
        )

    async def chat(
        self,
        model: str,
        messages: List[Dict],
        options: Optional[Dict] = None,
        stream: bool = False,
    ) -> Dict:
        """Send chat request, matching OllamaClient.chat signature."""
        options = options or {}
        temperature = options.get("temperature", 0.7)
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                prompt_preview = truncate_text(
                    messages[-1]["content"] if messages else "", 50
                )
                debug_action(
                    "OPENAI_CALL",
                    model,
                    f"Initiating chat (attempt {attempt}): {prompt_preview}...",
                    config.logging.verbosity,
                )

                if stream:
                    return self._stream_generator(model, messages, temperature)

                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )

                content = response.choices[0].message.content or ""
                debug_action(
                    "OPENAI_RESPONSE",
                    model,
                    f"Generated: {truncate_text(content, 80)}",
                    config.logging.verbosity,
                )

                # Return in Ollama-compatible dict shape
                return {"message": {"content": content}}

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    backoff = 2 ** (attempt - 1)
                    logger.warning(
                        f"Chat attempt {attempt}/{self.max_retries} failed for {model}: {e}. "
                        f"Retrying in {backoff}s..."
                    )
                    await asyncio.sleep(backoff)
                else:
                    error_message(
                        "OPENAI_ERROR",
                        model,
                        f"Chat failed after {self.max_retries} attempts: {str(e)}",
                    )

        return {"message": {"content": ""}, "error": str(last_error)}

    async def _stream_generator(self, model, messages, temperature):
        """Return an async generator that yields Ollama-shaped chunks."""
        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )

        async def _chunks():
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield {"message": {"content": delta.content}}

        return _chunks()

    async def health_check(self) -> bool:
        """Check if the OpenAI-compatible endpoint is reachable."""
        try:
            # A lightweight models list call
            await self.client.models.list()
            return True
        except Exception:
            return False
