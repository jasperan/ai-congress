"""
Auto-classify tests: anything not explicitly marked ``integration`` is
considered a ``unit`` test. The suite degrades gracefully without Oracle
or Ollama, so unmarked tests are safe to run in hermetic CI.

Also hosts the hermetic mock-Ollama client (4.6.3): a fake async client
that returns canned responses for success / timeout / error shapes so
integration-adjacent code can be exercised without Ollama running.
"""
import asyncio
import pytest


class MockOllamaClient:
    """Fake async Ollama client with scriptable behavior (4.6.3).

    ``responses`` maps either a prompt substring (``contains``) or the
    special key ``__default__`` to a canned response dict. For timeout
    simulation, a response value of ``MockOllamaClient.TIMEOUT`` raises
    ``asyncio.TimeoutError``. For failures raise the given exception.
    """

    TIMEOUT = object()

    def __init__(self, responses=None, default_response=None):
        self.responses = dict(responses or {})
        self.default_response = default_response or {"response": "mock answer", "model": "mock"}
        self.calls = []
        self.failures = []
        self.delay = 0.0

    async def generate(self, model=None, prompt=None, options=None, stream=False, temperature=None):
        self.calls.append({"model": model, "prompt": prompt, "options": options or {}, "temperature": temperature})
        if self.delay:
            await asyncio.sleep(self.delay)
        for needle, canned in self.responses.items():
            if needle == "__default__":
                continue
            if needle in (prompt or ""):
                if canned is MockOllamaClient.TIMEOUT:
                    raise asyncio.TimeoutError(f"mock timeout for {model}")
                if isinstance(canned, Exception):
                    raise canned
                if callable(canned):
                    return canned(model=model, prompt=prompt)
                return dict(canned)
        default = self.responses.get("__default__", self.default_response)
        if default is MockOllamaClient.TIMEOUT:
            raise asyncio.TimeoutError(f"mock timeout for {model}")
        if isinstance(default, Exception):
            raise default
        if callable(default):
            return default(model=model, prompt=prompt)
        return dict(default)

    async def list_models(self):
        return [{"name": "mock-model", "size": 1024**3, "modified_at": "2026-01-01T00:00:00"}]

    async def list(self):
        return await self.list_models()

    async def get_model_info(self, model_name):
        return {"name": model_name, "size": 1024**3}

    def fail_next(self, error_message="mock failure"):
        """Make the next call raise an exception (simulates Ollama down)."""
        self.failures.append(Exception(error_message))

    async def generate_wrapper(self, *args, **kwargs):
        if self.failures:
            raise self.failures.pop(0)
        return await self.generate(*args, **kwargs)


@pytest.fixture
def mock_ollama_client():
    """Hermetic async Ollama client for unit tests (4.6.3)."""
    return MockOllamaClient()


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "integration" not in item.keywords:
            item.add_marker(pytest.mark.unit)
