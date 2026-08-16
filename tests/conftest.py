"""
Auto-classify tests: anything not explicitly marked ``integration`` is
considered a ``unit`` test. The suite degrades gracefully without Oracle
or Ollama, so unmarked tests are safe to run in hermetic CI.
"""
import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "integration" not in item.keywords:
            item.add_marker(pytest.mark.unit)
