"""Small JSON persistence helpers shared by the learning layer (3.5.2 / 3.4.3).

Atomic writes (tmp file + rename) so a crash mid-write cannot corrupt a state
file; parent directories are created on demand. All persistence degrades to a
no-op/logged-warning when the filesystem is unwritable, matching the project's
graceful-degradation convention.
"""

import json
import logging
import os
import tempfile
from typing import Any

logger = logging.getLogger(__name__)


def load_json(path: str, default: Any = None) -> Any:
    """Load a JSON file, returning ``default`` when missing/unreadable."""
    if not path:
        return default
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, ValueError) as e:
        logger.warning("Could not load %s: %s", path, e)
        return default


def save_json(path: str, data: Any) -> bool:
    """Atomically write ``data`` as JSON to ``path``. Returns success."""
    if not path:
        return False
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            suffix=".tmp", prefix=os.path.basename(path) + ".", dir=directory or "."
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        return True
    except OSError as e:
        logger.warning("Could not save %s: %s", path, e)
        return False