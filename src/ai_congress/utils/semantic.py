"""
Shared semantic similarity utilities.

One embedding-backed similarity function with a lexical fallback, used across
voting, coalition formation, memory recall, and consensus detection. Mirrors
the graceful pattern from consensus_detector.py: sentence-transformers is
loaded lazily and its absence degrades to token-overlap Jaccard.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

_EMBED_MODEL = None
_EMBED_IMPORT_FAILED = False

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_WORD_RE.findall(text.lower()))


def jaccard_similarity(a: str, b: str) -> float:
    """Token-overlap Jaccard similarity, always importable."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Lazy-load the sentence-transformers model. Returns None on failure."""
    global _EMBED_MODEL, _EMBED_IMPORT_FAILED
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    if _EMBED_IMPORT_FAILED:
        return None
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _EMBED_MODEL = SentenceTransformer(model_name)
        return _EMBED_MODEL
    except Exception as exc:
        _EMBED_IMPORT_FAILED = True
        logger.info("sentence-transformers unavailable (%s); using lexical fallback", exc)
        return None


def _cosine(v1, v2) -> float:
    import numpy as np  # numpy ships with sentence-transformers
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def embedding_available() -> bool:
    """True when the sentence-transformers backend is loadable."""
    return _load_embedding_model() is not None


def text_similarity(
    a: str,
    b: str,
    embedding_model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
) -> float:
    """Semantic similarity (0-1) with lexical fallback.

    Uses cosine similarity over sentence-transformer embeddings when the
    dependency is installed; otherwise falls back to token-overlap Jaccard.
    """
    a, b = (a or "").strip(), (b or "").strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    model = _load_embedding_model(embedding_model) if embedding_model else None
    if model is not None:
        try:
            vectors = model.encode([a, b], show_progress_bar=False)
            return _cosine(vectors[0], vectors[1])
        except Exception as exc:
            logger.debug("embedding similarity failed, falling back to lexical: %s", exc)
    return jaccard_similarity(a, b)


def group_by_similarity(
    texts: list[str],
    threshold: float = 0.7,
    embedding_model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[list[int]]:
    """Greedily group text indices whose pairwise similarity >= threshold.

    Returns a list of index groups (each group is a list of indices into
    ``texts``). Order-preserving; the first index of each group is the anchor.
    """
    n = len(texts)
    groups: list[list[int]] = []
    assigned: set[int] = set()
    for i in range(n):
        if i in assigned:
            continue
        group = [i]
        assigned.add(i)
        for j in range(i + 1, n):
            if j in assigned:
                continue
            if text_similarity(texts[i], texts[j], embedding_model) >= threshold:
                group.append(j)
                assigned.add(j)
        groups.append(group)
    return groups
