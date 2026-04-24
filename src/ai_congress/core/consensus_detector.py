"""
Consensus Detector - detects premature consensus among agent outputs.

Uses sentence-transformers cosine similarity when available, falls back to
a token-overlap Jaccard similarity so the module is always importable even
without the heavy ML dependency.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_EMBED_MODEL = None
_EMBED_IMPORT_FAILED = False


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
        logger.info(f"sentence-transformers unavailable ({exc}); using lexical fallback")
        return None


_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_WORD_RE.findall(text.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _cosine_similarity_pair(v1, v2) -> float:
    import numpy as np  # numpy ships with sentence-transformers
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


@dataclass
class ConsensusReport:
    agreement_ratio: float          # share of pairs above pairwise_threshold
    mean_similarity: float          # mean pairwise similarity
    premature: bool                 # True when agreement_ratio > consensus_threshold
    method: str                     # "embedding" or "lexical"
    pairwise: List[Tuple[int, int, float]]
    outlier_index: Optional[int] = None   # least-similar agent to the majority

    def to_dict(self) -> dict:
        return {
            "agreement_ratio": self.agreement_ratio,
            "mean_similarity": self.mean_similarity,
            "premature": self.premature,
            "method": self.method,
            "outlier_index": self.outlier_index,
            "pairwise": self.pairwise,
        }


def detect_consensus(
    texts: List[str],
    consensus_threshold: float = 0.7,
    pairwise_threshold: float = 0.7,
    embedding_model: Optional[str] = None,
) -> ConsensusReport:
    """Detect premature agreement across a list of agent responses.

    A pair counts as "agreeing" when its similarity is >= pairwise_threshold.
    agreement_ratio is the fraction of unordered pairs that agree. Consensus
    is flagged as premature once agreement_ratio exceeds consensus_threshold.
    """
    n = len(texts)
    if n < 2:
        return ConsensusReport(
            agreement_ratio=1.0 if n == 1 else 0.0,
            mean_similarity=1.0 if n == 1 else 0.0,
            premature=False,
            method="trivial",
            pairwise=[],
            outlier_index=None,
        )

    method = "lexical"
    vectors = None
    if embedding_model:
        model = _load_embedding_model(embedding_model)
        if model is not None:
            try:
                vectors = model.encode(texts, show_progress_bar=False)
                method = "embedding"
            except Exception as exc:
                logger.warning(f"embedding failed, falling back to lexical: {exc}")
                vectors = None

    pairs: List[Tuple[int, int, float]] = []
    sims_by_agent: List[List[float]] = [[] for _ in range(n)]
    agree_count = 0
    total_pairs = 0
    total_sim = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            if vectors is not None:
                sim = _cosine_similarity_pair(vectors[i], vectors[j])
            else:
                sim = _jaccard(texts[i], texts[j])
            sim = max(0.0, min(1.0, sim))
            pairs.append((i, j, sim))
            sims_by_agent[i].append(sim)
            sims_by_agent[j].append(sim)
            total_sim += sim
            total_pairs += 1
            if sim >= pairwise_threshold:
                agree_count += 1

    agreement_ratio = agree_count / total_pairs if total_pairs else 0.0
    mean_similarity = total_sim / total_pairs if total_pairs else 0.0

    outlier = None
    if sims_by_agent:
        avg_per_agent = [
            sum(vals) / len(vals) if vals else 0.0 for vals in sims_by_agent
        ]
        outlier = int(min(range(n), key=lambda k: avg_per_agent[k]))

    return ConsensusReport(
        agreement_ratio=agreement_ratio,
        mean_similarity=mean_similarity,
        premature=agreement_ratio > consensus_threshold,
        method=method,
        pairwise=pairs,
        outlier_index=outlier,
    )


def pick_steelman_targets(
    report: ConsensusReport, count: int = 2
) -> List[int]:
    """Pick agent indices that should steelman the opposing view.

    Prefers agents whose output is least similar to the rest (true dissenters),
    then rounds out with the most-central agents so the steelmen actually
    have to work against their own position.
    """
    if not report.pairwise:
        return []
    n_agents = 1 + max((max(i, j) for i, j, _ in report.pairwise), default=0)
    sims_by_agent: List[List[float]] = [[] for _ in range(n_agents)]
    for i, j, sim in report.pairwise:
        sims_by_agent[i].append(sim)
        sims_by_agent[j].append(sim)
    avg = [
        sum(vals) / len(vals) if vals else 0.0 for vals in sims_by_agent
    ]
    order_low_to_high = sorted(range(n_agents), key=lambda k: avg[k])
    picks: List[int] = []
    for idx in order_low_to_high:
        if idx not in picks:
            picks.append(idx)
        if len(picks) >= count:
            break
    for idx in reversed(order_low_to_high):
        if len(picks) >= count:
            break
        if idx not in picks:
            picks.append(idx)
    return picks[:count]
