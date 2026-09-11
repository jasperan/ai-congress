"""
Offline Eval Harness (3.5.6 / 4.6.7)
------------------------------------
Curated question set with ground-truth key phrases; run models over it and
score responses with semantic similarity (embedding when available, lexical
otherwise). Writes a JSON report artifact and can fold per-model accuracy
back into ``config/models_benchmark.json`` so the benchmark stays a live
measurement instead of a stale hand-maintained table.

Used by:
  - ``run_cli.py eval``          (on-demand run)
  - ``tests/test_evals.py``     (evals-as-tests, hermetic with mock client)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .semantic import text_similarity

logger = logging.getLogger(__name__)

# ── Curated question set ────────────────────────────────────────────────
# Each entry: question + ground-truth tokens that must appear (or be
# semantically matched) in a correct answer.
EVAL_SET: List[Dict[str, Any]] = [
    {
        "id": "capital-france",
        "question": "What is the capital of France?",
        "ground_truth": "Paris",
        "keywords": ["paris"],
        "domain": "geography",
    },
    {
        "id": "boiling-water",
        "question": "At what temperature does water boil at sea level in degrees Celsius?",
        "ground_truth": "100 degrees Celsius",
        "keywords": ["100", "celsius"],
        "domain": "science",
    },
    {
        "id": "html-purpose",
        "question": "What does the acronym HTML stand for?",
        "ground_truth": "HyperText Markup Language",
        "keywords": ["hypertext", "markup"],
        "domain": "technology",
    },
    {
        "id": "photosynthesis",
        "question": "What process do plants use to convert sunlight into energy?",
        "ground_truth": "photosynthesis",
        "keywords": ["photosynthesis"],
        "domain": "science",
    },
    {
        "id": "git-purpose",
        "question": "What is a version control system used for?",
        "ground_truth": "tracking changes to code over time",
        "keywords": ["track", "change", "version"],
        "domain": "software",
    },
    {
        "id": "python-creator",
        "question": "Who created the Python programming language?",
        "ground_truth": "Guido van Rossum",
        "keywords": ["guido", "van rossum"],
        "domain": "software",
    },
    {
        "id": "rag-meaning",
        "question": "What does RAG stand for in AI?",
        "ground_truth": "Retrieval-Augmented Generation",
        "keywords": ["retrieval", "augmented", "generation"],
        "domain": "ai",
    },
    {
        "id": "req-time",
        "question": "A system receives 120 requests in 2 minutes. What is the average requests per minute?",
        "ground_truth": "60",
        "keywords": ["60"],
        "domain": "math",
    },
]


class _AdaptedClient:
    """Normalize the client's generate() signature for the harness.

    ``OllamaClient.generate`` takes ``options={...}`` while mock/pi clients
    accept ``temperature=``. Prefer the modern kwargs and fall back to the
    options-dict form for the local Ollama client.
    """
    def __init__(self, client: Any):
        self._c = client

    async def generate(self, *, prompt: str, model: str, stream: bool = False, temperature: float = 0.3):
        try:
            return await self._c.generate(prompt=prompt, model=model, stream=stream, temperature=temperature)
        except (TypeError, AttributeError):
            try:
                return await self._c.generate(
                    model=model, prompt=prompt, options={"temperature": temperature}, stream=stream
                )
            except AttributeError:
                # OpenAI-compatible clients expose chat() with messages instead
                result = await self._c.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": temperature},
                    stream=stream,
                )
                if isinstance(result, dict) and "message" in result:
                    return {"response": result["message"].get("content", "")}
                return result


def _score_response(response: str, ground_truth: str, keywords: List[str]) -> float:
    """Score a response against ground truth.

    Semantic similarity (embedding or lexical) against the ground-truth
    phrase, blended with keyword recall. Both legs use utils.semantic so
    behavior is identical with or without sentence-transformers installed.
    """
    response_l = (response or "").lower()
    kw_hits = sum(1 for kw in keywords if kw.lower() in response_l)
    kw_recall = kw_hits / max(1, len(keywords))
    sem = text_similarity(response, ground_truth)
    return 0.6 * sem + 0.4 * kw_recall


async def _run_single(
    client: Any,
    model: str,
    item: Dict[str, Any],
    temperature: float = 0.3,
    timeout_s: float = 45.0,
) -> Dict[str, Any]:
    start = time.monotonic()
    try:
        result = await client.generate(
            prompt=item["question"],
            model=model,
            stream=False,
            temperature=temperature,
        )
        if isinstance(result, dict):
            response = result.get("response") or result.get("content") or result.get("text") or ""
        else:
            response = str(result)
        score = _score_response(response, item["ground_truth"], item["keywords"])
        return {
            "model": model,
            "question_id": item["id"],
            "question": item["question"],
            "response": (response or "")[:200],
            "score": round(score, 4),
            "passed": score >= 0.5,
            "duration_s": round(time.monotonic() - start, 2),
            "error": None,
        }
    except asyncio.TimeoutError:
        return {
            "model": model, "question_id": item["id"], "question": item["question"],
            "response": "", "score": 0.0, "passed": False,
            "duration_s": round(time.monotonic() - start, 2), "error": "timeout",
        }
    except Exception as e:  # noqa: BLE001 — harness must not die on one model
        return {
            "model": model, "question_id": item["id"], "question": item["question"],
            "response": "", "score": 0.0, "passed": False,
            "duration_s": round(time.monotonic() - start, 2), "error": str(e)[:120],
        }


async def run_evals(
    client: Any,
    models: List[str],
    question_ids: Optional[List[str]] = None,
    temperature: float = 0.3,
    report_dir: str = "data/evals",
) -> Dict[str, Any]:
    """Run the eval set over each model, in parallel, and write the report."""
    items = [i for i in EVAL_SET if not question_ids or i["id"] in question_ids]
    results: List[Dict[str, Any]] = []
    sem = asyncio.Semaphore(max(2, len(models)))  # local Ollama is single-GPU
    client = _AdaptedClient(client)

    async def bounded(model: str, item: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            return await _run_single(client, model, item, temperature=temperature)

    jobs = [bounded(m, item) for m in models for item in items]
    results = await asyncio.gather(*jobs)

    # Per-model aggregation
    per_model: Dict[str, Dict[str, Any]] = {}
    for r in results:
        entry = per_model.setdefault(
            r["model"], {"correct": 0, "total": 0, "score_sum": 0.0, "timeouts": 0, "errors": 0}
        )
        entry["total"] += 1
        entry["score_sum"] += r["score"]
        if r["passed"]:
            entry["correct"] += 1
        if r["error"] == "timeout":
            entry["timeouts"] += 1
        elif r["error"]:
            entry["errors"] += 1

    ranked = sorted(
        (
            {
                "model": m,
                "accuracy": round(stats["correct"] / max(1, stats["total"]), 4),
                "avg_score": round(stats["score_sum"] / max(1, stats["total"]), 4),
                "correct": stats["correct"],
                "total": stats["total"],
                "timeouts": stats["timeouts"],
                "errors": stats["errors"],
            }
            for m, stats in per_model.items()
        ),
        key=lambda d: d["accuracy"],
        reverse=True,
    )

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question_count": len(items),
        "model_count": len(models),
        "ranked": ranked,
        "results": results,
    }

    Path(report_dir).mkdir(parents=True, exist_ok=True)
    report_path = Path(report_dir) / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Eval report written to %s", report_path)
    return report


def compute_benchmark_update(
    report: Dict[str, Any],
    blend: float = 0.5,
    benchmark_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Fold eval accuracies back into the benchmark table.

    New weight = (1 - blend) * old_accuracy + blend * eval_accuracy, so a
    single eval run nudges rather than rewrites the standings.
    """
    benchmark_path = Path(benchmark_path or "config/models_benchmark.json")
    if not benchmark_path.exists():
        benchmark: Dict[str, Dict[str, Any]] = {}
    else:
        benchmark = json.loads(benchmark_path.read_text())

    updates: Dict[str, Dict[str, Any]] = {}
    for row in report.get("ranked", []):
        model, acc = row["model"], row["accuracy"]
        entry = benchmark.get(model, {})
        old = float(entry.get("accuracy", 0.5))
        new = round((1 - blend) * old + blend * acc, 4)
        entry["accuracy"] = new
        entry.setdefault("description", f"Updated by eval harness on {report.get('generated_at', '')}")
        entry["last_eval_accuracy"] = acc
        benchmark[model] = entry
        updates[model] = {"old": old, "new": new}

    # Keep the sort order stable (dict order matters to some viewers)
    benchmark = dict(
        sorted(benchmark.items(), key=lambda kv: kv[1].get("accuracy", 0), reverse=True)
    )
    benchmark_path.write_text(json.dumps(benchmark, indent=2))
    return updates


def summarize(report: Dict[str, Any]) -> str:
    """Human-readable one-liner summary for CLI/tests."""
    lines = [f"Evals: {report['question_count']} questions × {report['model_count']} models"]
    for row in report.get("ranked", []):
        lines.append(
            f"  {row['model']}: accuracy {row['accuracy']:.0%} "
            f"({row['correct']}/{row['total']}, avg {row['avg_score']:.2f}"
            + (f", {row['timeouts']} timeouts" if row["timeouts"] else "")
            + (f", {row['errors']} errors" if row["errors"] else "")
            + ")"
        )
    return "\n".join(lines)