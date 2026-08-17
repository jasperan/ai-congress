"""Observability endpoints (3.7.5): the congress "control room".

Aggregates the state that is otherwise computed and discarded — dynamic
weights (leaderboard), circuit-breaker states, confidence calibration,
recent runs, event-logger fallback stats — into a single summary the
frontend dashboard can render.
"""
import logging

from fastapi import APIRouter

from ..state import config, event_logger, get_enhanced_orchestrator, model_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/observability", tags=["observability"])


@router.get("/summary")
async def observability_summary():
    """One-page summary: leaderboard, breakers, calibration, runs."""
    orch = get_enhanced_orchestrator()
    summary: dict = {}

    # --- Leaderboard: dynamic weights + outcomes + benchmark weights ---
    leaderboard = []
    try:
        dw = orch.dynamic_weight_manager.get_performance_stats()
        for model, stats in sorted(
            dw.items(),
            key=lambda kv: kv[1].get("current_weight", 0.0),
            reverse=True,
        ):
            leaderboard.append({
                "model": model,
                "weight": stats.get("current_weight", 0.0),
                "win_rate": stats.get("win_rate", 0.0),
                "participations": stats.get("total_participations", 0),
                "benchmark_weight": model_registry.get_model_weight(model),
            })
    except Exception as e:
        logger.warning("Leaderboard failed: %s", e)
    summary["leaderboard"] = leaderboard

    # --- Circuit-breaker states ---
    try:
        breaker_states = orch.circuit_breaker.get_all_states()
        summary["circuit_breakers"] = {
            model: {
                "state": info.get("state"),
                "failure_count": info.get("failure_count", 0),
                "last_failure_age_s": (
                    max(0, __import__("time").time() - info.get("last_failure_time", 0))
                    if info.get("last_failure_time")
                    else None
                ),
            }
            for model, info in breaker_states.items()
        }
        summary["breaker_open_count"] = sum(
            1 for i in breaker_states.values() if i.get("state") == "OPEN"
        )
    except Exception as e:
        logger.warning("Breaker states failed: %s", e)
        summary["circuit_breakers"] = {}

    # --- Confidence calibration ---
    try:
        summary["calibration"] = orch.confidence_calibrator.get_all_stats()
    except Exception as e:
        logger.warning("Calibration failed: %s", e)
        summary["calibration"] = {}

    # --- Recent runs (last 10) ---
    runs = []
    try:
        for run_id, run in list(orch._runs.items())[-10:]:
            runs.append({
                "run_id": run_id,
                "query": (run.query or "")[:80],
                "status": getattr(run, "status", "unknown"),
                "duration_seconds": getattr(run, "duration_seconds", None),
                "event_count": len(getattr(run, "event_log", []) or []),
                "final_answer": (getattr(run, "final_answer", "") or "")[:120],
            })
    except Exception as e:
        logger.warning("Runs failed: %s", e)
    summary["recent_runs"] = runs

    # --- Event-logger fallback (3.7.4) ---
    try:
        summary["event_logger"] = event_logger.get_fallback_stats()
    except Exception as e:
        logger.warning("Event logger stats failed: %s", e)
        summary["event_logger"] = {}

    # --- MoE routing (if the orchestrator exposes it) ---
    try:
        summary["moe_routing"] = orch.moe_router.get_routing_stats()
    except Exception as e:
        summary["moe_routing"] = {}

    # --- Calibration curves (dashboard L3): per-model confidence bins ---
    try:
        curves = []
        for model, bins in (orch.confidence_calibrator.get_all_stats() or {}).items():
            points = []
            for bin_idx, b in sorted(bins.items(), key=lambda kv: int(kv[0])):
                lo, hi = b["bin_range"]
                points.append({
                    "bin": f"{lo:.2f}-{hi:.2f}",
                    "bin_center": (lo + hi) / 2,
                    "accuracy": b["accuracy"],
                    "n": b["total"],
                })
            if points:
                curves.append({"model": model, "points": points})
        summary["calibration_curves"] = curves
    except Exception as e:
        logger.warning("Calibration curves failed: %s", e)
        summary["calibration_curves"] = []

    # --- Per-domain win rates (dashboard L3): feedback log grouped by
    #     domain tag (3.5.4) × model ---
    try:
        domains: dict[str, dict] = {}
        for entry in orch.feedback_collector.get_all_feedback():
            domain = entry.get("domain") or "general"
            model = entry.get("model", "?")
            bucket = domains.setdefault(
                domain, {"positive": 0, "negative": 0, "by_model": {}}
            )
            if entry.get("feedback") == "positive":
                bucket["positive"] += 1
            else:
                bucket["negative"] += 1
            mb = bucket["by_model"].setdefault(model, {"positive": 0, "negative": 0})
            if entry.get("feedback") == "positive":
                mb["positive"] += 1
            else:
                mb["negative"] += 1
        domain_rows = []
        for domain, bucket in sorted(domains.items()):
            total = bucket["positive"] + bucket["negative"]
            models_ranked = sorted(
                bucket["by_model"].items(),
                key=lambda kv: (kv[1]["positive"] - kv[1]["negative"]),
                reverse=True,
            )[:5]
            domain_rows.append({
                "domain": domain,
                "feedback_count": total,
                "win_rate": (bucket["positive"] / total) if total else 0.0,
                "top_models": [
                    {
                        "model": m,
                        "positive": s["positive"],
                        "negative": s["negative"],
                        "win_rate": (s["positive"] / (s["positive"] + s["negative"]))
                        if (s["positive"] + s["negative"])
                        else 0.0,
                    }
                    for m, s in models_ranked
                ],
            })
        summary["domain_win_rates"] = domain_rows
    except Exception as e:
        logger.warning("Domain win rates failed: %s", e)
        summary["domain_win_rates"] = []

    return summary


@router.get("/leaderboard")
async def leaderboard():
    """Ranked model standings (weight + win rate)."""
    orch = get_enhanced_orchestrator()
    try:
        dw = orch.dynamic_weight_manager.get_performance_stats()
        rows = [
            {
                "model": model,
                "weight": stats.get("current_weight", 0.0),
                "win_rate": stats.get("win_rate", 0.0),
                "participations": stats.get("total_participations", 0),
            }
            for model, stats in dw.items()
        ]
        rows.sort(key=lambda r: r["weight"], reverse=True)
        return {"leaderboard": rows}
    except Exception as e:
        logger.warning("Leaderboard failed: %s", e)
        return {"leaderboard": []}
