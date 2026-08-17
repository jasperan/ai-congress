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
