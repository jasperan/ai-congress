"""Pipeline helpers for EnhancedOrchestrator.enhanced_swarm."""

import logging
from dataclasses import dataclass
from typing import Any

from .acp.anchoring import anchor_responses, format_anchored_debate_prompt
from .acp.roles import AgentRole
from .acp.run_context import RunStatus
from .acp.supervisor import RestartPolicy, SupervisedTask
from .debate_artifact import DebateArtifact
from .intelligence.role_prompts import get_role_prompt
from .precedent.precedent_injector import PrecedentAction

logger = logging.getLogger(__name__)


ROLE_MAP = {
    "planner": AgentRole.PLANNER,
    "critic": AgentRole.CRITIC,
    "worker": AgentRole.WORKER,
    "synthesizer": AgentRole.SYNTHESIZER,
}


@dataclass
class RoleSetup:
    profiles: dict[str, Any]
    weights: dict[str, float]
    role_assignments: dict[Any, list[Any]]
    role_summary: dict[str, Any]


@dataclass
class DebateOutcome:
    revised_responses: list[dict[str, Any]]
    response_models: list[str]
    response_texts: list[str]
    debate_rounds: int
    artifact: DebateArtifact


def determine_degradation_mode(runtime: Any, models: list[str], run: Any) -> dict[str, Any]:
    """Determine graceful-degradation mode for the requested model set."""
    try:
        degradation_mode_info = runtime.graceful_degradation.determine_mode(
            available_models=len(models),
            total_models=len(models),
        )
        run.log_event("DEGRADATION_CHECK", detail=str(degradation_mode_info))
        return degradation_mode_info
    except Exception as e:
        logger.warning("Graceful degradation check failed: %s", e)
        return {"mode": "full"}


def filter_available_models(
    runtime: Any,
    models: list[str],
    run: Any,
) -> tuple[list[str], dict[str, Any]]:
    """Apply circuit breaker filtering to the requested model list."""
    available_models = []
    circuit_breaker_states = {}
    try:
        for model in models:
            if runtime.circuit_breaker.can_execute(model):
                available_models.append(model)
            else:
                run.log_event("CIRCUIT_BREAKER_SKIP", model, "Model circuit breaker OPEN")
                logger.info("Skipping model %s: circuit breaker OPEN", model)
        circuit_breaker_states = runtime.circuit_breaker.get_all_states()
    except Exception as e:
        logger.warning("Circuit breaker check failed: %s", e)
        available_models = list(models)

    return available_models, circuit_breaker_states


def assign_swarm_roles(runtime: Any, available_models: list[str], run: Any) -> RoleSetup:
    """Register models, assign roles, and apply org-chart weight bonuses."""
    runtime._register_agents(available_models)
    profiles = {m: runtime.personality_loader.get_profile(m) for m in available_models}
    weights = {m: runtime.dynamic_weight_manager.get_weight(m) for m in available_models}

    role_assignments = runtime.role_dispatcher.assign_roles(available_models, profiles, weights)
    role_summary = runtime.role_dispatcher.get_role_summary(role_assignments)
    run.log_event("ROLES_ASSIGNED", detail=str(role_summary))

    for model in available_models:
        bonus = runtime.org_chart.get_rank_bonus(model)
        if bonus != 1.0:
            weights[model] = weights.get(model, 0.5) * bonus

    for model in available_models:
        assigned_role = "worker"
        for role, assignments in role_assignments.items():
            if any(a.model_name == model for a in assignments):
                assigned_role = role.value
                break
        run.register_agent(model, role=assigned_role)

    return RoleSetup(
        profiles=profiles,
        weights=weights,
        role_assignments=role_assignments,
        role_summary=role_summary,
    )


def load_personality_state(runtime: Any, available_models: list[str], profiles: dict[str, Any], run: Any) -> None:
    """Apply persisted personality state where available."""
    try:
        for model in available_models:
            profile = profiles.get(model)
            if profile:
                runtime.personality_persistence.apply_persisted_state(model, profile)
        run.log_event("PERSONALITY_LOADED", detail="Persisted state applied")
    except Exception as e:
        logger.warning("Personality persistence load failed: %s", e)


async def run_query_decomposition(
    runtime: Any,
    prompt: str,
    effective_prompt: str,
    run: Any,
    role_assignments: dict[Any, list[Any]],
    enable_decomposition: bool,
) -> tuple[list[str], str]:
    """Run optional planner decomposition and return planners plus the effective prompt."""
    planners = runtime.role_dispatcher.get_models_for_role(role_assignments, AgentRole.PLANNER)

    if enable_decomposition and planners:
        planner_model = planners[0]
        run.log_event("DECOMPOSITION_START", planner_model)

        planner_system = get_role_prompt("planner")
        decompose_prompt = (
            "Break this question into 1-3 focused sub-questions that would help answer it comprehensively. "
            "If the question is simple enough to answer directly, just repeat it as-is.\n\n"
            f"Question: {prompt}\n\n"
            "Output each sub-question on its own line, prefixed with '- '."
        )
        decomp_result = await runtime._query_model(
            planner_model,
            decompose_prompt,
            temperature=0.3,
            system_prompt=planner_system,
            timeout=runtime.adaptive_timeout.get_timeout(planner_model),
        )
        if decomp_result["success"]:
            lines = [
                line.strip().lstrip("- ").strip()
                for line in decomp_result["response"].split("\n")
                if line.strip().startswith("- ") or line.strip().startswith("-")
            ]
            if lines:
                run.sub_queries = [{"text": sq, "source": planner_model} for sq in lines]
                run.log_event("DECOMPOSED", planner_model, f"sub_queries={len(lines)}")

            try:
                runtime.adaptive_timeout.record_latency(planner_model, decomp_result.get("latency_ms", 0))
            except Exception:
                pass

    if run.sub_queries:
        sub_q_text = "\n".join(f"- {sq['text']}" for sq in run.sub_queries)
        effective_prompt = (
            f"{effective_prompt}\n\nTo answer comprehensively, address these sub-questions:\n{sub_q_text}"
        )

    return planners, effective_prompt


async def run_initial_response_wave(
    runtime: Any,
    run: Any,
    available_models: list[str],
    role_assignments: dict[Any, list[Any]],
    effective_prompt: str,
    temperature: float,
    precedent_action: PrecedentAction,
    cited_precedents: list[Any],
) -> list[dict[str, Any]]:
    """Run the first supervised response wave and record model health signals."""
    run.log_event("WAVE_1_START", detail="Initial parallel queries with role prompts")
    run.advance_turn()

    supervised_tasks = []
    for model in available_models:
        model_role = "worker"
        for role, assignments in role_assignments.items():
            if any(a.model_name == model for a in assignments):
                model_role = role.value
                break

        role_system_prompt = get_role_prompt(model_role)

        if runtime.goal_engine:
            agent_role = ROLE_MAP.get(model_role, AgentRole.WORKER)
            objective = runtime.goal_engine.create_agent_objective(model, agent_role)
            alignment_prompt = runtime.goal_engine.build_alignment_prompt(objective)
            role_system_prompt = f"{role_system_prompt}\n\n{alignment_prompt}"

        if precedent_action == PrecedentAction.SOFT_CITE and cited_precedents:
            role_system_prompt = runtime.precedent_injector.augment_system_prompt(
                role_system_prompt,
                cited_precedents,
                precedent_action,
            )

        model_timeout = runtime.adaptive_timeout.get_timeout(model)

        supervised_tasks.append(
            SupervisedTask(
                agent_id=model,
                coro_factory=runtime._throttled_query,
                args=(model, effective_prompt, temperature),
                kwargs={"system_prompt": role_system_prompt, "timeout": model_timeout},
                restart_policy=RestartPolicy.RESTART,
                max_retries=2,
                stall_timeout=model_timeout + 10.0,
            )
        )

    completed_tasks = await runtime.supervisor.supervise_all(supervised_tasks)

    initial_responses = []
    for task in completed_tasks:
        if task.success and task.result:
            initial_responses.append(task.result)
            run.record_response(task.agent_id, task.result.get("response", ""))

            try:
                latency_ms = task.result.get("latency_ms", 0)
                if latency_ms > 0:
                    runtime.adaptive_timeout.record_latency(task.agent_id, latency_ms)
            except Exception:
                pass

            try:
                runtime.circuit_breaker.record_success(task.agent_id)
            except Exception:
                pass
        else:
            try:
                runtime.circuit_breaker.record_failure(task.agent_id)
            except Exception:
                pass

    return initial_responses


async def revise_sub_queries_after_wave(
    runtime: Any,
    run: Any,
    prompt: str,
    initial_responses: list[dict[str, Any]],
    planners: list[str],
    available_models: list[str],
) -> str | None:
    """Optionally revise sub-queries after the first wave."""
    if not run.sub_queries or runtime.task_reviser.revisions_remaining(run) <= 0:
        return None

    try:
        signal = await runtime.task_reviser.assess(run, initial_responses)
        if signal.should_revise:
            planner_model = planners[0] if planners else available_models[0]
            run.log_event(
                "SUB_QUERY_REVISION_TRIGGERED",
                planner_model,
                signal.reason,
            )
            revised_sq = await runtime.task_reviser.revise(run, signal, planner_model)
            run.sub_queries = revised_sq
            sub_q_text = "\n".join(f"- {sq['text']}" for sq in revised_sq)
            effective_prompt = (
                f"{prompt}\n\nAddress these refined sub-questions:\n{sub_q_text}"
            )
            run.log_event(
                "SUB_QUERY_REVISED",
                planner_model,
                f"revised={len(revised_sq)} sub-queries",
            )
            return effective_prompt

        run.log_event(
            "SUB_QUERY_REVISION_SKIPPED",
            detail=f"scores: div={signal.divergence_score:.2f} cov={signal.coverage_score:.2f} conf={signal.avg_confidence:.2f}",
        )
    except Exception as e:
        logger.warning("Sub-query revision failed: %s", e)

    return None


async def run_debate_wave(
    runtime: Any,
    run: Any,
    prompt: str,
    available_models: list[str],
    role_assignments: dict[Any, list[Any]],
    initial_responses: list[dict[str, Any]],
    planners: list[str],
    temperature: float,
) -> DebateOutcome:
    """Run the hash-anchored critique wave and return the debate outcome."""
    run.log_event("WAVE_2_START", detail="Anchored critique wave with dynamic depth")
    run.advance_turn()
    run.status = RunStatus.DEBATING

    response_texts = [r["response"] for r in initial_responses]
    response_models = [r["model"] for r in initial_responses]
    anchored = anchor_responses(response_models, response_texts)

    for ar in anchored:
        run.anchored_responses[ar.anchor] = ar.text

    debate_rounds = 1
    try:
        if len(response_texts) >= 2:
            similarities = []
            for i in range(len(response_texts)):
                for j in range(i + 1, len(response_texts)):
                    sim = runtime.coalition_formation.compute_similarity(
                        response_texts[i],
                        response_texts[j],
                    )
                    similarities.append(sim)
            initial_consensus = sum(similarities) / len(similarities) if similarities else 0.5
        else:
            initial_consensus = 1.0

        depth_result = runtime.dynamic_debate_depth.determine_depth(
            consensus=initial_consensus,
            num_models=len(available_models),
        )
        debate_rounds = depth_result.get("rounds", 1)
        run.log_event(
            "DEBATE_DEPTH",
            detail=f"rounds={debate_rounds}, reason={depth_result.get('reason', '')}",
        )
    except Exception as e:
        logger.warning("Dynamic depth determination failed: %s", e)
        initial_consensus = 1.0
        debate_rounds = 1

    try:
        temp_schedule = runtime.dynamic_debate_depth.get_temperature_schedule(debate_rounds)
    except Exception:
        temp_schedule = [max(0.3, temperature - 0.2)] * max(debate_rounds, 1)

    revised_responses = list(initial_responses)

    for round_idx in range(debate_rounds):
        round_temp = temp_schedule[round_idx] if round_idx < len(temp_schedule) else 0.4
        critique_prompt = format_anchored_debate_prompt(
            prompt,
            anchored,
            "Review these responses. Reference specific responses by model#hash. "
            "If you see merit in another position, revise your answer. "
            "If you still believe your answer is correct, strengthen your argument.",
        )

        if 0.3 < initial_consensus < 0.7 and round_idx == 0:
            try:
                critics = runtime.role_dispatcher.get_models_for_role(role_assignments, AgentRole.CRITIC)
                da_model = critics[0] if critics else response_models[0]
                majority_answer = revised_responses[0]["response"] if revised_responses else ""
                devils_advocate_result = await runtime.devils_advocate.run_devils_advocate(
                    question=prompt,
                    majority_answer=majority_answer,
                    critic_model=da_model,
                    ollama_client=runtime.ollama_client,
                )
                if devils_advocate_result.get("is_compelling"):
                    critique_prompt += (
                        "\n\nA devil's advocate challenge has been raised:\n"
                        f"{devils_advocate_result.get('challenge', '')}\n\n"
                        "Address this challenge in your response."
                    )
                    run.log_event(
                        "DEVILS_ADVOCATE",
                        da_model,
                        f"strength={devils_advocate_result.get('strength', 0):.2f}",
                    )
            except Exception as e:
                logger.warning("Devil's advocate failed: %s", e)

        critic_system_prompt = get_role_prompt("critic")
        critique_tasks = [
            SupervisedTask(
                agent_id=f"{model}_critique_r{round_idx}",
                coro_factory=runtime._throttled_query,
                args=(model, critique_prompt, round_temp),
                kwargs={
                    "system_prompt": critic_system_prompt,
                    "timeout": runtime.adaptive_timeout.get_timeout(model),
                },
                restart_policy=RestartPolicy.SKIP,
                max_retries=1,
                stall_timeout=runtime.adaptive_timeout.get_timeout(model) + 10.0,
            )
            for model in response_models
        ]

        critique_completed = await runtime.supervisor.supervise_all(critique_tasks)

        new_revised = []
        for task, original in zip(critique_completed, revised_responses):
            if task.success and task.result:
                new_revised.append(task.result)
                run.record_response(task.agent_id, task.result.get("response", ""))
                try:
                    model_name = task.result.get("model", task.agent_id.split("_critique")[0])
                    latency = task.result.get("latency_ms", 0)
                    if latency > 0:
                        runtime.adaptive_timeout.record_latency(model_name, latency)
                    runtime.circuit_breaker.record_success(model_name)
                except Exception:
                    pass
            else:
                new_revised.append(original)
                try:
                    model_name = task.agent_id.split("_critique")[0]
                    runtime.circuit_breaker.record_failure(model_name)
                except Exception:
                    pass

        revised_responses = new_revised

        if (
            run.sub_queries
            and runtime.task_reviser
            and runtime.task_reviser.revisions_remaining(run) > 0
            and round_idx < debate_rounds - 1
        ):
            try:
                signal = await runtime.task_reviser.assess(run, revised_responses)
                if signal.should_revise:
                    planner_model = planners[0] if planners else available_models[0]
                    revised_sq = await runtime.task_reviser.revise(run, signal, planner_model)
                    run.sub_queries = revised_sq
                    run.log_event(
                        "SUB_QUERY_REVISED",
                        planner_model,
                        f"mid-debate revision at round {round_idx + 1}",
                    )
            except Exception as e:
                logger.warning("Mid-debate revision failed: %s", e)

        anchored = anchor_responses(
            [r["model"] for r in revised_responses],
            [r["response"] for r in revised_responses],
        )

    artifact = DebateArtifact()
    artifact.save_round(
        round_num=0,
        responses={r["model"]: r["response"][:200] for r in initial_responses},
    )
    for round_idx in range(debate_rounds):
        artifact.save_round(
            round_num=round_idx + 1,
            responses={r["model"]: r["response"][:200] for r in revised_responses},
        )

    return DebateOutcome(
        revised_responses=revised_responses,
        response_models=response_models,
        response_texts=response_texts,
        debate_rounds=debate_rounds,
        artifact=artifact,
    )
