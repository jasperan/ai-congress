"""
EnhancedOrchestrator - Integrates all 35 improvements from the brainstorming doc:

Intelligence: role_prompts, reasoning_router, query_classifier, moe_router
Voting: ensemble_voter, confidence_calibrator, minority_report, contextual_selector
Debate: structured_argumentation, devils_advocate, dynamic_depth
Coordination: circuit_breaker, graceful_degradation, adaptive_timeout, coalition_formation
Learning: dynamic_weights, feedback_loop, personality_persistence
RAG: swarm_rag
Observability: decision_explanation, performance_profiler

Plus the original 8 Symphony/Pi-inspired improvements (ACP, supervision, anchoring, etc.).
"""

import asyncio
import logging
import time
from typing import Optional

from .acp.handoff import AgentHandoff
from .acp.message import AgentIdentity, PersonalityProfile
from .acp.message_bus import ACPMessageBus
from .acp.registry import AgentRegistry
from .acp.roles import RoleDispatcher
from .acp.run_context import ImplementationRun, RunStatus
from .acp.supervisor import AgentSupervisor
from .debate_artifact import DebateArtifact
from .enhanced_pipeline import (
    assign_swarm_roles,
    determine_degradation_mode,
    filter_available_models,
    load_personality_state,
    revise_sub_queries_after_wave,
    run_debate_wave,
    run_initial_response_wave,
    run_query_decomposition,
)
from .model_registry import ModelRegistry
from .personality.emotional_voting import EmotionalVotingEngine
from .personality.profile import ModelPersonalityLoader
from .voting_engine import VotingEngine

# New module imports - intelligence
from .intelligence.reasoning_router import ReasoningRouter
from .intelligence.query_classifier import QueryClassifier
from .intelligence.moe_router import MixtureOfExpertsRouter

# New module imports - voting
from .voting.ensemble_voter import EnsembleVoter
from .voting.confidence_calibrator import ConfidenceCalibrator
from .voting.minority_report import MinorityReportGenerator
from .voting.contextual_selector import ContextualVotingSelector

# New module imports - debate
from .debate.dynamic_depth import DynamicDebateDepth
from .debate.devils_advocate import DevilsAdvocate
from .debate.structured_argumentation import StructuredArgumentation

# New module imports - coordination
from .coordination.circuit_breaker import CircuitBreaker
from .coordination.graceful_degradation import GracefulDegradation
from .coordination.adaptive_timeout import AdaptiveTimeout
from .coordination.coalition_formation import CoalitionFormation
from .coordination.concurrency_governor import ConcurrencyGovernor
from .coordination.task_reviser import TaskReviser

# New module imports - learning
from .learning.dynamic_weights import DynamicWeightManager
from .learning.feedback_loop import FeedbackCollector
from .learning.personality_persistence import PersonalityPersistence

# New module imports - rag
from .rag.swarm_rag import SwarmRAGIntegrator

# New module imports - observability
from .observability.decision_explanation import DecisionExplainer
from .observability.performance_profiler import PerformanceProfiler

# Paperclip-inspired modules
from .acp.audit_trail import AuditTrail, AuditEvent, AuditEventType
from .acp.goal_alignment import GoalAlignmentEngine
from .acp.org_chart import OrgChart
from .acp.heartbeat import HeartbeatManager

# Precedent-based reasoning (stare decisis)
from .precedent.precedent_store import PrecedentStore
from .precedent.precedent_injector import PrecedentInjector, PrecedentAction

logger = logging.getLogger(__name__)


class EnhancedOrchestrator:
    """Orchestrator integrating all 35 improvements plus the original 8 Symphony/Pi patterns."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        voting_engine: VotingEngine,
        ollama_client,
        personality_loader: ModelPersonalityLoader,
        coordination_level: str = "moderate",
        rag_engine=None,
        web_search_engine=None,
    ):
        self.model_registry = model_registry
        self.voting_engine = voting_engine
        self.ollama_client = ollama_client
        self.personality_loader = personality_loader
        self.rag_engine = rag_engine
        self.web_search_engine = web_search_engine

        # ACP infrastructure (original)
        self.registry = AgentRegistry()
        self.message_bus = ACPMessageBus()
        self.handoff = AgentHandoff(self.message_bus, self.registry)
        self.role_dispatcher = RoleDispatcher(self.registry)
        self.emotional_voting = EmotionalVotingEngine()

        # Supervision (original)
        self.supervisor = AgentSupervisor(
            max_retries=2,
            stall_timeout=60.0,
            backoff_base=0.5,
        )

        # GPU-aware concurrency control
        self.concurrency_governor = ConcurrencyGovernor()

        # Mid-debate sub-query revision
        self.task_reviser = TaskReviser(ollama_client=ollama_client)

        # State
        self._runs: dict[str, ImplementationRun] = {}

        # --- New module instances ---

        # Intelligence
        self.reasoning_router = ReasoningRouter()
        self.query_classifier = QueryClassifier()
        self.moe_router = MixtureOfExpertsRouter()

        # Voting
        self.ensemble_voter = EnsembleVoter(self.voting_engine)
        self.confidence_calibrator = ConfidenceCalibrator()
        self.minority_report_gen = MinorityReportGenerator()
        self.contextual_voting_selector = ContextualVotingSelector()

        # Debate
        self.dynamic_debate_depth = DynamicDebateDepth()
        self.devils_advocate = DevilsAdvocate()
        self.structured_argumentation = StructuredArgumentation()

        # Coordination
        self.circuit_breaker = CircuitBreaker()
        self.graceful_degradation = GracefulDegradation()
        self.adaptive_timeout = AdaptiveTimeout()
        self.coalition_formation = CoalitionFormation()

        # Learning
        base_weights = {}
        try:
            # ModelRegistry may not have list_models(); pull from benchmarks dict
            if hasattr(self.model_registry, "list_models"):
                model_names = self.model_registry.list_models() or []
            elif hasattr(self.model_registry, "benchmarks"):
                model_names = list(self.model_registry.benchmarks.keys())
            elif hasattr(self.model_registry, "weights"):
                model_names = list(self.model_registry.weights.keys())
            elif hasattr(self.model_registry, "_weights"):
                model_names = list(self.model_registry._weights.keys())
            else:
                model_names = []
            for model_name in model_names:
                base_weights[model_name] = self.model_registry.get_model_weight(model_name)
        except Exception:
            pass
        self.dynamic_weight_manager = DynamicWeightManager(base_weights=base_weights)
        self.feedback_collector = FeedbackCollector()
        self.personality_persistence = PersonalityPersistence()

        # RAG (optional)
        self.swarm_rag: Optional[SwarmRAGIntegrator] = None
        if self.rag_engine is not None:
            try:
                self.swarm_rag = SwarmRAGIntegrator(self.rag_engine)
            except Exception as e:
                logger.warning("Failed to initialize SwarmRAGIntegrator: %s", e)

        # Observability
        self.decision_explainer = DecisionExplainer()
        self.performance_profiler = PerformanceProfiler()

        # Precedent-based reasoning (stare decisis) — requires pool + embedder
        self.precedent_store: Optional[PrecedentStore] = None
        self.precedent_injector = PrecedentInjector()

        # Paperclip-inspired systems
        self.audit_trail = AuditTrail()
        self.org_chart = OrgChart()
        self.org_chart.define_default_structure()
        self.goal_engine: Optional[GoalAlignmentEngine] = None
        self.heartbeat_manager: Optional[HeartbeatManager] = None

        # Re-wire message bus with audit trail and org chart
        self.message_bus = ACPMessageBus(
            audit_trail=self.audit_trail,
            org_chart=self.org_chart,
        )
        # Re-create handoff with new bus
        self.handoff = AgentHandoff(self.message_bus, self.registry)

    def load_mission(self, mission_data: dict) -> None:
        """Load a mission configuration and initialize the goal alignment engine."""
        mission = GoalAlignmentEngine.mission_from_dict(mission_data)
        self.goal_engine = GoalAlignmentEngine(mission)
        self.audit_trail.record(AuditEvent(
            event_type=AuditEventType.MISSION_LOADED,
            agent_name="orchestrator",
            payload={"mission_id": mission.id, "statement": mission.statement},
        ))

    def _register_agents(self, models: list[str]):
        """Register models as ACP agents with personality profiles and audit trail."""
        for name in models:
            profile = self.personality_loader.get_profile(name)
            identity = AgentIdentity(
                name=name,
                role="voter",
                personality=profile,
                capabilities=["reasoning", "voting", "critique"],
            )
            self.registry.register(identity)
            self.message_bus.register_agent(name)
            self.audit_trail.record(AuditEvent(
                event_type=AuditEventType.AGENT_REGISTERED,
                agent_name=name,
                payload={"role": "voter", "capabilities": identity.capabilities},
            ))

    async def _query_model(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.7,
        system_prompt: str = "",
        timeout: float = 60.0,
        **kwargs,
    ) -> dict:
        """Query a single model with error handling, role prompts, and adaptive timeout."""
        start_time = time.time()
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            else:
                # Fallback to personality-based system message
                profile = self.personality_loader.get_profile(model_name)
                if profile.communication_style == "formal":
                    messages.append({"role": "system", "content": "Provide structured, professional responses with clear sections and precise language."})
                elif profile.communication_style == "casual":
                    messages.append({"role": "system", "content": "Explain in simple, conversational terms. Be approachable and clear."})

            messages.append({"role": "user", "content": prompt})

            response = await asyncio.wait_for(
                self.ollama_client.chat(
                    model=model_name,
                    messages=messages,
                    options={"temperature": temperature},
                ),
                timeout=timeout,
            )
            latency_ms = (time.time() - start_time) * 1000.0
            return {
                "model": model_name,
                "response": response["message"]["content"],
                "temperature": temperature,
                "success": True,
                "latency_ms": latency_ms,
            }
        except asyncio.TimeoutError:
            latency_ms = (time.time() - start_time) * 1000.0
            logger.error("Timeout querying %s after %.1fms", model_name, latency_ms)
            return {
                "model": model_name,
                "response": "",
                "temperature": temperature,
                "success": False,
                "error": f"Timeout after {timeout:.0f}s",
                "latency_ms": latency_ms,
            }
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000.0
            logger.error("Error querying %s: %s", model_name, e)
            return {
                "model": model_name,
                "response": "",
                "temperature": temperature,
                "success": False,
                "error": str(e),
                "latency_ms": latency_ms,
            }

    async def _throttled_query(self, model, prompt, temperature, **kwargs):
        """Wrap _query_model with GPU-aware concurrency throttling."""
        async with self.concurrency_governor.throttled() as stats:
            logger.debug(
                "GPU throttle: model=%s limit=%d vram=%.0f%%",
                model, stats["current_limit"], stats["vram_usage_pct"],
            )
            return await self._query_model(model, prompt, temperature, **kwargs)

    async def enhanced_swarm(
        self,
        prompt: str,
        models: list[str],
        temperature: float = 0.7,
        enable_decomposition: bool = True,
        enable_debate: bool = True,
    ) -> dict:
        """Full enhanced swarm with all 35 improvements integrated.

        Flow:
        a. Start performance profiler
        b. Create ImplementationRun context
        c. Graceful degradation check
        d. Circuit breaker filtering
        e. Register agents and assign roles
        f. Load persisted personality state
        g. Auto-select reasoning mode via ReasoningRouter
        h. Optional: RAG augmentation via SwarmRAGIntegrator
        i. Query decomposition (planners, if enabled)
        j. Wire sub-queries into effective prompt
        k. Supervised parallel queries (Wave 1) with role-differentiated prompts
        l. Record latencies in AdaptiveTimeout
        m. Record successes/failures in CircuitBreaker
        n. Hash-anchored critique wave (Wave 2) with dynamic depth
        o. Compute conviction scores
        p. Coalition formation before voting
        q. Contextual voting algorithm selection
        r. Ensemble voting
        s. Minority report generation
        t. Decision explanation
        u. Dynamic weight update
        v. Apply emotional drift
        w. Persist personality state
        x. End profiler, build result
        """
        # === (a) Start performance profiler ===
        profiler = PerformanceProfiler()
        profiler.start_stage("total_pipeline")
        profiler.start_stage("initialization")

        # === (b) Create ImplementationRun context ===
        run = ImplementationRun(query=prompt)
        run.status = RunStatus.RUNNING
        self._runs[run.run_id] = run
        run.log_event("RUN_START", detail=f"models={models}")

        # Tracking variables for new modules
        reasoning_mode = "direct"
        rag_context = None
        degradation_mode_info = {}
        circuit_breaker_states = {}
        minority_report = {}
        decision_explanation = ""
        performance_profile = {}
        coalition_info = []
        precedent_info = None
        precedent_action = PrecedentAction.NO_PRECEDENT
        cited_precedents = []

        # === (c) Graceful degradation check ===
        degradation_mode_info = determine_degradation_mode(self, models, run)

        # If error mode (no models), bail early
        if degradation_mode_info.get("mode") == "error":
            run.fail("No models available")
            profiler.end_stage("initialization")
            profiler.end_stage("total_pipeline")
            return self._build_result(
                run, [], {}, {},
                degradation_mode=degradation_mode_info,
                performance_profile=profiler.get_profile(),
            )

        # === (d) Circuit breaker filtering ===
        available_models, circuit_breaker_states = filter_available_models(self, models, run)

        if not available_models:
            run.fail("All models have open circuit breakers")
            profiler.end_stage("initialization")
            profiler.end_stage("total_pipeline")
            return self._build_result(
                run, [], {}, {},
                circuit_breaker_states=circuit_breaker_states,
                degradation_mode=degradation_mode_info,
                performance_profile=profiler.get_profile(),
            )

        # Re-check degradation with filtered model count
        if len(available_models) < len(models):
            try:
                degradation_mode_info = self.graceful_degradation.determine_mode(
                    available_models=len(available_models),
                    total_models=len(models),
                )
            except Exception as e:
                logger.warning("Re-degradation check failed: %s", e)

        # === (e) Register agents and assign roles ===
        role_setup = assign_swarm_roles(self, available_models, run)
        profiles = role_setup.profiles
        weights = role_setup.weights
        role_assignments = role_setup.role_assignments
        role_summary = role_setup.role_summary

        # === (f) Load persisted personality state ===
        load_personality_state(self, available_models, profiles, run)

        profiler.end_stage("initialization")

        # === (g) Auto-select reasoning mode via ReasoningRouter ===
        profiler.start_stage("reasoning_routing")
        try:
            reasoning_mode = self.reasoning_router.select_reasoning_mode(prompt)
            query_domain = self.query_classifier.classify_domain(prompt)
            run.log_event("REASONING_MODE", detail=f"mode={reasoning_mode}, domain={query_domain}")
        except Exception as e:
            logger.warning("Reasoning router failed: %s", e)
            reasoning_mode = "direct"
            query_domain = "general"
        profiler.end_stage("reasoning_routing")

        # === Precedent lookup (stare decisis) ===
        profiler.start_stage("precedent_lookup")
        if self.precedent_store is not None:
            try:
                cited_precedents = await self.precedent_store.search_precedents(
                    prompt, domain=query_domain,
                )
                precedent_action = self.precedent_injector.classify_action(cited_precedents)
                run.log_event("PRECEDENT_LOOKUP", detail=f"found={len(cited_precedents)}, action={precedent_action.value}")

                if precedent_action == PrecedentAction.FAST_FOLLOW:
                    fast_result = self.precedent_injector.build_fast_follow_response(cited_precedents[0])
                    run.log_event("PRECEDENT_FAST_FOLLOW", detail=f"id={cited_precedents[0].id}")
                    run.complete({})
                    profiler.end_stage("precedent_lookup")
                    profiler.end_stage("total_pipeline")
                    fast_result["run_id"] = run.run_id
                    fast_result["query"] = run.query
                    fast_result["reasoning_mode"] = reasoning_mode
                    fast_result["event_log"] = run.event_log
                    fast_result["duration_seconds"] = run.duration_seconds
                    return fast_result

            except Exception as e:
                logger.warning("Precedent lookup failed: %s", e)
                precedent_action = PrecedentAction.NO_PRECEDENT
                cited_precedents = []
        profiler.end_stage("precedent_lookup")

        # === (h) Optional: RAG augmentation ===
        profiler.start_stage("rag_augmentation")
        effective_prompt = prompt
        if self.swarm_rag is not None:
            try:
                rag_result = await self.swarm_rag.augment_swarm_prompt(prompt)
                if rag_result.get("chunks_used"):
                    effective_prompt = rag_result["augmented_prompt"]
                    rag_context = {
                        "chunks_used": len(rag_result["chunks_used"]),
                        "sources": [c.get("source", "unknown") for c in rag_result["chunks_used"]],
                    }
                    run.log_event("RAG_AUGMENTED", detail=f"chunks={len(rag_result['chunks_used'])}")
            except Exception as e:
                logger.warning("RAG augmentation failed: %s", e)
        profiler.end_stage("rag_augmentation")

        # === (i) Query decomposition (planners, if enabled) ===
        profiler.start_stage("query_decomposition")
        planners, effective_prompt = await run_query_decomposition(
            self,
            prompt,
            effective_prompt,
            run,
            role_assignments,
            enable_decomposition,
        )
        profiler.end_stage("query_decomposition")

        # Start GPU-aware concurrency control
        await self.concurrency_governor.start()

        # === (k) Supervised parallel queries (Wave 1) with role-differentiated prompts ===
        profiler.start_stage("wave_1_queries")
        initial_responses = await run_initial_response_wave(
            self,
            run,
            available_models,
            role_assignments,
            effective_prompt,
            temperature,
            precedent_action,
            cited_precedents,
        )

        profiler.end_stage("wave_1_queries")

        # === Sub-query revision check after Wave 1 ===
        revised_prompt = await revise_sub_queries_after_wave(
            self,
            run,
            prompt,
            initial_responses,
            planners,
            available_models,
        )
        if revised_prompt is not None:
            effective_prompt = revised_prompt

        if not initial_responses:
            run.fail("No successful initial responses")
            await self._stop_governor_safe()
            profiler.end_stage("total_pipeline")
            return self._build_result(
                run, [], {}, role_summary,
                reasoning_mode=reasoning_mode,
                degradation_mode=degradation_mode_info,
                circuit_breaker_states=self.circuit_breaker.get_all_states(),
                performance_profile=profiler.get_profile(),
            )

        # === (n) Hash-anchored critique wave (Wave 2) with dynamic depth ===
        profiler.start_stage("wave_2_debate")
        debate_outcome = await run_debate_wave(
            self,
            run,
            prompt,
            available_models,
            role_assignments,
            initial_responses,
            planners,
            temperature,
        )
        revised_responses = debate_outcome.revised_responses
        response_models = debate_outcome.response_models
        debate_rounds = debate_outcome.debate_rounds
        artifact = debate_outcome.artifact
        profiler.end_stage("wave_2_debate")

        # Stop GPU monitoring (safe to call multiple times)
        await self._stop_governor_safe()

        # === (o) Compute conviction scores ===
        profiler.start_stage("conviction_and_voting")
        for model in response_models:
            initial = next((r["response"] for r in initial_responses if r["model"] == model), "")
            revised = next((r["response"] for r in revised_responses if r["model"] == model), "")
            if initial and revised:
                try:
                    similarity = self.coalition_formation.compute_similarity(initial, revised)
                except Exception:
                    similarity = 1.0 if initial[:200] == revised[:200] else 0.5
                if model in run.agent_states:
                    run.agent_states[model].conviction_score = similarity

        # === (p) Coalition formation before voting ===
        try:
            coalition_input = [
                {
                    "model": r["model"],
                    "response": r["response"],
                    "weight": weights.get(r["model"], 0.5),
                }
                for r in revised_responses
            ]
            coalition_info = self.coalition_formation.form_coalitions(coalition_input)
            if coalition_info:
                run.log_event("COALITIONS_FORMED", detail=f"count={len(coalition_info)}")
        except Exception as e:
            logger.warning("Coalition formation failed: %s", e)
            coalition_info = []

        # === (q) Contextual voting algorithm selection ===
        selected_algorithm = "weighted_majority"
        try:
            response_lengths = [len(r["response"]) for r in revised_responses]
            selected_algorithm = self.contextual_voting_selector.select_algorithm(
                query=prompt,
                num_responses=len(revised_responses),
                response_lengths=response_lengths,
            )
            run.log_event("VOTING_ALGORITHM_SELECTED", detail=selected_algorithm)
        except Exception as e:
            logger.warning("Contextual voting selector failed: %s", e)

        # === (r) Ensemble voting ===
        run.log_event("VOTING_START")
        run.status = RunStatus.VOTING

        final_texts = [r["response"] for r in revised_responses]
        final_models = [r["model"] for r in revised_responses]
        final_weights = [weights.get(m, 0.5) for m in final_models]
        final_temperatures = [r.get("temperature", temperature) for r in revised_responses]

        winner = ""
        confidence = 0.0
        vote_details = {}

        try:
            ensemble_result = self.ensemble_voter.ensemble_vote(
                responses=final_texts,
                weights=final_weights,
                model_names=final_models,
                temperatures=final_temperatures,
            )
            winner = ensemble_result.get("winner", "")
            confidence = ensemble_result.get("confidence", 0.0)
            vote_details = ensemble_result
            run.log_event("ENSEMBLE_VOTE_COMPLETE", detail=f"agreement={ensemble_result.get('agreement_ratio', 0):.2f}")
        except Exception as e:
            logger.warning("Ensemble voting failed, falling back to emotional vote: %s", e)
            try:
                personalities = [profiles.get(m, PersonalityProfile()) for m in final_models]
                winner, confidence, vote_details = self.emotional_voting.emotional_weighted_vote(
                    final_texts, final_weights, personalities, final_models,
                )
            except Exception as e2:
                logger.error("Fallback emotional voting also failed: %s", e2)
                if final_texts:
                    winner = final_texts[0]
                    confidence = 0.0

        # Record consensus in audit trail
        self.audit_trail.record(AuditEvent(
            event_type=AuditEventType.CONSENSUS_REACHED,
            agent_name="orchestrator",
            payload={
                "confidence": confidence,
                "num_models": len(final_models),
                "run_id": run.run_id,
            },
            run_id=run.run_id,
            mission_alignment=self.goal_engine.score_alignment(winner) if self.goal_engine and winner else None,
        ))

        # Calibrate confidence
        try:
            if winner and final_models:
                winning_model = None
                for r in revised_responses:
                    if r["response"] == winner:
                        winning_model = r["model"]
                        break
                if winning_model:
                    confidence = self.confidence_calibrator.calibrate(winning_model, confidence)
        except Exception as e:
            logger.warning("Confidence calibration failed: %s", e)

        profiler.end_stage("conviction_and_voting")

        # === (s) Minority report generation ===
        profiler.start_stage("post_vote_analysis")
        try:
            # Build a vote_details-compatible dict for minority extraction
            if isinstance(vote_details, dict) and "candidates" in vote_details:
                # Use ensemble voter candidates for minority analysis
                minority_input = {}
                for resp_text, model_name, weight in zip(final_texts, final_models, final_weights):
                    norm_key = resp_text.strip().lower()
                    if norm_key not in minority_input:
                        minority_input[norm_key] = {
                            "original": resp_text,
                            "weight": weight,
                            "models": [model_name],
                        }
                    else:
                        minority_input[norm_key]["weight"] += weight
                        minority_input[norm_key]["models"].append(model_name)
                minority_report = self.minority_report_gen.extract_minority(minority_input, winner)
            else:
                minority_report = self.minority_report_gen.extract_minority(vote_details, winner)
        except Exception as e:
            logger.warning("Minority report generation failed: %s", e)
            minority_report = {}

        # === (t) Decision explanation ===
        try:
            explanation_input = {
                "winner": winner,
                "confidence": confidence,
                "scores": {m: w for m, w in zip(final_models, final_weights)},
                "total_models": len(final_models),
                "weights": {m: w for m, w in zip(final_models, final_weights)},
                "debate_rounds": debate_rounds,
            }
            decision_explanation = self.decision_explainer.explain(
                vote_result=explanation_input,
                debate_transcript=artifact.rounds if artifact else None,
                role_assignments={m: r for m, r in zip(available_models, [
                    next(
                        (role.value for role, assigns in role_assignments.items()
                         if any(a.model_name == m for a in assigns)),
                        "worker"
                    ) for m in available_models
                ])},
            )
        except Exception as e:
            logger.warning("Decision explanation failed: %s", e)
            decision_explanation = ""

        profiler.end_stage("post_vote_analysis")

        # === (u) Dynamic weight update ===
        profiler.start_stage("learning_updates")
        try:
            for model in final_models:
                is_winner = any(
                    r["model"] == model and r["response"] == winner
                    for r in revised_responses
                )
                self.dynamic_weight_manager.record_outcome(model, was_winner=is_winner)
                # Also update MoE router
                self.moe_router.record_outcome(model, query_domain, was_winner=is_winner)
                # Record in confidence calibrator
                self.confidence_calibrator.record(model, confidence, is_winner)
            self.dynamic_weight_manager.update_weights()
        except Exception as e:
            logger.warning("Dynamic weight update failed: %s", e)

        # === (v) Apply emotional drift ===
        try:
            personalities = [profiles.get(m, PersonalityProfile()) for m in final_models]
            for profile_obj, resp in zip(personalities, final_texts):
                agreed = resp.strip().lower() == winner.strip().lower()
                self.emotional_voting.apply_emotional_drift(profile_obj, agent_agreed_with_majority=agreed)
        except Exception as e:
            logger.warning("Emotional drift application failed: %s", e)

        # === (w) Persist personality state ===
        try:
            for model in final_models:
                profile_obj = profiles.get(model)
                if profile_obj:
                    state_to_save = {}
                    for attr in ("stress_level", "confidence_modifier", "engagement_score",
                                 "communication_style", "assertiveness"):
                        if hasattr(profile_obj, attr):
                            state_to_save[attr] = getattr(profile_obj, attr)
                    if state_to_save:
                        self.personality_persistence.save_state(model, state_to_save)
        except Exception as e:
            logger.warning("Personality persistence save failed: %s", e)

        profiler.end_stage("learning_updates")

        # === Store new precedent ===
        new_precedent_id = ""
        if self.precedent_store is not None and winner:
            try:
                new_precedent_id = await self.precedent_store.store_precedent(
                    session_id=run.run_id,
                    query_text=prompt,
                    ruling_text=winner,
                    domain=query_domain,
                    consensus=confidence,
                    models_used=final_models,
                    vote_data=vote_details if isinstance(vote_details, dict) else {},
                    debate_rounds=debate_rounds,
                )
                if new_precedent_id:
                    run.log_event("PRECEDENT_STORED", detail=f"id={new_precedent_id}")
            except Exception as e:
                logger.warning("Precedent storage failed: %s", e)

        # === Hook 3: Supersession check ===
        if (
            precedent_action == PrecedentAction.SOFT_CITE
            and cited_precedents
            and new_precedent_id
            and self.precedent_store is not None
        ):
            try:
                distinguish_count = sum(
                    1 for r in revised_responses
                    if r.get("success", True) and self.precedent_injector.detect_distinguish(r.get("response", ""))
                )
                if distinguish_count > len(revised_responses) / 2:
                    await self.precedent_store.supersede(cited_precedents[0].id, new_precedent_id)
                    run.log_event("PRECEDENT_SUPERSEDED", detail=f"{cited_precedents[0].id} -> {new_precedent_id}")
                    precedent_info = {
                        "action": precedent_action.value,
                        "cited": cited_precedents[0].to_dict(),
                        "disposition": "distinguished",
                        "superseded": True,
                    }
                else:
                    precedent_info = {
                        "action": precedent_action.value,
                        "cited": cited_precedents[0].to_dict(),
                        "disposition": "followed",
                        "superseded": False,
                    }
            except Exception as e:
                logger.warning("Precedent supersession check failed: %s", e)

        # === (x) End profiler, build result ===
        profiler.end_stage("total_pipeline")
        try:
            performance_profile = profiler.get_profile()
        except Exception:
            performance_profile = {}

        # Update circuit breaker states snapshot
        try:
            circuit_breaker_states = self.circuit_breaker.get_all_states()
        except Exception:
            pass

        run.complete({})
        result = self._build_result(
            run,
            revised_responses,
            vote_details,
            role_summary,
            winner=winner,
            confidence=confidence,
            artifact=artifact,
            minority_report=minority_report,
            decision_explanation=decision_explanation,
            performance_profile=performance_profile,
            degradation_mode=degradation_mode_info,
            circuit_breaker_states=circuit_breaker_states,
            reasoning_mode=reasoning_mode,
            rag_context=rag_context,
            coalition_info=coalition_info,
            audit_events=len(self.audit_trail._events),
            precedent_info=precedent_info,
        )
        run.final_result = result
        # Safety: ensure governor is stopped even on unexpected code paths
        await self._stop_governor_safe()
        return result

    async def _stop_governor_safe(self):
        """Stop the concurrency governor, suppressing errors. Idempotent."""
        try:
            await self.concurrency_governor.stop()
        except Exception as e:
            logger.warning("Governor stop failed: %s", e)

    def _build_result(
        self,
        run: ImplementationRun,
        responses: list,
        vote_details: dict,
        role_summary: dict,
        winner: str = "",
        confidence: float = 0.0,
        artifact: DebateArtifact = None,
        minority_report: dict = None,
        decision_explanation: str = "",
        performance_profile: dict = None,
        degradation_mode: dict = None,
        circuit_breaker_states: dict = None,
        reasoning_mode: str = "direct",
        rag_context: dict = None,
        coalition_info: list = None,
        audit_events: int = 0,
        precedent_info: dict = None,
    ) -> dict:
        return {
            "run_id": run.run_id,
            "query": run.query,
            "final_answer": winner or "Error: No consensus reached",
            "confidence": confidence,
            "responses": responses,
            "vote_breakdown": vote_details,
            "role_assignments": role_summary,
            "sub_queries": run.sub_queries,
            "anchored_response_count": len(run.anchored_responses),
            "supervisor_summary": self.supervisor.get_summary(),
            "handoff_log": self.handoff.get_handoff_log(),
            "debate_artifact": artifact.to_dict() if artifact else {},
            "run_context": run.to_dict(),
            "event_log": run.event_log,
            "duration_seconds": run.duration_seconds,
            # New fields from 35 improvements
            "minority_report": minority_report or {},
            "decision_explanation": decision_explanation,
            "performance_profile": performance_profile or {},
            "degradation_mode": degradation_mode or {},
            "circuit_breaker_states": circuit_breaker_states or {},
            "reasoning_mode": reasoning_mode,
            "rag_context": rag_context,
            "coalition_info": [
                {
                    "coalition_id": c.get("coalition_id"),
                    "members": c.get("members", []),
                    "combined_weight": c.get("combined_weight", 0),
                }
                for c in (coalition_info or [])
            ],
            # Precedent-based reasoning (stare decisis)
            "precedent": precedent_info,
            # Paperclip-inspired fields
            "audit_event_count": audit_events,
            "mission_active": self.goal_engine is not None,
        }

    def record_feedback(self, session_id: str, model: str, feedback: str) -> None:
        """Record user feedback for a model response.

        Args:
            session_id: The session identifier (typically run_id).
            model: The model identifier.
            feedback: Either 'positive' or 'negative'.
        """
        try:
            self.feedback_collector.record_feedback(session_id, model, feedback)
        except Exception as e:
            logger.warning("Failed to record feedback: %s", e)

    def get_performance_stats(self) -> dict:
        """Return combined performance statistics from weight manager and calibrator.

        Returns:
            Dict with dynamic_weights, calibration, moe_routing, and adaptive_timeouts.
        """
        stats = {}
        try:
            stats["dynamic_weights"] = self.dynamic_weight_manager.get_performance_stats()
        except Exception as e:
            logger.warning("Failed to get weight stats: %s", e)
            stats["dynamic_weights"] = {}
        try:
            stats["calibration"] = self.confidence_calibrator.get_all_stats()
        except Exception as e:
            logger.warning("Failed to get calibration stats: %s", e)
            stats["calibration"] = {}
        try:
            stats["moe_routing"] = self.moe_router.get_routing_stats()
        except Exception as e:
            logger.warning("Failed to get MoE stats: %s", e)
            stats["moe_routing"] = {}
        try:
            stats["adaptive_timeouts"] = self.adaptive_timeout.get_stats()
        except Exception as e:
            logger.warning("Failed to get timeout stats: %s", e)
            stats["adaptive_timeouts"] = {}
        return stats

    def get_run(self, run_id: str) -> Optional[ImplementationRun]:
        return self._runs.get(run_id)
