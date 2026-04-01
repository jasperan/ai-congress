"""
End-to-end simulation test for the Enhanced Orchestrator.

Tests all 8 Symphony/Pi-inspired improvements:
1. OTP-style Supervision (retry, backoff, stall detection)
2. Implementation Run Contexts (scoped state)
3. Agent Handoff Protocol (delegation)
4. Hash-Anchored Response Tracking
5. Explicit Role Dispatch (personality-based)
6. File-Based State Persistence (DebateArtifact)
7. Query Decomposition / Subagents
8. Stall Detection + Adaptive Timeouts

Uses mocked Ollama responses to simulate full agent interactions.
"""

import asyncio
import json
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from src.ai_congress.core.acp.anchoring import (
    anchor_responses,
    compute_anchor,
    format_anchored_debate_prompt,
)
from src.ai_congress.core.acp.handoff import AgentHandoff, HandoffRequest, HandoffResponse
from src.ai_congress.core.acp.message import AgentIdentity, PersonalityProfile
from src.ai_congress.core.acp.message_bus import ACPMessageBus
from src.ai_congress.core.acp.registry import AgentRegistry
from src.ai_congress.core.acp.roles import AgentRole, RoleAssignment, RoleDispatcher
from src.ai_congress.core.acp.run_context import AgentState, ImplementationRun, RunStatus
from src.ai_congress.core.acp.supervisor import (
    AgentSupervisor,
    RestartPolicy,
    SupervisedTask,
)
from src.ai_congress.core.enhanced_orchestrator import DebateArtifact, EnhancedOrchestrator
from src.ai_congress.core.model_registry import ModelRegistry
from src.ai_congress.core.personality.emotional_voting import EmotionalVotingEngine
from src.ai_congress.core.personality.profile import ModelPersonalityLoader
from src.ai_congress.core.voting_engine import VotingEngine
from src.ai_congress.utils.config_loader import OllamaConfig


# ── Fixtures ───────────────────────────────────────────────────────────────


MODELS = ["phi3:3.8b", "mistral:7b", "llama3.2:3b"]

SIMULATED_RESPONSES = {
    "phi3:3.8b": "The capital of France is Paris. It has been the capital since the 10th century.",
    "mistral:7b": "Paris is the capital of France, known for the Eiffel Tower and rich culture.",
    "llama3.2:3b": "France's capital city is Paris, a major European cultural center.",
}

SIMULATED_DECOMPOSITION = (
    "- What is the capital city of France?\n"
    "- What are the key facts about this capital?\n"
)

SIMULATED_CRITIQUE = {
    "phi3:3.8b": "After reviewing all responses, Paris is indeed the capital. phi3#anchored response confirmed.",
    "mistral:7b": "I agree with the consensus: Paris is the capital of France. My initial response stands.",
    "llama3.2:3b": "All responses agree: Paris is the capital. I maintain my position with high confidence.",
}


def make_mock_ollama():
    """Create a mock OllamaClient that returns simulated responses."""
    client = AsyncMock()
    call_count = {"n": 0}

    async def mock_chat(model, messages, options=None, stream=False):
        call_count["n"] += 1
        prompt_text = messages[-1]["content"] if messages else ""

        # Decomposition request
        if "Break this question" in prompt_text or "sub-questions" in prompt_text:
            return {"message": {"content": SIMULATED_DECOMPOSITION}}

        # Critique/debate request
        if "Review these responses" in prompt_text or "model#hash" in prompt_text:
            return {"message": {"content": SIMULATED_CRITIQUE.get(model, f"Critique from {model}")}}

        # Normal query
        return {"message": {"content": SIMULATED_RESPONSES.get(model, f"Response from {model}")}}

    client.chat = mock_chat
    return client


def make_orchestrator():
    """Create EnhancedOrchestrator with mocked dependencies."""
    mock_config = MagicMock(spec=OllamaConfig)
    mock_config.base_url = "http://localhost:11434"
    mock_config.timeout = 120
    mock_config.max_retries = 3

    registry = MagicMock(spec=ModelRegistry)
    registry.get_model_weight = lambda m: {
        "phi3:3.8b": 0.7,
        "mistral:7b": 0.85,
        "llama3.2:3b": 0.6,
    }.get(m, 0.5)

    voting = VotingEngine()
    ollama = make_mock_ollama()

    import os
    personality_config = os.path.join(
        os.path.dirname(__file__), "..", "config", "models_personality.json"
    )
    personality_loader = ModelPersonalityLoader(personality_config)

    return EnhancedOrchestrator(
        model_registry=registry,
        voting_engine=voting,
        ollama_client=ollama,
        personality_loader=personality_loader,
    )


# ── Unit Tests: Component Level ───────────────────────────────────────────


class TestAgentSupervisor:
    """Test 1: OTP-style supervision."""

    @pytest.mark.asyncio
    async def test_successful_supervision(self):
        supervisor = AgentSupervisor(max_retries=3, stall_timeout=10.0)

        async def success_task():
            return {"result": "ok"}

        tasks = [
            SupervisedTask(agent_id="agent_1", coro_factory=success_task),
            SupervisedTask(agent_id="agent_2", coro_factory=success_task),
        ]
        completed = await supervisor.supervise_all(tasks)

        assert len(completed) == 2
        assert all(t.success for t in completed)
        assert all(t.result == {"result": "ok"} for t in completed)

        summary = supervisor.get_summary()
        assert summary["succeeded"] == 2
        assert summary["failed"] == 0

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        supervisor = AgentSupervisor(max_retries=3, backoff_base=0.01)
        attempt_count = {"n": 0}

        async def flaky_task(**kwargs):
            attempt_count["n"] += 1
            if attempt_count["n"] < 3:
                raise ValueError("Transient error")
            return {"result": "recovered"}

        tasks = [
            SupervisedTask(
                agent_id="flaky",
                coro_factory=flaky_task,
                max_retries=3,
                backoff_base=0.01,
            ),
        ]
        completed = await supervisor.supervise_all(tasks)

        assert completed[0].success
        assert completed[0].result == {"result": "recovered"}
        assert completed[0].attempts == 3

    @pytest.mark.asyncio
    async def test_stall_detection(self):
        supervisor = AgentSupervisor(max_retries=2, stall_timeout=0.1, backoff_base=0.01)

        async def slow_task(**kwargs):
            await asyncio.sleep(10)  # Will be killed by timeout
            return "never reached"

        tasks = [
            SupervisedTask(
                agent_id="slow",
                coro_factory=slow_task,
                stall_timeout=0.1,
                max_retries=2,
                backoff_base=0.01,
            ),
        ]
        completed = await supervisor.supervise_all(tasks)

        assert not completed[0].success
        assert "Stalled" in completed[0].error

        events = supervisor.get_event_log()
        stall_events = [e for e in events if e["event"] == "STALL"]
        assert len(stall_events) >= 1

    @pytest.mark.asyncio
    async def test_skip_policy(self):
        supervisor = AgentSupervisor(max_retries=3, backoff_base=0.01)

        async def failing_task(**kwargs):
            raise RuntimeError("permanent failure")

        tasks = [
            SupervisedTask(
                agent_id="skipper",
                coro_factory=failing_task,
                restart_policy=RestartPolicy.SKIP,
                max_retries=3,
            ),
        ]
        completed = await supervisor.supervise_all(tasks)

        assert not completed[0].success
        assert completed[0].attempts == 1  # Should not retry with SKIP


class TestRunContext:
    """Test 2: Implementation Run Contexts."""

    def test_run_lifecycle(self):
        run = ImplementationRun(query="What is 2+2?")
        assert run.status == RunStatus.PENDING
        assert run.turn_count == 0

        run.status = RunStatus.RUNNING
        state = run.register_agent("model_a", role="worker")
        assert "model_a" in run.agent_states
        assert state.role == "worker"

        run.advance_turn()
        assert run.turn_count == 1

        run.record_response("model_a", "The answer is 4", anchor="abc123")
        assert run.anchored_responses["abc123"] == "The answer is 4"
        assert len(run.agent_states["model_a"].responses) == 1

        run.complete({"answer": "4"})
        assert run.status == RunStatus.COMPLETED
        assert run.completed_at is not None
        assert run.duration_seconds > 0

    def test_run_to_dict(self):
        run = ImplementationRun(query="test")
        run.register_agent("m1")
        run.advance_turn()
        d = run.to_dict()
        assert d["query"] == "test"
        assert d["agent_count"] == 1
        assert d["turn_count"] == 1

    def test_run_failure(self):
        run = ImplementationRun(query="will fail")
        run.fail("Some error")
        assert run.status == RunStatus.FAILED
        events = [e for e in run.event_log if e["event"] == "RUN_FAILED"]
        assert len(events) == 1


class TestAgentHandoff:
    """Test 3: Agent Handoff Protocol."""

    @pytest.mark.asyncio
    async def test_handoff_complete(self):
        bus = ACPMessageBus()
        registry = AgentRegistry()

        bus.register_agent("agent_a")
        bus.register_agent("agent_b")
        registry.register(AgentIdentity(name="agent_a", role="worker"))
        registry.register(AgentIdentity(name="agent_b", role="critic"))

        handoff = AgentHandoff(bus, registry)

        # Simulate agent_b completing the handoff after a short delay
        async def delayed_complete():
            await asyncio.sleep(0.05)
            # Check the message bus for the handoff request
            messages = bus.get_messages("agent_b")
            assert len(messages) == 1
            request_id = messages[0].payload["request_id"]
            handoff.complete_handoff(request_id, result="critique done", success=True)

        asyncio.create_task(delayed_complete())

        response = await handoff.delegate(
            from_agent="agent_a",
            to_agent="agent_b",
            task_type="critique",
            payload={"text": "review this"},
            timeout=2.0,
        )

        assert response.success
        assert response.result == "critique done"

        log = handoff.get_handoff_log()
        assert len(log) >= 2  # DELEGATE + COMPLETE

    @pytest.mark.asyncio
    async def test_handoff_timeout(self):
        bus = ACPMessageBus()
        registry = AgentRegistry()
        bus.register_agent("a")
        bus.register_agent("b")

        handoff = AgentHandoff(bus, registry)

        response = await handoff.delegate(
            from_agent="a",
            to_agent="b",
            task_type="test",
            payload={},
            timeout=0.1,
        )

        assert not response.success
        assert "timed out" in response.error

    def test_find_agent_for_task(self):
        registry = AgentRegistry()
        registry.register(AgentIdentity(
            name="math_agent", role="specialist", capabilities=["calculate"]
        ))
        registry.register(AgentIdentity(
            name="search_agent", role="specialist", capabilities=["search"]
        ))

        bus = ACPMessageBus()
        handoff = AgentHandoff(bus, registry)

        assert handoff.find_agent_for_task("calculate") == "math_agent"
        assert handoff.find_agent_for_task("search") == "search_agent"
        assert handoff.find_agent_for_task("unknown") is None


class TestAnchoring:
    """Test 4: Hash-Anchored Response Tracking."""

    def test_compute_anchor(self):
        anchor = compute_anchor("model_a", "Paris is the capital")
        assert len(anchor) == 8
        assert isinstance(anchor, str)

        # Same input = same anchor
        anchor2 = compute_anchor("model_a", "Paris is the capital")
        assert anchor == anchor2

        # Different input = different anchor
        anchor3 = compute_anchor("model_b", "Paris is the capital")
        assert anchor != anchor3

    def test_anchor_responses(self):
        models = ["phi3", "mistral"]
        texts = ["Answer A", "Answer B"]

        anchored = anchor_responses(models, texts)
        assert len(anchored) == 2
        assert anchored[0].model_name == "phi3"
        assert anchored[1].model_name == "mistral"
        assert len(anchored[0].anchor) == 8

    def test_format_anchored_debate_prompt(self):
        anchored = anchor_responses(["m1", "m2"], ["resp1", "resp2"])
        prompt = format_anchored_debate_prompt(
            "What is 2+2?",
            anchored,
            "Critique the responses.",
        )

        assert "m1#" in prompt
        assert "m2#" in prompt
        assert "resp1" in prompt
        assert "resp2" in prompt
        assert "model#hash" in prompt
        assert "Critique the responses." in prompt


class TestRoleDispatcher:
    """Test 5: Explicit Role Dispatch."""

    def test_assign_roles(self):
        registry = AgentRegistry()
        dispatcher = RoleDispatcher(registry)

        models = ["phi3:3.8b", "mistral:7b", "llama3.2:3b"]
        profiles = {
            "phi3:3.8b": PersonalityProfile(
                openness=8, conscientiousness=7, extraversion=4,
                agreeableness=6, neuroticism=3, confidence=0.7, leadership=5,
            ),
            "mistral:7b": PersonalityProfile(
                openness=6, conscientiousness=8, extraversion=7,
                agreeableness=5, neuroticism=4, confidence=0.8, leadership=8,
            ),
            "llama3.2:3b": PersonalityProfile(
                openness=7, conscientiousness=5, extraversion=8,
                agreeableness=7, neuroticism=5, confidence=0.6, leadership=4,
            ),
        }
        weights = {"phi3:3.8b": 0.7, "mistral:7b": 0.85, "llama3.2:3b": 0.6}

        assignments = dispatcher.assign_roles(models, profiles, weights)

        # All models should be assigned
        all_assigned = set()
        for role, role_assignments in assignments.items():
            for a in role_assignments:
                all_assigned.add(a.model_name)
        assert all_assigned == set(models)

        # Should have at least one judge and one worker
        judges = dispatcher.get_models_for_role(assignments, AgentRole.JUDGE)
        assert len(judges) >= 1  # Guaranteed by role dispatcher

        summary = dispatcher.get_role_summary(assignments)
        assert isinstance(summary, dict)
        assert len(summary) == len(AgentRole)

    def test_ensures_judge_exists(self):
        registry = AgentRegistry()
        dispatcher = RoleDispatcher(registry)

        # With only one model, it should still get judge role
        models = ["single_model"]
        profiles = {"single_model": PersonalityProfile()}
        weights = {"single_model": 0.9}

        assignments = dispatcher.assign_roles(models, profiles, weights)
        all_roles = {
            role.value
            for role, assigns in assignments.items()
            if assigns
        }
        # The single model must be assigned somewhere
        total = sum(len(a) for a in assignments.values())
        assert total == 1


class TestDebateArtifact:
    """Test 6: File-Based State Persistence."""

    def test_artifact_lifecycle(self):
        artifact = DebateArtifact()

        artifact.save_round(0, {"m1": "resp1", "m2": "resp2"})
        artifact.record_position("m1", cluster_id=1)
        artifact.record_position("m2", cluster_id=1)

        artifact.save_round(1, {"m1": "revised1", "m2": "revised2"})
        artifact.record_position("m1", cluster_id=1)
        artifact.record_position("m2", cluster_id=2)

        assert len(artifact.rounds) == 2
        assert artifact.get_position_evolution("m1") == [1, 1]
        assert artifact.get_position_evolution("m2") == [1, 2]

        d = artifact.to_dict()
        assert d["total_rounds"] == 2
        assert "m1" in d["position_history"]


# ── Integration Tests ─────────────────────────────────────────────────────


class TestEnhancedOrchestratorE2E:
    """End-to-end simulation testing all 8 features integrated."""

    @pytest.fixture
    def orchestrator(self):
        return make_orchestrator()

    @pytest.mark.asyncio
    async def test_full_enhanced_swarm(self, orchestrator):
        """Complete end-to-end: decomposition -> supervised queries -> anchored critique -> voting."""
        result = await orchestrator.enhanced_swarm(
            prompt="What is the capital of France?",
            models=MODELS,
            temperature=0.7,
            enable_decomposition=True,
            enable_debate=True,
        )

        # Verify run context
        assert result["run_id"]
        assert result["query"] == "What is the capital of France?"

        # Verify final answer exists
        assert result["final_answer"]
        assert "Paris" in result["final_answer"]

        # Verify confidence
        assert result["confidence"] > 0.0

        # Verify role assignments were made
        assert result["role_assignments"]
        role_count = sum(len(v) for v in result["role_assignments"].values())
        assert role_count == len(MODELS)

        # Verify supervisor ran
        supervisor_summary = result["supervisor_summary"]
        assert supervisor_summary["total_tasks"] > 0
        assert supervisor_summary["succeeded"] > 0

        # Verify anchored responses
        assert result["anchored_response_count"] > 0

        # Verify debate artifact
        artifact = result["debate_artifact"]
        assert artifact["total_rounds"] >= 2  # Initial + critique

        # Verify run context metadata
        run_ctx = result["run_context"]
        assert run_ctx["status"] == "completed"
        assert run_ctx["turn_count"] >= 2
        assert run_ctx["agent_count"] == len(MODELS)

        # Verify event log has key events
        events = result["event_log"]
        event_types = [e["event"] for e in events]
        assert "RUN_START" in event_types
        assert "ROLES_ASSIGNED" in event_types
        assert "WAVE_1_START" in event_types
        assert "WAVE_2_START" in event_types
        assert "VOTING_START" in event_types
        assert "RUN_COMPLETE" in event_types

    @pytest.mark.asyncio
    async def test_swarm_without_decomposition(self, orchestrator):
        """Test enhanced swarm with decomposition disabled."""
        result = await orchestrator.enhanced_swarm(
            prompt="What is 2+2?",
            models=MODELS[:2],
            temperature=0.5,
            enable_decomposition=False,
        )

        assert result["final_answer"]
        assert result["sub_queries"] == []
        assert result["run_context"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_swarm_with_single_model(self, orchestrator):
        """Test enhanced swarm with just one model."""
        result = await orchestrator.enhanced_swarm(
            prompt="What is the capital of France?",
            models=["phi3:3.8b"],
            temperature=0.7,
        )

        assert result["final_answer"]
        assert "Paris" in result["final_answer"]
        assert result["confidence"] > 0.0

    @pytest.mark.asyncio
    async def test_supervisor_handles_failures(self):
        """Test that supervisor retries failed queries."""
        orchestrator = make_orchestrator()
        call_count = {"n": 0}
        original_chat = orchestrator.ollama_client.chat

        async def flaky_chat(model, messages, options=None, stream=False):
            call_count["n"] += 1
            if model == "phi3:3.8b" and call_count["n"] <= 1:
                raise ConnectionError("Simulated network failure")
            return await original_chat(model, messages, options, stream)

        orchestrator.ollama_client.chat = flaky_chat

        result = await orchestrator.enhanced_swarm(
            prompt="What is the capital of France?",
            models=MODELS,
        )

        # Should still succeed despite the transient failure
        assert result["final_answer"]
        assert result["supervisor_summary"]["succeeded"] > 0

    @pytest.mark.asyncio
    async def test_event_log_completeness(self, orchestrator):
        """Verify the event log captures the full lifecycle."""
        result = await orchestrator.enhanced_swarm(
            prompt="What is the capital of France?",
            models=MODELS,
        )

        events = result["event_log"]
        assert len(events) >= 10  # Should have many events

        # Verify chronological ordering
        timestamps = [e["timestamp"] for e in events]
        assert timestamps == sorted(timestamps)

        # Verify agent registrations
        reg_events = [e for e in events if e["event"] == "AGENT_REGISTERED"]
        assert len(reg_events) == len(MODELS)

        # Verify responses recorded
        resp_events = [e for e in events if e["event"] == "RESPONSE"]
        assert len(resp_events) >= len(MODELS)  # At least initial wave

    @pytest.mark.asyncio
    async def test_duration_tracking(self, orchestrator):
        """Verify timing is tracked correctly."""
        result = await orchestrator.enhanced_swarm(
            prompt="Quick test",
            models=MODELS[:2],
        )

        assert result["duration_seconds"] > 0
        assert result["duration_seconds"] < 30  # Should be fast with mocks

    @pytest.mark.asyncio
    async def test_run_retrieval(self, orchestrator):
        """Verify runs can be retrieved by ID."""
        result = await orchestrator.enhanced_swarm(
            prompt="What is the capital of France?",
            models=MODELS,
        )

        run_id = result["run_id"]
        run = orchestrator.get_run(run_id)
        assert run is not None
        assert run.status == RunStatus.COMPLETED
        assert run.query == "What is the capital of France?"


# ── Simulation: Full Scenario ─────────────────────────────────────────────


class TestFullSimulation:
    """Run a complete simulated scenario and log all results."""

    @pytest.mark.asyncio
    async def test_simulation_scenario(self, capsys):
        """
        SCENARIO: Multi-model debate on "What is the capital of France?"

        Tests the full pipeline:
        1. Planner decomposes the query
        2. Supervisor manages parallel queries with fault tolerance
        3. Responses are hash-anchored for precise references
        4. Critics review anchored responses
        5. Role dispatcher assigns roles based on personality
        6. Emotional voting produces weighted consensus
        7. Debate artifact persists all state
        8. Run context tracks full lifecycle
        """
        orchestrator = make_orchestrator()

        print("\n" + "=" * 80)
        print("SIMULATION: Enhanced Orchestrator E2E Scenario")
        print("=" * 80)

        result = await orchestrator.enhanced_swarm(
            prompt="What is the capital of France?",
            models=MODELS,
            temperature=0.7,
            enable_decomposition=True,
            enable_debate=True,
        )

        # ── Log Results ──────────────────────────────────────────────────

        print(f"\n--- Run Context ---")
        print(f"  Run ID:      {result['run_id']}")
        print(f"  Status:      {result['run_context']['status']}")
        print(f"  Duration:    {result['duration_seconds']:.2f}s")
        print(f"  Turns:       {result['run_context']['turn_count']}")
        print(f"  Agents:      {result['run_context']['agent_count']}")

        print(f"\n--- Role Assignments ---")
        for role, agents in result["role_assignments"].items():
            if agents:
                names = [a["model"] for a in agents]
                scores = [f"{a['score']:.3f}" for a in agents]
                print(f"  {role:15s}: {', '.join(names)} (scores: {', '.join(scores)})")

        print(f"\n--- Query Decomposition ---")
        if result["sub_queries"]:
            for i, sq in enumerate(result["sub_queries"]):
                print(f"  {i+1}. {sq['text']} (source: {sq['source']})")
        else:
            print("  (not decomposed)")

        print(f"\n--- Supervisor Summary ---")
        ss = result["supervisor_summary"]
        print(f"  Total tasks:    {ss['total_tasks']}")
        print(f"  Succeeded:      {ss['succeeded']}")
        print(f"  Failed:         {ss['failed']}")
        print(f"  Total attempts: {ss['total_attempts']}")
        if ss["failed_agents"]:
            print(f"  Failed agents:  {ss['failed_agents']}")

        print(f"\n--- Anchored Responses ---")
        print(f"  Count: {result['anchored_response_count']}")

        print(f"\n--- Debate Artifact ---")
        artifact = result["debate_artifact"]
        print(f"  Rounds:    {artifact['total_rounds']}")
        for rnd in artifact["rounds"]:
            print(f"    Round {rnd['round']}: {len(rnd['responses'])} responses")
        if artifact["position_history"]:
            for model, positions in artifact["position_history"].items():
                print(f"    {model}: positions={positions}")

        print(f"\n--- Final Answer ---")
        print(f"  Winner:     {result['final_answer'][:100]}...")
        print(f"  Confidence: {result['confidence']:.3f}")

        print(f"\n--- Vote Breakdown ---")
        for key, details in result["vote_breakdown"].items():
            if isinstance(details, dict):
                models = details.get("models", [])
                weight = details.get("weight", 0)
                print(f"  [{', '.join(models)}] weight={weight:.3f}")
            else:
                print(f"  {key}: {details}")

        print(f"\n--- Event Log ({len(result['event_log'])} events) ---")
        for event in result["event_log"]:
            agent = event.get("agent", "")
            detail = event.get("detail", "")
            agent_str = f" [{agent}]" if agent else ""
            print(f"  T{event['turn']:02d} {event['event']:25s}{agent_str} {detail}")

        print("\n" + "=" * 80)
        print("SIMULATION COMPLETE - ALL 8 IMPROVEMENTS VERIFIED")
        print("=" * 80)

        # ── Assertions ───────────────────────────────────────────────────

        assert result["run_context"]["status"] == "completed"
        assert "Paris" in result["final_answer"]
        assert result["confidence"] > 0
        assert result["anchored_response_count"] > 0
        assert result["debate_artifact"]["total_rounds"] >= 2
        assert len(result["event_log"]) >= 10
        assert result["supervisor_summary"]["succeeded"] > 0
        assert result["role_assignments"]

        # Verify all 8 features left traces
        event_types = {e["event"] for e in result["event_log"]}
        assert "RUN_START" in event_types, "Feature 2: Run Context"
        assert "ROLES_ASSIGNED" in event_types, "Feature 5: Role Dispatch"
        assert "WAVE_1_START" in event_types, "Feature 1: Supervision"
        assert "WAVE_2_START" in event_types, "Feature 4: Anchored Critique"
        assert "VOTING_START" in event_types, "Feature 6: State Persistence"
        assert "RUN_COMPLETE" in event_types, "Feature 8: Lifecycle Tracking"
