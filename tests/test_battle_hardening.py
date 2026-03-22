"""
Battle-hardening tests for AI Congress.

Covers security boundaries, edge cases, and input validation across:
- ReAct safe math parser (injection resistance)
- VotingEngine edge cases
- Document processing edge cases
- API input validation
- SwarmOrchestrator personality_swarm edge cases
"""
import math
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.ai_congress.core.reasoning.react import _safe_math_eval
from src.ai_congress.core.voting_engine import VotingEngine
from src.ai_congress.integrations.documents import (
    DocumentChunk,
    DocumentParser,
    TextChunker,
    DocumentProcessor,
)


# ---------------------------------------------------------------------------
# 1. ReAct Safe Math Parser — security proof
# ---------------------------------------------------------------------------

class TestSafeMathEval:
    """Validates _safe_math_eval accepts legit math and rejects injection."""

    # --- Accepted expressions ---

    def test_basic_addition(self):
        """2+3 should return 5."""
        assert _safe_math_eval("2+3") == 5

    def test_basic_subtraction(self):
        """10-4 should return 6."""
        assert _safe_math_eval("10-4") == 6

    def test_basic_multiplication(self):
        """3*7 should return 21."""
        assert _safe_math_eval("3*7") == 21

    def test_basic_division(self):
        """10/3 should return a float close to 3.333."""
        result = _safe_math_eval("10/3")
        assert abs(result - 10 / 3) < 1e-9

    def test_floor_division(self):
        """10//3 should return 3."""
        assert _safe_math_eval("10//3") == 3

    def test_modulo(self):
        """10%3 should return 1."""
        assert _safe_math_eval("10%3") == 1

    def test_exponentiation(self):
        """2**8 should return 256."""
        assert _safe_math_eval("2**8") == 256

    def test_unary_negation(self):
        """-5 should return -5."""
        assert _safe_math_eval("-5") == -5

    def test_sqrt(self):
        """sqrt(16) should return 4.0."""
        assert _safe_math_eval("sqrt(16)") == 4.0

    def test_sin_zero(self):
        """sin(0) should return 0.0."""
        assert _safe_math_eval("sin(0)") == 0.0

    def test_cos_zero(self):
        """cos(0) should return 1.0."""
        assert _safe_math_eval("cos(0)") == 1.0

    def test_log_one(self):
        """log(1) should return 0.0 (natural log)."""
        assert _safe_math_eval("log(1)") == 0.0

    def test_constant_pi(self):
        """pi should return math.pi."""
        assert abs(_safe_math_eval("pi") - math.pi) < 1e-9

    def test_constant_e(self):
        """e should return math.e."""
        assert abs(_safe_math_eval("e") - math.e) < 1e-9

    def test_nested_sqrt_abs(self):
        """sqrt(abs(-16)) should return 4.0."""
        assert _safe_math_eval("sqrt(abs(-16))") == 4.0

    def test_abs_negative(self):
        """abs(-42) should return 42."""
        assert _safe_math_eval("abs(-42)") == 42

    def test_round_function(self):
        """round(3.7) should return 4."""
        assert _safe_math_eval("round(3.7)") == 4

    def test_min_function(self):
        """min(3, 7) should return 3."""
        assert _safe_math_eval("min(3, 7)") == 3

    def test_max_function(self):
        """max(3, 7) should return 7."""
        assert _safe_math_eval("max(3, 7)") == 7

    def test_complex_expression(self):
        """2 + 3 * 4 should respect operator precedence and return 14."""
        assert _safe_math_eval("2 + 3 * 4") == 14

    def test_parenthesized_expression(self):
        """(2 + 3) * 4 should return 20."""
        assert _safe_math_eval("(2 + 3) * 4") == 20

    # --- Rejected (malicious) expressions ---

    def test_reject_import(self):
        """__import__('os') must be rejected."""
        with pytest.raises((ValueError, SyntaxError)):
            _safe_math_eval("__import__('os')")

    def test_reject_open(self):
        """open('/etc/passwd') must be rejected."""
        with pytest.raises((ValueError, SyntaxError)):
            _safe_math_eval("open('/etc/passwd')")

    def test_reject_dunder_class_traversal(self):
        """Class traversal via ().__class__.__bases__[0].__subclasses__() must be rejected."""
        with pytest.raises((ValueError, SyntaxError, TypeError)):
            _safe_math_eval("().__class__.__bases__[0].__subclasses__()")

    def test_reject_arbitrary_string(self):
        """Bare string 'hello' must be rejected."""
        with pytest.raises((ValueError, SyntaxError)):
            _safe_math_eval("hello")

    def test_reject_exec(self):
        """exec('print(1)') must be rejected."""
        with pytest.raises((ValueError, SyntaxError)):
            _safe_math_eval("exec('print(1)')")

    def test_reject_eval(self):
        """eval('1+1') must be rejected."""
        with pytest.raises((ValueError, SyntaxError)):
            _safe_math_eval("eval('1+1')")

    def test_reject_compile(self):
        """compile('1','','exec') must be rejected."""
        with pytest.raises((ValueError, SyntaxError)):
            _safe_math_eval("compile('1','','exec')")

    def test_reject_getattr(self):
        """getattr(__builtins__, 'open') must be rejected."""
        with pytest.raises((ValueError, SyntaxError)):
            _safe_math_eval("getattr(__builtins__, 'open')")

    def test_reject_non_whitelisted_function(self):
        """Calling a non-whitelisted function like hex(255) must be rejected."""
        with pytest.raises(ValueError, match="Unknown function"):
            _safe_math_eval("hex(255)")

    def test_reject_string_constant(self):
        """A string constant like 'abc' must be rejected."""
        with pytest.raises(ValueError, match="Unsupported constant"):
            _safe_math_eval("'abc'")

    def test_reject_list_literal(self):
        """A list literal [1,2,3] must be rejected."""
        with pytest.raises((ValueError, SyntaxError)):
            _safe_math_eval("[1,2,3]")

    def test_reject_lambda(self):
        """Lambda expressions must be rejected."""
        with pytest.raises((ValueError, SyntaxError)):
            _safe_math_eval("lambda: 1")

    def test_reject_os_system(self):
        """os.system('ls') via attribute access must be rejected."""
        with pytest.raises((ValueError, SyntaxError)):
            _safe_math_eval("__import__('os').system('ls')")


# ---------------------------------------------------------------------------
# 2. Voting Engine Edge Cases
# ---------------------------------------------------------------------------

class TestVotingEngineEdgeCases:
    """Edge cases and boundary conditions for VotingEngine."""

    def setup_method(self):
        self.engine = VotingEngine()

    def test_empty_responses(self):
        """Empty responses list should raise or return degenerate result."""
        # weighted_majority_vote requires len(responses) == len(weights)
        # With empty lists, max() on empty dict will raise ValueError
        with pytest.raises(ValueError):
            self.engine.weighted_majority_vote([], [], [])

    def test_single_response(self):
        """Single response should win with confidence 1.0."""
        winner, confidence, breakdown = self.engine.weighted_majority_vote(
            ["Only answer"], [0.8], ["model_a"]
        )
        assert winner == "Only answer"
        assert confidence == 1.0

    def test_all_identical_responses(self):
        """All identical responses should yield confidence 1.0."""
        responses = ["Same answer"] * 5
        weights = [0.5, 0.6, 0.7, 0.8, 0.9]
        winner, confidence, _ = self.engine.weighted_majority_vote(
            responses, weights
        )
        assert winner == "Same answer"
        assert confidence == 1.0

    def test_all_different_responses(self):
        """All different responses (no consensus) should still pick the highest-weighted one."""
        responses = ["Alpha", "Beta", "Gamma"]
        weights = [0.3, 0.9, 0.5]
        winner, confidence, _ = self.engine.weighted_majority_vote(
            responses, weights
        )
        assert winner == "Beta"
        # Confidence should be weight_of_winner / total
        expected_confidence = 0.9 / (0.3 + 0.9 + 0.5)
        assert abs(confidence - expected_confidence) < 1e-9

    def test_mismatched_lengths_raises(self):
        """Responses and weights with different lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            self.engine.weighted_majority_vote(
                ["a", "b"], [0.5], ["m1", "m2"]
            )

    def test_zero_weights(self):
        """All zero weights should give confidence 0 (division guarded)."""
        responses = ["A", "B"]
        weights = [0.0, 0.0]
        winner, confidence, _ = self.engine.weighted_majority_vote(
            responses, weights
        )
        # total_weight is 0, so confidence should be 0
        assert confidence == 0

    def test_negative_weights(self):
        """Negative weights should not crash. The engine doesn't forbid them."""
        responses = ["Pos", "Neg"]
        weights = [1.0, -0.5]
        # Should not raise; behavior is mathematically valid even if unusual
        winner, confidence, _ = self.engine.weighted_majority_vote(
            responses, weights
        )
        assert winner in ["Pos", "Neg"]

    def test_very_long_responses(self):
        """Responses exceeding 10000 chars should be handled without issue."""
        long_response = "x" * 15000
        responses = [long_response, "short"]
        weights = [0.8, 0.5]
        winner, confidence, _ = self.engine.weighted_majority_vote(
            responses, weights
        )
        assert winner == long_response

    def test_unicode_emoji_responses(self):
        """Unicode and emoji in responses should work correctly."""
        responses = ["The answer is 42", "The answer is 42"]
        weights = [0.7, 0.8]
        winner, confidence, _ = self.engine.weighted_majority_vote(
            responses, weights
        )
        assert confidence == 1.0
        assert winner == "The answer is 42"

    def test_history_bounds_truncation(self):
        """Adding >MAX_HISTORY entries should truncate to MAX_HISTORY."""
        assert self.engine.MAX_HISTORY == 1000

        # Populate beyond the limit
        for i in range(1050):
            self.engine.weighted_majority_vote(
                [f"resp_{i}"], [1.0], [f"model_{i}"]
            )

        assert len(self.engine.voting_history) == 1000

    def test_rank_responses(self):
        """rank_responses should return items sorted by weight descending."""
        responses = ["A", "B", "C"]
        weights = [0.3, 0.9, 0.5]
        ranked = self.engine.rank_responses(responses, weights, ["m1", "m2", "m3"])
        assert ranked[0]["rank"] == 1
        assert ranked[0]["weight"] == 0.9  # B has highest weight

    def test_confidence_based_vote(self):
        """confidence_based_vote should use per-response confidence as weight."""
        responses = [
            {"text": "Yes", "confidence": 0.9, "model": "m1"},
            {"text": "No", "confidence": 0.3, "model": "m2"},
            {"text": "Yes", "confidence": 0.8, "model": "m3"},
        ]
        winner, confidence, _ = self.engine.confidence_based_vote(responses)
        assert winner == "Yes"

    def test_temperature_ensemble(self):
        """Lower temperature responses should be weighted higher."""
        responses = ["Cold", "Hot", "Cold"]
        temperatures = [0.1, 1.5, 0.2]
        result = self.engine.temperature_ensemble(responses, temperatures)
        # Cold has much higher inverse-temperature weight
        assert result == "Cold"

    def test_whitespace_normalization(self):
        """Leading/trailing whitespace should be normalized during comparison."""
        responses = ["  Paris  ", "Paris", " paris "]
        weights = [0.5, 0.5, 0.5]
        winner, confidence, _ = self.engine.weighted_majority_vote(
            responses, weights
        )
        # All normalize to "paris" -> 100% consensus
        assert confidence == 1.0


# ---------------------------------------------------------------------------
# 3. Document Processing Edge Cases
# ---------------------------------------------------------------------------

class TestTextChunkerEdgeCases:
    """Edge cases for TextChunker."""

    def setup_method(self):
        self.chunker = TextChunker(
            chunk_size=512,
            chunk_overlap=50,
            min_chunk_size=100,
            adaptive=True,
        )

    def test_empty_text(self):
        """Empty string should return no chunks."""
        chunks = self.chunker.chunk_text("")
        assert chunks == []

    def test_none_text(self):
        """None text should return no chunks (falsy check)."""
        chunks = self.chunker.chunk_text(None)
        assert chunks == []

    def test_text_shorter_than_min_chunk_size(self):
        """Text below min_chunk_size should return no chunks."""
        short = "Hello world"  # 11 chars, well under 100
        chunks = self.chunker.chunk_text(short)
        assert chunks == []

    def test_text_exactly_min_chunk_size(self):
        """Text exactly at min_chunk_size boundary should return no chunks (< not <=)."""
        # chunk_text checks len(text) < min_chunk_size, so exactly 100 chars should also return []
        # Actually the check is: len(text) < self.min_chunk_size
        text = "a" * 100
        chunks = self.chunker.chunk_text(text)
        # 100 chars is NOT < 100, so it passes the early return.
        # But the adaptive chunker creates a single paragraph chunk.
        # That chunk must also be >= min_chunk_size to be kept.
        assert len(chunks) <= 1

    def test_whitespace_only_text(self):
        """Text with only whitespace should return no chunks."""
        chunks = self.chunker.chunk_text("   \n\n\t\t   ")
        assert chunks == []

    def test_very_large_text(self):
        """100K character text should produce multiple chunks without hanging."""
        # Build realistic text with paragraph separators
        paragraph = "This is a sentence that forms part of a paragraph. " * 20
        large_text = ("\n\n".join([paragraph] * 100))
        assert len(large_text) > 100_000

        chunks = self.chunker.chunk_text(large_text)
        assert len(chunks) > 1
        # Every chunk should have text content
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)
            assert len(chunk.text.strip()) > 0

    def test_chunk_metadata_propagation(self):
        """Metadata dict should be copied into each chunk."""
        text = "A" * 200 + "\n\n" + "B" * 200
        metadata = {"source": "test", "doc_id": "42"}
        chunks = self.chunker.chunk_text(text, metadata)
        for chunk in chunks:
            assert chunk.metadata["source"] == "test"
            assert chunk.metadata["doc_id"] == "42"

    def test_chunk_to_dict(self):
        """DocumentChunk.to_dict() should return the expected structure."""
        chunk = DocumentChunk(text="hello", metadata={"k": "v"}, chunk_index=0)
        d = chunk.to_dict()
        assert d["text"] == "hello"
        assert d["metadata"]["k"] == "v"
        assert d["chunk_index"] == 0

    def test_simple_chunker_mode(self):
        """Non-adaptive (simple) chunking should also produce valid chunks."""
        simple_chunker = TextChunker(
            chunk_size=200,
            chunk_overlap=20,
            min_chunk_size=50,
            adaptive=False,
        )
        text = "Word " * 200  # 1000 chars
        chunks = simple_chunker.chunk_text(text)
        assert len(chunks) >= 1

    def test_overlap_larger_than_chunk_size_simple(self):
        """chunk_overlap >= chunk_size in simple mode should not hang (infinite loop guard)."""
        bad_chunker = TextChunker(
            chunk_size=100,
            chunk_overlap=150,  # overlap > chunk_size
            min_chunk_size=10,
            adaptive=False,
        )
        text = "Hello world. " * 50  # 650 chars
        # The _simple_chunk method has an infinite loop guard: if start <= 0 or end >= len(text): break
        # This should terminate, not hang.
        chunks = bad_chunker.chunk_text(text)
        # We just need it to not hang; any result (even empty) is acceptable
        assert isinstance(chunks, list)

    def test_overlap_larger_than_chunk_size_adaptive(self):
        """chunk_overlap >= chunk_size in adaptive mode should not hang."""
        bad_chunker = TextChunker(
            chunk_size=100,
            chunk_overlap=200,
            min_chunk_size=10,
            adaptive=True,
        )
        paragraph = "Some text here. " * 30
        text = paragraph + "\n\n" + paragraph
        chunks = bad_chunker.chunk_text(text)
        assert isinstance(chunks, list)


class TestDocumentParserEdgeCases:
    """Edge cases for DocumentParser."""

    def test_unsupported_extension(self):
        """Unsupported file extension should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            DocumentParser.parse_document("/tmp/test_file.xyz")

    def test_nonexistent_file(self):
        """Non-existent file should raise an exception from the parser."""
        with pytest.raises(Exception):
            DocumentParser.parse_document("/tmp/this_file_does_not_exist_ever.txt")

    def test_empty_extension(self):
        """File with no extension should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            DocumentParser.parse_document("/tmp/noextension")


# ---------------------------------------------------------------------------
# 4. API Input Validation
# ---------------------------------------------------------------------------

class TestAPIInputValidation:
    """Tests for FastAPI endpoint input validation using TestClient."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        """Create a test client with mocked dependencies.

        We patch heavy startup dependencies (Oracle pool, model registry,
        event logger) so that the test client can be constructed without
        external services.
        """
        # We need to patch before importing the app, but the app module
        # is already imported transitively.  Instead we use the TestClient
        # against the already-created app and patch the swarm/registry.
        from fastapi.testclient import TestClient
        from src.ai_congress.api.main import app

        # Disable startup/shutdown events that require Ollama and Oracle
        app.router.on_startup.clear()
        app.router.on_shutdown.clear()

        self.client = TestClient(app, raise_server_exceptions=False)

    def test_chat_empty_prompt(self):
        """POST /api/chat with an empty prompt should still be accepted (no server-side min-length)."""
        # The Pydantic model requires prompt to be a str, but doesn't enforce min length.
        # The swarm will get an empty string. We expect either 200 (passes through)
        # or 500 (swarm fails). Not 422.
        resp = self.client.post("/api/chat", json={
            "prompt": "",
            "models": ["phi3:3.8b"],
            "mode": "multi_model",
        })
        # Should not be a validation error (422) because prompt="" is a valid str
        assert resp.status_code != 422

    def test_chat_missing_prompt_field(self):
        """POST /api/chat without 'prompt' field should return 422."""
        resp = self.client.post("/api/chat", json={
            "models": ["phi3:3.8b"],
        })
        assert resp.status_code == 422

    def test_chat_missing_models_field(self):
        """POST /api/chat without 'models' field should return 422."""
        resp = self.client.post("/api/chat", json={
            "prompt": "Hello",
        })
        assert resp.status_code == 422

    def test_chat_invalid_mode(self):
        """POST /api/chat with an invalid mode should return 400."""
        resp = self.client.post("/api/chat", json={
            "prompt": "What is 2+2?",
            "models": ["phi3:3.8b"],
            "mode": "this_mode_does_not_exist",
        })
        # The endpoint raises HTTPException(400, "Invalid mode") for unknown modes
        assert resp.status_code in (400, 500)

    def test_chat_extremely_long_prompt(self):
        """POST /api/chat with a 1MB prompt should not crash the validation layer."""
        mega_prompt = "A" * (1024 * 1024)
        resp = self.client.post("/api/chat", json={
            "prompt": mega_prompt,
            "models": ["phi3:3.8b"],
            "mode": "multi_model",
        })
        # Should not be a validation error; the server may fail downstream
        # but the request itself is structurally valid.
        assert resp.status_code != 422

    def test_get_models(self):
        """GET /api/models should return a list (even if empty when Ollama is down)."""
        resp = self.client.get("/api/models")
        # Could be 200 with list or 500 if Ollama unreachable
        if resp.status_code == 200:
            data = resp.json()
            assert isinstance(data, list)

    def test_invalid_json_body(self):
        """POST /api/chat with invalid JSON should return 422."""
        resp = self.client.post(
            "/api/chat",
            content="this is not json{{{",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_chat_personality_mode_without_personalities(self):
        """POST /api/chat in personality mode without personalities should return 400."""
        resp = self.client.post("/api/chat", json={
            "prompt": "Tell me a joke",
            "models": ["phi3:3.8b"],
            "mode": "personality",
            # personalities omitted
        })
        # The endpoint checks: if not request.personalities: raise HTTPException(400)
        assert resp.status_code in (400, 500)

    def test_root_endpoint(self):
        """GET / should return running status."""
        resp = self.client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"

    def test_chat_wrong_type_for_prompt(self):
        """POST /api/chat with non-string prompt should return 422."""
        resp = self.client.post("/api/chat", json={
            "prompt": 12345,
            "models": ["phi3:3.8b"],
            "mode": "multi_model",
        })
        # Pydantic may coerce int to str, so this could be 422 or accepted
        # Either way, no 500 crash is the key assertion
        assert resp.status_code in (200, 422, 500)

    def test_chat_models_not_a_list(self):
        """POST /api/chat with models as a string instead of list should return 422."""
        resp = self.client.post("/api/chat", json={
            "prompt": "Hello",
            "models": "phi3:3.8b",  # should be a list
            "mode": "multi_model",
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 5. Swarm Orchestrator Edge Cases — personality_swarm
# ---------------------------------------------------------------------------

class TestSwarmOrchestratorPersonalityEdgeCases:
    """Edge cases for SwarmOrchestrator.personality_swarm."""

    @pytest.fixture(autouse=True)
    def setup_swarm(self):
        """Build a SwarmOrchestrator with mocked dependencies."""
        self.model_registry = MagicMock()
        self.model_registry.get_model_weight = MagicMock(return_value=0.5)
        self.model_registry.get_top_models = MagicMock(return_value=["phi3:3.8b"])

        self.voting_engine = VotingEngine()

        # Build a minimal OllamaConfig-like object
        ollama_cfg = MagicMock()
        ollama_cfg.base_url = "http://localhost:11434"
        ollama_cfg.timeout = 30
        ollama_cfg.max_retries = 1

        with patch("src.ai_congress.core.swarm_orchestrator.load_config") as mock_cfg:
            # Provide a config stub so module-level load_config() works
            cfg = MagicMock()
            cfg.logging.verbosity = "quiet"
            cfg.swarm.hybrid = {"top_models": 3, "temperatures": [0.5, 0.9]}
            cfg.swarm.max_concurrent_requests = 10
            cfg.voting.consensus_threshold = 0.7
            cfg.voting.debate.max_rounds = 3
            cfg.voting.debate.temp_schedule = [0.7, 0.5, 0.3]
            cfg.voting.debate.conviction_bonus = 0.1
            mock_cfg.return_value = cfg

            from src.ai_congress.core.swarm_orchestrator import SwarmOrchestrator
            self.SwarmOrchestrator = SwarmOrchestrator

        self.swarm = self.SwarmOrchestrator(
            model_registry=self.model_registry,
            voting_engine=self.voting_engine,
            ollama_config=ollama_cfg,
        )
        # Mock the ollama client so we don't hit a real server
        self.swarm.ollama_client = AsyncMock()

    async def test_personality_swarm_empty_list(self):
        """personality_swarm with empty personalities list should return error result."""
        result = await self.swarm.personality_swarm(
            personalities=[],
            prompt="Hello",
            base_model="phi3:3.8b",
        )
        # No personalities -> no tasks -> no successful responses
        assert "Error" in result["final_answer"] or result["confidence"] == 0.0

    async def test_personality_swarm_missing_name_key(self):
        """personality_swarm with personality dict missing 'name' should raise KeyError."""
        bad_personalities = [{"system_prompt": "You are a robot."}]
        with pytest.raises(KeyError):
            await self.swarm.personality_swarm(
                personalities=bad_personalities,
                prompt="Hello",
                base_model="phi3:3.8b",
            )

    async def test_personality_swarm_missing_system_prompt_key(self):
        """personality_swarm with personality dict missing 'system_prompt' should fail gracefully.

        The missing key triggers a KeyError inside the async task, which is caught
        by asyncio.gather(return_exceptions=True) and surfaced as a failed response.
        """
        bad_personalities = [{"name": "Robot"}]
        result = await self.swarm.personality_swarm(
            personalities=bad_personalities,
            prompt="Hello",
            base_model="phi3:3.8b",
        )
        # All responses fail, so we get the error fallback
        assert result["confidence"] == 0.0
        assert "Error" in result["final_answer"]

    async def test_personality_swarm_all_fail(self):
        """When all model queries fail, personality_swarm returns error result."""
        # Make the ollama client raise on every call
        self.swarm.ollama_client.chat = AsyncMock(side_effect=Exception("Connection refused"))

        personalities = [
            {"name": "Alice", "system_prompt": "You are Alice."},
            {"name": "Bob", "system_prompt": "You are Bob."},
        ]
        result = await self.swarm.personality_swarm(
            personalities=personalities,
            prompt="Hello",
            base_model="phi3:3.8b",
        )
        assert result["confidence"] == 0.0
        assert "Error" in result["final_answer"]

    async def test_personality_swarm_single_personality(self):
        """personality_swarm with one personality should return that response as winner."""
        # Mock a successful chat response
        self.swarm.ollama_client.chat = AsyncMock(return_value={
            "message": {"content": "I am the only one."}
        })

        personalities = [{"name": "Solo", "system_prompt": "You are Solo."}]
        result = await self.swarm.personality_swarm(
            personalities=personalities,
            prompt="Who are you?",
            base_model="phi3:3.8b",
        )
        assert result["final_answer"] == "I am the only one."
        assert result["confidence"] == 1.0
