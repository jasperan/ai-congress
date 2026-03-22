"""
Tests for the Adaptive Query Router.

Covers:
- ComplexityEstimator scoring for simple, medium, and complex queries
- RoutingDecision validation and immutability
- AdaptiveQueryRouter routing logic across all three modes
- Edge cases: empty query, very long query, unicode, single model available
"""
import pytest
from src.ai_congress.core.adaptive_router import (
    ComplexityEstimator,
    RoutingDecision,
    AdaptiveQueryRouter,
)

AVAILABLE_MODELS = [
    "phi3:3.8b",
    "mistral:7b",
    "llama3.2:3b",
    "deepseek-r1:1.5b",
    "qwen3:0.6b",
]


# ---------------------------------------------------------------------------
# ComplexityEstimator tests
# ---------------------------------------------------------------------------

class TestComplexityEstimator:
    def setup_method(self):
        self.estimator = ComplexityEstimator()

    # Simple queries should score low (< 0.3)
    @pytest.mark.parametrize("query", [
        "What is Python?",
        "Who is Alan Turing?",
        "When is Christmas?",
        "Where is France?",
        "Define entropy",
        "Translate hello",
        "Convert 5 miles",
        "2 + 2",
        "yes",
        "no",
        "true",
    ])
    def test_simple_queries_score_low(self, query):
        score = self.estimator.estimate(query)
        assert score < 0.3, f"Simple query '{query}' scored {score:.3f}, expected < 0.3"

    # Complex queries should score high (>= 0.6)
    @pytest.mark.parametrize("query", [
        "Compare and contrast the pros and cons of microservices versus monolithic architecture in detail",
        "Analyze the advantages and disadvantages of renewable energy sources step by step",
        "Design a comprehensive system that evaluates how does distributed consensus work, "
        "and explain the relationship between Raft and Paxos algorithms",
        "Evaluate the pros and cons of Python vs Rust, analyze their performance characteristics, "
        "and architect a hybrid solution that uses both languages effectively",
    ])
    def test_complex_queries_score_high(self, query):
        score = self.estimator.estimate(query)
        assert score >= 0.6, f"Complex query scored {score:.3f}, expected >= 0.6"

    # Medium queries should land in the middle range
    @pytest.mark.parametrize("query", [
        "How does garbage collection work in Java?",
        "Why is Rust memory safe without a garbage collector?",
        "Explain the relationship between Docker and Kubernetes",
    ])
    def test_medium_queries_score_mid(self, query):
        score = self.estimator.estimate(query)
        assert 0.15 <= score < 0.75, f"Medium query scored {score:.3f}, expected 0.15-0.75"

    def test_empty_query_returns_zero(self):
        assert self.estimator.estimate("") == 0.0
        assert self.estimator.estimate("   ") == 0.0

    def test_very_long_query_scores_higher(self):
        short = "What is AI?"
        long_query = "Explain " + " and ".join([f"concept_{i}" for i in range(60)])
        short_score = self.estimator.estimate(short)
        long_score = self.estimator.estimate(long_query)
        assert long_score > short_score, "Longer queries should score higher"

    def test_unicode_query_does_not_crash(self):
        score = self.estimator.estimate("Qu'est-ce que l'intelligence artificielle?")
        assert 0.0 <= score <= 1.0

    def test_emoji_query_does_not_crash(self):
        score = self.estimator.estimate("What is love? \U0001f496")
        assert 0.0 <= score <= 1.0

    def test_score_clamped_to_range(self):
        # Even an extremely complex query shouldn't exceed 1.0
        extreme = (
            "Compare and contrast, analyze, evaluate, design, and architect "
            "the comprehensive step-by-step pros and cons of every approach, "
            "and explain the relationship between all of them? "
            "Why? How does it work? " * 10
        )
        score = self.estimator.estimate(extreme)
        assert 0.0 <= score <= 1.0

    def test_multiple_questions_increase_score(self):
        one_q = "What is Python?"
        multi_q = "What is Python? How does it compare to Java? Why should I learn it?"
        assert self.estimator.estimate(multi_q) > self.estimator.estimate(one_q)

    def test_clause_markers_increase_score(self):
        simple = "Tell me about dogs"
        claused = "Tell me about dogs, cats, birds, and fish, but also reptiles"
        assert self.estimator.estimate(claused) > self.estimator.estimate(simple)


# ---------------------------------------------------------------------------
# RoutingDecision tests
# ---------------------------------------------------------------------------

class TestRoutingDecision:
    def test_valid_creation(self):
        rd = RoutingDecision(mode="single", model_count=1, skip_debate=True, max_debate_rounds=0)
        assert rd.mode == "single"
        assert rd.model_count == 1
        assert rd.skip_debate is True
        assert rd.max_debate_rounds == 0

    def test_frozen_dataclass(self):
        rd = RoutingDecision(mode="lite", model_count=3, skip_debate=False, max_debate_rounds=1)
        with pytest.raises(AttributeError):
            rd.mode = "full"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid routing mode"):
            RoutingDecision(mode="turbo", model_count=1, skip_debate=True, max_debate_rounds=0)

    def test_zero_model_count_raises(self):
        with pytest.raises(ValueError, match="model_count must be >= 1"):
            RoutingDecision(mode="single", model_count=0, skip_debate=True, max_debate_rounds=0)

    def test_negative_debate_rounds_raises(self):
        with pytest.raises(ValueError, match="max_debate_rounds must be >= 0"):
            RoutingDecision(mode="full", model_count=5, skip_debate=False, max_debate_rounds=-1)

    def test_all_valid_modes(self):
        for mode in ("single", "lite", "full"):
            rd = RoutingDecision(mode=mode, model_count=1, skip_debate=True, max_debate_rounds=0)
            assert rd.mode == mode


# ---------------------------------------------------------------------------
# AdaptiveQueryRouter tests
# ---------------------------------------------------------------------------

class TestAdaptiveQueryRouter:
    def setup_method(self):
        self.router = AdaptiveQueryRouter(low_threshold=0.3, high_threshold=0.6)

    # Simple queries -> single mode
    @pytest.mark.parametrize("query", [
        "What is Python?",
        "2 + 2",
        "Define entropy",
        "yes",
    ])
    def test_simple_routes_to_single(self, query):
        decision = self.router.route(query, AVAILABLE_MODELS)
        assert decision.mode == "single"
        assert decision.model_count == 1
        assert decision.skip_debate is True
        assert decision.max_debate_rounds == 0

    # Complex queries -> full mode
    @pytest.mark.parametrize("query", [
        "Compare and contrast the pros and cons of microservices versus monolithic architecture in detail",
        "Analyze the advantages and disadvantages of renewable energy, evaluate the relationship "
        "between solar and wind power step by step, and design a comprehensive plan",
    ])
    def test_complex_routes_to_full(self, query):
        decision = self.router.route(query, AVAILABLE_MODELS)
        assert decision.mode == "full"
        assert decision.model_count == len(AVAILABLE_MODELS)
        assert decision.skip_debate is False
        assert decision.max_debate_rounds == 3

    # Medium queries -> lite mode
    def test_medium_routes_to_lite(self):
        query = "Explain the relationship between Docker and Kubernetes"
        decision = self.router.route(query, AVAILABLE_MODELS)
        assert decision.mode == "lite"
        assert decision.model_count == 3
        assert decision.skip_debate is False
        assert decision.max_debate_rounds == 1

    def test_lite_respects_small_model_list(self):
        # If only 2 models available, lite uses min(3, 2) = 2
        query = "Why is Rust memory safe without a garbage collector?"
        decision = self.router.route(query, ["model_a", "model_b"])
        if decision.mode == "lite":
            assert decision.model_count == 2

    def test_single_model_available(self):
        decision = self.router.route("Compare X and Y step by step in detail", ["only_model"])
        # Full mode but only 1 model available
        assert decision.model_count == 1

    def test_empty_models_raises(self):
        with pytest.raises(ValueError, match="available_models must not be empty"):
            self.router.route("hello", [])

    def test_empty_query_routes_to_single(self):
        decision = self.router.route("", AVAILABLE_MODELS)
        assert decision.mode == "single"
        assert decision.skip_debate is True

    def test_whitespace_only_query_routes_to_single(self):
        decision = self.router.route("   ", AVAILABLE_MODELS)
        assert decision.mode == "single"

    def test_unicode_query_routes_without_error(self):
        decision = self.router.route("Qu'est-ce que la vie?", AVAILABLE_MODELS)
        assert decision.mode in ("single", "lite", "full")
        assert decision.model_count >= 1

    def test_very_long_query_routes_to_full_or_lite(self):
        long_query = "Explain " + " and ".join([f"topic_{i}" for i in range(80)]) + " in detail"
        decision = self.router.route(long_query, AVAILABLE_MODELS)
        assert decision.mode in ("lite", "full")
        assert decision.model_count > 1

    # Threshold configuration
    def test_custom_thresholds(self):
        strict_router = AdaptiveQueryRouter(low_threshold=0.1, high_threshold=0.2)
        # Nearly everything should route to full with such tight thresholds
        decision = strict_router.route("How does this work?", AVAILABLE_MODELS)
        assert decision.mode in ("lite", "full")

    def test_invalid_thresholds_raise(self):
        with pytest.raises(ValueError, match="Thresholds must satisfy"):
            AdaptiveQueryRouter(low_threshold=0.7, high_threshold=0.3)

    def test_equal_thresholds_raise(self):
        with pytest.raises(ValueError, match="Thresholds must satisfy"):
            AdaptiveQueryRouter(low_threshold=0.5, high_threshold=0.5)

    def test_decision_properties_consistency(self):
        """Single mode always skips debate with 0 rounds; lite/full never skip."""
        for query in [
            "What is X?",
            "How does garbage collection work in Java?",
            "Compare and contrast pros and cons of X and Y step by step in comprehensive detail",
        ]:
            decision = self.router.route(query, AVAILABLE_MODELS)
            if decision.mode == "single":
                assert decision.skip_debate is True
                assert decision.max_debate_rounds == 0
                assert decision.model_count == 1
            elif decision.mode == "lite":
                assert decision.skip_debate is False
                assert decision.max_debate_rounds == 1
                assert 1 <= decision.model_count <= 3
            elif decision.mode == "full":
                assert decision.skip_debate is False
                assert decision.max_debate_rounds == 3
                assert decision.model_count == len(AVAILABLE_MODELS)
