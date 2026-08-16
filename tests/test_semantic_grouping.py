"""Tests for semantic grouping in voting (Phase 1: semantic core)."""
import pytest

from src.ai_congress.core.voting_engine import VotingEngine
import src.ai_congress.utils.semantic as _semantic_mod


utils = type("utils", (), {"semantic": _semantic_mod})()


def _patch_similarity(monkeypatch, func):
    """Route utils.semantic.text_similarity through a fake (embedding mode)."""
    monkeypatch.setattr(utils.semantic, "embedding_available", lambda: True)
    monkeypatch.setattr(utils.semantic, "text_similarity", func)


class TestSemanticGrouping:
    def test_paraphrases_pool_weight(self, monkeypatch):
        """Embedding-mode: two models saying the same thing pool their weight."""
        def fake_sim(a, b):
            pairs = {
                ("The answer is 42 because of the calculations shown",
                 "42 is correct based on the math above"),
                ("42 is correct based on the math above",
                 "The answer is 42 because of the calculations shown"),
            }
            return 0.9 if (a, b) in pairs else 0.1
        _patch_similarity(monkeypatch, fake_sim)

        ve = VotingEngine(semantic_grouping=True)
        responses = [
            "The answer is 42 because of the calculations shown",
            "42 is correct based on the math above",
            "I think the result is 7 and here is my different reasoning",
        ]
        winner, confidence, details = ve.weighted_majority_vote(responses, [1.0] * 3)
        assert confidence > 0.6, f"paraphrases should pool: {confidence:.2f}"
        assert winner == responses[0], "first paraphrase is representative"

    def test_winner_is_highest_weight_member(self, monkeypatch):
        """The representative (winner text) is the highest-weight group member."""
        def fake_sim(a, b):
            return 0.9 if ("42" in a and "42" in b) else 0.1
        _patch_similarity(monkeypatch, fake_sim)

        ve = VotingEngine(semantic_grouping=True)
        responses = [
            "The answer is 42 because of the calculations shown",
            "42 is correct based on the math above",
            "I think the result is 7",
        ]
        # The second paraphrase carries more weight -> its text becomes winner
        winner, confidence, _ = ve.weighted_majority_vote(responses, [0.5, 2.0, 1.0])
        assert winner == responses[1]

    def test_lexical_fallback_does_not_merge_disjoint(self):
        """Real lexical path: obviously different answers stay split."""
        ve = VotingEngine(semantic_grouping=True)
        responses = [
            "The capital of France is Paris",
            "The chemical symbol for gold is Au",
            "Python is a programming language",
        ]
        winner, confidence, details = ve.weighted_majority_vote(responses, [1.0] * 3)
        assert confidence <= 0.34

    def test_exact_match_still_pools(self):
        """Identical responses always pool (exact normalization)."""
        ve = VotingEngine(semantic_grouping=True)
        responses = ["Yes, deploy it.", "Yes, deploy it."]
        winner, confidence, details = ve.weighted_majority_vote(responses, [1.0, 1.0])
        assert confidence == 1.0

    def test_disabled_grouping_is_exact(self):
        """semantic_grouping=False keeps old exact-match behavior."""
        ve = VotingEngine(semantic_grouping=False)
        responses = ["The answer is 42 because of the math", "42 is correct per the math"]
        winner, confidence, details = ve.weighted_majority_vote(responses, [1.0, 1.0])
        assert confidence <= 0.51, "no pooling when semantic grouping disabled"

    def test_rank_responses_semantic(self, monkeypatch):
        """rank_responses works with semantic grouping (winner first)."""
        def fake_sim(a, b):
            return 0.9 if "acquisition" in a and "acquisition" in b else 0.1
        _patch_similarity(monkeypatch, fake_sim)

        ve = VotingEngine(semantic_grouping=True)
        responses = [
            "We should adopt the acquisition offer because of synergies",
            "The acquisition synergies make this offer worth accepting",
            "We should decline because of regulatory risk",
        ]
        ranked = ve.rank_responses(responses, [1.0, 1.0, 1.0])
        assert ranked[0]['rank'] == 1
        assert ranked[0]['weight'] >= 2.0, "paraphrase group should lead"
