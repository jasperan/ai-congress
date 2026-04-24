"""
Tests for ConsensusDetector (detects premature consensus among agent outputs).
"""
import pytest

from src.ai_congress.core.consensus_detector import (
    ConsensusReport,
    detect_consensus,
    pick_steelman_targets,
    _jaccard,
)


@pytest.mark.unit
class TestConsensusDetectorLexical:
    def test_trivial_empty(self):
        report = detect_consensus([], embedding_model="")
        assert report.agreement_ratio == 0.0
        assert report.premature is False
        assert report.pairwise == []

    def test_trivial_single(self):
        report = detect_consensus(["only one"], embedding_model="")
        assert report.agreement_ratio == 1.0
        assert report.premature is False

    def test_three_identical(self):
        texts = ["use monolith", "use monolith", "use monolith"]
        report = detect_consensus(
            texts, consensus_threshold=0.5, pairwise_threshold=0.5, embedding_model="",
        )
        assert report.method == "lexical"
        assert report.agreement_ratio == 1.0
        assert report.premature is True

    def test_full_divergence(self):
        texts = [
            "use microservices",
            "stick with the monolith",
            "go serverless functions",
        ]
        report = detect_consensus(
            texts, consensus_threshold=0.5, pairwise_threshold=0.5, embedding_model="",
        )
        assert report.agreement_ratio < 0.5
        assert report.premature is False

    def test_outlier_detection(self):
        texts = [
            "monolith monolith monolith",
            "monolith monolith monolith",
            "monolith monolith monolith",
            "microservices are better",
        ]
        report = detect_consensus(
            texts, consensus_threshold=0.4, pairwise_threshold=0.4, embedding_model="",
        )
        assert report.outlier_index == 3

    def test_jaccard_symmetry(self):
        a = "the quick brown fox"
        b = "a quick brown dog"
        assert _jaccard(a, b) == _jaccard(b, a)

    def test_mean_similarity_bounds(self):
        texts = ["x y z", "x y z", "a b c"]
        report = detect_consensus(
            texts, pairwise_threshold=0.5, embedding_model="",
        )
        assert 0.0 <= report.mean_similarity <= 1.0


@pytest.mark.unit
class TestSteelmanPicks:
    def test_empty_when_no_pairs(self):
        report = ConsensusReport(
            agreement_ratio=0.0,
            mean_similarity=0.0,
            premature=False,
            method="lexical",
            pairwise=[],
            outlier_index=None,
        )
        assert pick_steelman_targets(report, count=2) == []

    def test_picks_outlier_first(self):
        texts = [
            "monolith monolith monolith",
            "monolith monolith monolith",
            "monolith monolith monolith",
            "microservices are better and different",
        ]
        report = detect_consensus(
            texts, pairwise_threshold=0.4, embedding_model="",
        )
        picks = pick_steelman_targets(report, count=2)
        assert picks[0] == 3
        assert len(picks) == 2
        assert picks[1] != picks[0]

    def test_count_capped_by_agents(self):
        texts = ["a", "a", "a"]
        report = detect_consensus(
            texts, pairwise_threshold=0.1, embedding_model="",
        )
        picks = pick_steelman_targets(report, count=5)
        assert len(picks) == 3
