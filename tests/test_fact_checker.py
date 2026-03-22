"""Tests for the Cross-Model Fact Verification module."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.ai_congress.core.verification.fact_checker import (
    ClaimExtractor,
    ClaimVerification,
    CrossModelFactChecker,
    FactCheckReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ollama_mock(response_content: str) -> AsyncMock:
    """Build a mock OllamaClient whose .chat() returns the given content."""
    mock = AsyncMock()
    mock.chat.return_value = {"message": {"content": response_content}}
    return mock


def make_multi_response_mock(responses: list[str]) -> AsyncMock:
    """Mock that returns different content on successive .chat() calls."""
    mock = AsyncMock()
    mock.chat.side_effect = [
        {"message": {"content": r}} for r in responses
    ]
    return mock


# ---------------------------------------------------------------------------
# ClaimExtractor tests
# ---------------------------------------------------------------------------

class TestClaimExtractor:

    @pytest.mark.asyncio
    async def test_extract_valid_json(self):
        """LLM returns a clean JSON array of claims."""
        claims_json = '["The Earth orbits the Sun", "Water boils at 100C", "Python was created in 1991"]'
        mock_client = make_ollama_mock(claims_json)
        extractor = ClaimExtractor()

        result = await extractor.extract("some response", mock_client, "test-model")

        assert len(result) == 3
        assert "The Earth orbits the Sun" in result
        assert "Water boils at 100C" in result
        mock_client.chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_extract_json_with_surrounding_text(self):
        """LLM wraps the JSON array in extra prose; regex still finds it."""
        content = 'Here are the claims:\n["Claim A", "Claim B"]\nHope that helps!'
        mock_client = make_ollama_mock(content)
        extractor = ClaimExtractor()

        result = await extractor.extract("resp", mock_client, "m")

        assert result == ["Claim A", "Claim B"]

    @pytest.mark.asyncio
    async def test_extract_malformed_response_returns_empty(self):
        """When the LLM response isn't parseable, return an empty list."""
        mock_client = make_ollama_mock("I can't do that, sorry.")
        extractor = ClaimExtractor()

        result = await extractor.extract("resp", mock_client, "m")

        assert result == []

    @pytest.mark.asyncio
    async def test_extract_exception_returns_empty(self):
        """If the LLM call itself raises, return an empty list."""
        mock_client = AsyncMock()
        mock_client.chat.side_effect = RuntimeError("connection failed")
        extractor = ClaimExtractor()

        result = await extractor.extract("resp", mock_client, "m")

        assert result == []

    @pytest.mark.asyncio
    async def test_extract_caps_at_five_claims(self):
        """Even if the LLM returns more than 5, we cap at 5."""
        claims = '["a","b","c","d","e","f","g"]'
        mock_client = make_ollama_mock(claims)
        extractor = ClaimExtractor()

        result = await extractor.extract("resp", mock_client, "m")

        assert len(result) == 5


# ---------------------------------------------------------------------------
# CrossModelFactChecker._check_claim tests
# ---------------------------------------------------------------------------

class TestCheckClaim:

    @pytest.mark.asyncio
    async def test_parse_verified(self):
        mock_client = make_ollama_mock("VERIFIED: This is correct.")
        checker = CrossModelFactChecker(mock_client)

        verdict = await checker._check_claim("some claim", "model-a")

        assert verdict["model"] == "model-a"
        assert verdict["verdict"] == "verified"
        assert "This is correct" in verdict["reason"]

    @pytest.mark.asyncio
    async def test_parse_disputed(self):
        mock_client = make_ollama_mock("DISPUTED: Actually that's wrong.")
        checker = CrossModelFactChecker(mock_client)

        verdict = await checker._check_claim("bad claim", "model-b")

        assert verdict["verdict"] == "disputed"

    @pytest.mark.asyncio
    async def test_parse_uncertain(self):
        mock_client = make_ollama_mock("UNCERTAIN: Not enough info to tell.")
        checker = CrossModelFactChecker(mock_client)

        verdict = await checker._check_claim("ambiguous claim", "model-c")

        assert verdict["verdict"] == "uncertain"

    @pytest.mark.asyncio
    async def test_parse_no_status_keyword_defaults_uncertain(self):
        mock_client = make_ollama_mock("I have no idea what you're talking about.")
        checker = CrossModelFactChecker(mock_client)

        verdict = await checker._check_claim("weird claim", "model-d")

        assert verdict["verdict"] == "uncertain"

    @pytest.mark.asyncio
    async def test_exception_returns_uncertain(self):
        mock_client = AsyncMock()
        mock_client.chat.side_effect = ConnectionError("timeout")
        checker = CrossModelFactChecker(mock_client)

        verdict = await checker._check_claim("claim", "model-e")

        assert verdict["verdict"] == "uncertain"
        assert "timeout" in verdict["reason"]


# ---------------------------------------------------------------------------
# CrossModelFactChecker.verify_response tests
# ---------------------------------------------------------------------------

class TestVerifyResponse:

    @pytest.mark.asyncio
    async def test_all_verified(self):
        """3 dissenting models all say VERIFIED -> claims marked verified."""
        # First call: extraction; next 3 calls: verification (1 claim x 3 models)
        responses = [
            '["The sky is blue"]',
            "VERIFIED: Correct per physics.",
            "VERIFIED: Yes, Rayleigh scattering.",
            "VERIFIED: Confirmed.",
        ]
        mock_client = make_multi_response_mock(responses)
        checker = CrossModelFactChecker(mock_client)

        report = await checker.verify_response(
            "The sky is blue because of scattering.",
            ["m1", "m2", "m3"],
            "judge",
        )

        assert report.verified_count == 1
        assert report.disputed_count == 0
        assert report.uncertain_count == 0
        assert report.corrected_response is None

    @pytest.mark.asyncio
    async def test_majority_disputed(self):
        """2 out of 3 models say DISPUTED -> claim is disputed."""
        responses = [
            '["Goldfish have 3-second memory"]',
            "DISPUTED: That's a myth.",
            "DISPUTED: Studies show months of memory.",
            "VERIFIED: Common knowledge.",
        ]
        mock_client = make_multi_response_mock(responses)
        checker = CrossModelFactChecker(mock_client)

        report = await checker.verify_response(
            "Goldfish have 3-second memory.",
            ["m1", "m2", "m3"],
            "judge",
        )

        assert report.disputed_count == 1
        assert report.verified_count == 0
        assert report.corrected_response is not None
        assert "Goldfish have 3-second memory" in report.corrected_response

    @pytest.mark.asyncio
    async def test_mixed_verdicts_uncertain(self):
        """1 VERIFIED, 1 DISPUTED, 1 UNCERTAIN -> no majority, marked uncertain."""
        responses = [
            '["Pluto is a planet"]',
            "VERIFIED: Sure, it was one.",
            "DISPUTED: Reclassified in 2006.",
            "UNCERTAIN: Depends on your definition.",
        ]
        mock_client = make_multi_response_mock(responses)
        checker = CrossModelFactChecker(mock_client)

        report = await checker.verify_response(
            "Pluto is a planet.",
            ["m1", "m2", "m3"],
            "judge",
        )

        assert report.uncertain_count == 1
        assert report.disputed_count == 0
        assert report.verified_count == 0

    @pytest.mark.asyncio
    async def test_empty_claims_returns_empty_report(self):
        """When extractor finds no claims, return an empty report."""
        mock_client = make_ollama_mock("No claims found, sorry.")
        checker = CrossModelFactChecker(mock_client)

        report = await checker.verify_response("vague response", ["m1"], "judge")

        assert report.claims == []
        assert report.disputed_count == 0
        assert report.verified_count == 0
        assert report.uncertain_count == 0

    @pytest.mark.asyncio
    async def test_corrected_response_appends_warning(self):
        """Disputed claims produce a corrected response with a warning block."""
        responses = [
            '["Cats are reptiles"]',
            "DISPUTED: Cats are mammals.",
            "DISPUTED: Obviously wrong.",
        ]
        mock_client = make_multi_response_mock(responses)
        checker = CrossModelFactChecker(mock_client)

        report = await checker.verify_response(
            "Cats are reptiles.", ["m1", "m2"], "judge"
        )

        assert report.corrected_response is not None
        assert report.corrected_response.startswith("Cats are reptiles.")
        assert "Fact-check warning" in report.corrected_response
        assert "- Cats are reptiles" in report.corrected_response


# ---------------------------------------------------------------------------
# FactCheckReport.to_dict tests
# ---------------------------------------------------------------------------

class TestFactCheckReport:

    def test_to_dict_structure(self):
        """to_dict returns the expected keys and shape."""
        claim = ClaimVerification(
            claim="Water is H2O",
            status="verified",
            checker_verdicts=[
                {"model": "m1", "verdict": "verified", "reason": "correct"}
            ],
            confidence=1.0,
        )
        report = FactCheckReport(
            claims=[claim],
            disputed_count=0,
            verified_count=1,
            uncertain_count=0,
        )

        d = report.to_dict()

        assert "claims" in d
        assert "summary" in d
        assert "has_corrections" in d
        assert d["summary"]["verified"] == 1
        assert d["summary"]["disputed"] == 0
        assert d["has_corrections"] is False
        assert len(d["claims"]) == 1
        assert d["claims"][0]["claim"] == "Water is H2O"
        assert d["claims"][0]["confidence"] == 1.0

    def test_to_dict_with_corrections(self):
        report = FactCheckReport(
            claims=[],
            disputed_count=1,
            verified_count=0,
            uncertain_count=0,
            corrected_response="fixed text",
        )

        assert report.to_dict()["has_corrections"] is True

    def test_empty_report_to_dict(self):
        report = FactCheckReport([], 0, 0, 0)
        d = report.to_dict()

        assert d["claims"] == []
        assert d["summary"] == {"verified": 0, "disputed": 0, "uncertain": 0}
        assert d["has_corrections"] is False
