"""Sequential agent chains for multi-step reasoning pipelines."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class AgentChain:
    """Executes a sequential chain of agent steps, passing output forward.

    Each step has a role, model, and prompt template. The template can
    reference {query} (original query) and {previous_output} (output of
    the preceding step).
    """

    def __init__(self, ollama_client: Any, steps: list[dict]) -> None:
        """Initialize the agent chain.

        Args:
            ollama_client: Async Ollama client with a `generate()` method.
            steps: List of step dicts, each with keys:
                - role: str (e.g. "researcher", "analyst")
                - model: str (Ollama model name)
                - prompt_template: str with {query} and {previous_output} placeholders
        """
        self.ollama_client = ollama_client
        self.steps = steps

    async def execute(self, query: str) -> dict:
        """Run steps sequentially, each step seeing previous output.

        Args:
            query: The original user query.

        Returns:
            Dict with keys:
                - final_answer: str
                - steps: list of {role, model, output, duration_ms}
                - total_duration_ms: int
        """
        total_start = time.monotonic()
        step_results = []
        previous_output = ""

        for step in self.steps:
            step_start = time.monotonic()
            role = step.get("role", "agent")
            model = step.get("model", "")
            template = step.get("prompt_template", "{query}\n\n{previous_output}")

            prompt = template.format(
                query=query,
                previous_output=previous_output,
            )

            try:
                result = await self.ollama_client.generate(
                    model=model,
                    prompt=prompt,
                )
                output = result.get("response", "") if isinstance(result, dict) else str(result)
            except Exception as e:
                logger.error("Chain step '%s' (%s) failed: %s", role, model, e)
                output = f"[Error in {role} step: {e}]"

            duration_ms = int((time.monotonic() - step_start) * 1000)
            step_results.append({
                "role": role,
                "model": model,
                "output": output,
                "duration_ms": duration_ms,
            })
            previous_output = output

        total_duration_ms = int((time.monotonic() - total_start) * 1000)

        return {
            "final_answer": previous_output,
            "steps": step_results,
            "total_duration_ms": total_duration_ms,
        }

    @classmethod
    def research_chain(cls, ollama_client: Any, model: str) -> "AgentChain":
        """Create a research chain: Researcher -> Analyst -> Synthesizer -> Critic.

        Args:
            ollama_client: Async Ollama client.
            model: Ollama model name to use for all steps.
        """
        steps = [
            {
                "role": "researcher",
                "model": model,
                "prompt_template": (
                    "You are a researcher. Gather all relevant information about "
                    "the following question. Be thorough and factual.\n\n"
                    "Question: {query}"
                ),
            },
            {
                "role": "analyst",
                "model": model,
                "prompt_template": (
                    "You are an analyst. Analyze the following research findings "
                    "in the context of the original question. Identify key insights, "
                    "patterns, and conclusions.\n\n"
                    "Original question: {query}\n\n"
                    "Research findings:\n{previous_output}"
                ),
            },
            {
                "role": "synthesizer",
                "model": model,
                "prompt_template": (
                    "You are a synthesizer. Create a comprehensive, well-structured "
                    "answer to the original question based on the analysis below.\n\n"
                    "Original question: {query}\n\n"
                    "Analysis:\n{previous_output}"
                ),
            },
            {
                "role": "critic",
                "model": model,
                "prompt_template": (
                    "You are a critic. Review the following answer for accuracy, "
                    "completeness, and clarity. If improvements are needed, provide "
                    "a corrected version. If it is good, confirm it.\n\n"
                    "Original question: {query}\n\n"
                    "Answer to review:\n{previous_output}"
                ),
            },
        ]
        return cls(ollama_client, steps)

    @classmethod
    def reasoning_chain(cls, ollama_client: Any, model: str) -> "AgentChain":
        """Create a reasoning chain: Decomposer -> Solver -> Verifier.

        Args:
            ollama_client: Async Ollama client.
            model: Ollama model name to use for all steps.
        """
        steps = [
            {
                "role": "decomposer",
                "model": model,
                "prompt_template": (
                    "You are a problem decomposer. Break the following question "
                    "into smaller, manageable sub-problems. List each sub-problem "
                    "clearly.\n\n"
                    "Question: {query}"
                ),
            },
            {
                "role": "solver",
                "model": model,
                "prompt_template": (
                    "You are a problem solver. Solve each of the following "
                    "sub-problems step by step, then combine them into a "
                    "final answer.\n\n"
                    "Original question: {query}\n\n"
                    "Sub-problems:\n{previous_output}"
                ),
            },
            {
                "role": "verifier",
                "model": model,
                "prompt_template": (
                    "You are a solution verifier. Check the following solution "
                    "for correctness. Verify each step. If there are errors, "
                    "provide the corrected solution.\n\n"
                    "Original question: {query}\n\n"
                    "Solution:\n{previous_output}"
                ),
            },
        ]
        return cls(ollama_client, steps)

    @classmethod
    def creative_chain(cls, ollama_client: Any, model: str) -> "AgentChain":
        """Create a creative chain: Brainstormer -> Refiner -> Editor.

        Args:
            ollama_client: Async Ollama client.
            model: Ollama model name to use for all steps.
        """
        steps = [
            {
                "role": "brainstormer",
                "model": model,
                "prompt_template": (
                    "You are a creative brainstormer. Generate multiple creative "
                    "ideas and approaches for the following request. Be imaginative "
                    "and diverse in your suggestions.\n\n"
                    "Request: {query}"
                ),
            },
            {
                "role": "refiner",
                "model": model,
                "prompt_template": (
                    "You are a creative refiner. Take the best ideas from below "
                    "and develop them into a polished, coherent piece. Combine "
                    "the strongest elements.\n\n"
                    "Original request: {query}\n\n"
                    "Brainstormed ideas:\n{previous_output}"
                ),
            },
            {
                "role": "editor",
                "model": model,
                "prompt_template": (
                    "You are an editor. Polish the following creative work for "
                    "clarity, flow, and impact. Fix any issues while preserving "
                    "the creative voice.\n\n"
                    "Original request: {query}\n\n"
                    "Draft:\n{previous_output}"
                ),
            },
        ]
        return cls(ollama_client, steps)
