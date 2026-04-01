"""Intelligence pattern modules for advanced AI Congress reasoning and coordination."""

from .role_prompts import (
    ROLE_SYSTEM_PROMPTS,
    get_role_prompt,
    build_role_messages,
)
from .reasoning_router import ReasoningRouter
from .self_verification import SelfVerifier
from .agent_memory import AgentMemory
from .query_classifier import QueryClassifier
from .chain_of_agents import AgentChain
from .hypothesis_explorer import HypothesisExplorer
from .meta_cognitive import MetaCognitiveMonitor
from .adversarial_tester import AdversarialTester
from .moe_router import MixtureOfExpertsRouter

__all__ = [
    "ROLE_SYSTEM_PROMPTS",
    "get_role_prompt",
    "build_role_messages",
    "ReasoningRouter",
    "SelfVerifier",
    "AgentMemory",
    "QueryClassifier",
    "AgentChain",
    "HypothesisExplorer",
    "MetaCognitiveMonitor",
    "AdversarialTester",
    "MixtureOfExpertsRouter",
]
