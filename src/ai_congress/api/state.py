"""
API state (composition root).

All long-lived singletons live here so route modules and the main app can
import them without circular dependencies. Route handlers only read/write
these objects; startup/shutdown wiring stays in api.main.
"""
from __future__ import annotations

import os
import logging
from typing import List, Optional

from ..core.model_registry import ModelRegistry
from ..core.voting_engine import VotingEngine
from ..core.swarm_orchestrator import SwarmOrchestrator
from ..core.enhanced_orchestrator import EnhancedOrchestrator
from ..core.personality.profile import ModelPersonalityLoader
from ..datalake.connection import OraclePoolManager
from ..datalake.logger import EventLogger
from ..integrations.embeddings import get_embedding_generator
from ..core.precedent.precedent_store import PrecedentStore
from ..utils.config_loader import load_config

logger = logging.getLogger(__name__)

config = load_config()

# Core components
model_registry = ModelRegistry(config.ollama)
voting_engine = VotingEngine()
swarm = SwarmOrchestrator(
    model_registry, voting_engine, config.ollama,
    openai_config=config.openai,
    pi_config=config.pi,
)

# Data lake (Oracle 26ai Free) — config-driven
oracle_pool = OraclePoolManager(
    host=config.datalake.host,
    port=config.datalake.port,
    service=config.datalake.service,
    user=config.datalake.user,
    password=config.datalake.password,
    pool_min=config.datalake.pool_min,
    pool_max=config.datalake.pool_max,
)
event_logger = EventLogger(oracle_pool)

# Lazy singletons for optional integrations
rag_engine = None
voice_transcriber = None
web_search_engine = None
web_browser = None
image_generator = None

# Enhanced orchestrator (lazy init)
enhanced_orchestrator: Optional[EnhancedOrchestrator] = None


def get_enhanced_orchestrator() -> EnhancedOrchestrator:
    global enhanced_orchestrator
    if enhanced_orchestrator is None:
        personality_config = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "config", "models_personality.json"
        )
        personality_loader = ModelPersonalityLoader(personality_config)
        enhanced_orchestrator = EnhancedOrchestrator(
            model_registry=model_registry,
            voting_engine=voting_engine,
            ollama_client=swarm.ollama_client,
            personality_loader=personality_loader,
            pi_client=swarm.pi_client,
            inference_backend=swarm.inference_backend,
        )

        # Wire precedent store if Oracle is available
        if oracle_pool.is_available:
            try:
                embedder = get_embedding_generator()
                enhanced_orchestrator.precedent_store = PrecedentStore(oracle_pool, embedder)
                logger.info("Precedent store initialized (stare decisis enabled)")
            except Exception as e:
                logger.warning("Precedent store init failed (stare decisis disabled): %s", e)
    return enhanced_orchestrator


# ELO tracker (lazy init)
_elo_tracker = None


def get_elo_tracker():
    global _elo_tracker
    if _elo_tracker is None:
        from ..core.learning.elo_tracker import ELOTracker
        _elo_tracker = ELOTracker()
    return _elo_tracker


def default_models() -> List[str]:
    """Preferred models from config, falling back to a sane default."""
    pref = config.models.preferred
    if pref:
        return list(pref)
    return ["qwen3.5:9b"]


def pi_models() -> List[dict]:
    """Models exposed by the pi backend (deepseek-v4-flash via opencode-go).

    Returns [{name, size, weight, backend: "pi"}] when the pi backend is
    enabled (API key present). Sizes are unknown for remote models (0).
    """
    if not config.pi.enabled:
        return []
    return [
        {
            "name": m,
            "size": 0,
            "weight": model_registry.get_model_weight(m),
            "backend": "pi",
        }
        for m in config.pi.preferred_models
    ]


async def load_personalities() -> List[dict]:
    """Load predefined and custom personalities"""
    import json

    personalities = []

    # Load predefined
    predefined_file = "config/personalities.json"
    if os.path.exists(predefined_file):
        try:
            with open(predefined_file, 'r') as f:
                personalities.extend(json.load(f))
        except Exception as e:
            logger.error(f"Error loading predefined personalities: {e}")

    # Load custom
    custom_file = "personalities/custom_personalities.json"
    if os.path.exists(custom_file):
        try:
            with open(custom_file, 'r') as f:
                personalities.extend(json.load(f))
        except Exception as e:
            logger.error(f"Error loading custom personalities: {e}")

    return personalities
