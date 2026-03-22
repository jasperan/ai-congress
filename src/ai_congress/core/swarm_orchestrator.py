"""
Swarm Orchestrator - Coordinates concurrent LLM requests and aggregates responses
"""
import asyncio
import os
import re
from typing import List, Dict, Optional
from .voting_engine import VotingEngine
from .model_registry import ModelRegistry
from .semantic_voting import SemanticVotingEngine, ModelResponse
from .debate_manager import DebateManager, DebateConfig
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient
from ..utils.config_loader import OllamaConfig, OpenAIConfig, load_config
from ..utils.logger import debug_action, info_message, swarm_status_panel, truncate_text, error_message
from .acp.registry import AgentRegistry
from .acp.coordination import CoordinationController
from .acp.message_bus import ACPMessageBus
from .acp.message import AgentIdentity
from .personality.profile import ModelPersonalityLoader
from .personality.emotional_voting import EmotionalVotingEngine
import logging

logger = logging.getLogger(__name__)
config = load_config()


class SwarmOrchestrator:
    """Orchestrates LLM swarm with concurrent model querying"""

    def __init__(
        self,
        model_registry: ModelRegistry,
        voting_engine: VotingEngine,
        ollama_config: OllamaConfig,
        coordination_level: str = "moderate",
        openai_config: Optional[OpenAIConfig] = None,
        inference_backend: str = "ollama",
    ):
        self.model_registry = model_registry
        self.voting_engine = voting_engine
        self.ollama_client = OllamaClient(
            base_url=ollama_config.base_url,
            timeout=ollama_config.timeout,
            max_retries=ollama_config.max_retries
        )
        self.openai_client: Optional[OpenAIClient] = None
        if openai_config and openai_config.base_url:
            self.openai_client = OpenAIClient(
                base_url=openai_config.base_url,
                api_key=openai_config.api_key,
                model=openai_config.model,
                timeout=openai_config.timeout,
                max_retries=openai_config.max_retries,
            )
        self.inference_backend = inference_backend  # "ollama" or "openai"
        self.max_concurrent = 10

        # ACP components
        personality_config = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "config", "models_personality.json"
        )
        self.personality_loader = ModelPersonalityLoader(personality_config)
        self.registry = AgentRegistry()
        self.coordination = CoordinationController(level=coordination_level)
        self.message_bus = ACPMessageBus()
        self.emotional_voting = EmotionalVotingEngine()

    def register_model_agents(self, model_names: list[str]) -> None:
        """Register LLM models as ACP agents with personality profiles."""
        for name in model_names:
            profile = self.personality_loader.get_profile(name)
            identity = AgentIdentity(
                name=name,
                role="voter",
                personality=profile,
                capabilities=["reasoning", "voting"],
            )
            self.registry.register(identity)
            self.message_bus.register_agent(name)

    async def multi_model_swarm_with_debate(
        self,
        prompt: str,
        models: list[str],
        temperature: float = 0.7,
        history: list[dict] | None = None,
    ) -> dict:
        """Enhanced multi-model swarm with debate waves based on coordination level.

        - none/minimal: Single wave (query all -> vote)
        - moderate: Initial + 1 critique wave
        - chatty: Initial + critique + revision wave
        """
        self.register_model_agents(models)
        weights = [self.model_registry.get_model_weight(m) for m in models]
        debate_waves = 0

        # Wave 1: Initial responses
        initial_tasks = [
            self.query_model(m, prompt, temperature=temperature)
            for m in models
        ]
        initial_responses = await asyncio.gather(*initial_tasks)
        debate_waves += 1

        responses = [r.get("response", "") for r in initial_responses if r.get("success")]
        model_names_ok = [
            r.get("model", m)
            for r, m in zip(initial_responses, models)
            if r.get("success")
        ]
        weights_ok = [
            w for r, w in zip(initial_responses, weights) if r.get("success")
        ]

        # Wave 2: Critique wave (moderate or chatty)
        if self.coordination.level in ("moderate", "chatty") and len(responses) > 1:
            critique_prompt = self._build_debate_prompt(
                prompt, model_names_ok, responses,
                "Review these responses and provide your revised answer.",
            )
            critique_responses = await asyncio.gather(*[
                self.query_model(m, critique_prompt, temperature=temperature)
                for m in model_names_ok
            ])
            debate_waves += 1
            responses = self._merge_wave_responses(critique_responses, responses)

        # Wave 3: Final revision (chatty only)
        if self.coordination.level == "chatty" and len(responses) > 1:
            revision_prompt = self._build_debate_prompt(
                prompt, model_names_ok, responses,
                "Provide your final, definitive answer.",
            )
            revision_responses = await asyncio.gather(*[
                self.query_model(m, revision_prompt, temperature=max(0.3, temperature - 0.2))
                for m in model_names_ok
            ])
            debate_waves += 1
            responses = self._merge_wave_responses(revision_responses, responses)

        # Vote with emotional weighting
        personalities = [
            self.personality_loader.get_profile(m) for m in model_names_ok
        ]
        winner, confidence, vote_details = self.emotional_voting.emotional_weighted_vote(
            responses, weights_ok, personalities, model_names_ok
        )

        # Apply emotional drift (reuse personalities from voting)
        for profile, resp in zip(personalities, responses):
            agreed = resp.strip().lower() == winner.strip().lower()
            self.emotional_voting.apply_emotional_drift(
                profile, agent_agreed_with_majority=agreed
            )

        return {
            "responses": [
                {"model": m, "response": r, "success": True}
                for m, r in zip(model_names_ok, responses)
            ],
            "final_answer": winner,
            "confidence": confidence,
            "vote_breakdown": vote_details,
            "models_used": model_names_ok,
            "debate_waves": debate_waves,
            "coordination_level": self.coordination.level,
        }

    @staticmethod
    def _build_debate_prompt(
        question: str, names: list[str], responses: list[str], instruction: str,
    ) -> str:
        entries = "\n".join(f"- {n}: {r}" for n, r in zip(names, responses))
        return f"Original question: {question}\n\n{entries}\n\n{instruction}"

    @staticmethod
    def _merge_wave_responses(
        wave_responses: list[dict], previous: list[str],
    ) -> list[str]:
        return [
            r.get("response", orig) if r.get("success") else orig
            for r, orig in zip(wave_responses, previous)
        ]

    async def semantic_confidence(
        self,
        responses: List[str],
        model_names: List[str],
        original_prompt: str
    ) -> float:
        """
        Compute semantic confidence using phi3 summarizer model.
        Assesses how semantically similar the responses are.
        """
        if not responses:
            return 0.0

        # Construct prompt for summarizer
        response_list = "\n".join([
            f"{i+1}. {resp} (from {name})"
            for i, (resp, name) in enumerate(zip(responses, model_names))
        ])

        prompt = f"""Here are responses from multiple AI agents to the query: "{original_prompt}"

Responses:
{response_list}

Assess how semantically similar/agreeing they are overall. Ignore phrasing differences if meanings align (e.g., 'four' and '4' both mean the same, or 'tremendous' and 'great'). Focus on the core meaning and factual agreement.

Output only a confidence score from 0.0 (no agreement, completely different meanings) to 1.0 (full agreement, same meaning). Just the number, nothing else."""

        try:
            # Query phi3 for confidence score
            # Use the highest-weighted available model as summarizer
            top_models = self.model_registry.get_top_models(n=1)
            summarizer_model = top_models[0] if top_models else "phi3:3.8b"
            messages = [{'role': 'user', 'content': prompt}]

            response = await self.ollama_client.chat(
                model=summarizer_model,
                messages=messages,
                options={'temperature': 0.1}  # Low temperature for consistent scoring
            )

            content = response['message']['content'].strip()

            # Extract float from response
            match = re.search(r'(\d*\.?\d+)', content)
            if match:
                score = float(match.group(1))
                return min(max(score, 0.0), 1.0)  # Clamp to 0-1
            else:
                logger.warning(f"Could not parse confidence score from: {content}")
                return 0.5  # Default fallback

        except Exception as e:
            logger.error(f"Error computing semantic confidence: {e}")
            return 0.5  # Fallback to neutral

    async def query_model(
        self,
        model_name: str,
        prompt: str = "",
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
        update_callback: Optional[callable] = None,
        entity_name: Optional[str] = None,
        reasoning_mode: Optional[str] = None
    ) -> Dict:
        """Query a single model asynchronously"""
        try:
            if reasoning_mode:
                from .reasoning import CoTAgent, ReActAgent
                agent = None
                if reasoning_mode.lower() == "cot":
                    agent = CoTAgent(self.ollama_client, model_name)
                elif reasoning_mode.lower() == "react":
                    agent = ReActAgent(self.ollama_client, model_name)

                if agent:
                    logger.info(f"Using reasoning mode {reasoning_mode} for {model_name}")
                    if stream:
                        response_text = ""
                        if update_callback:
                            update_callback('start', entity_name or model_name)
                        async for chunk in agent.stream(prompt):
                            response_text += chunk
                            if update_callback:
                                update_callback('chunk', entity_name or model_name, chunk, response_text)
                        if update_callback:
                            update_callback('complete', entity_name or model_name, response_text)

                        return {
                            'model': model_name,
                            'response': response_text,
                            'temperature': temperature,
                            'success': True,
                            'reasoning_mode': reasoning_mode
                        }
                    else:
                        if update_callback:
                            update_callback('start', entity_name or model_name)
                        response_text = await agent.run(prompt)
                        if update_callback:
                            update_callback('complete', entity_name or model_name, response_text)

                        return {
                            'model': model_name,
                            'response': response_text,
                            'temperature': temperature,
                            'success': True,
                            'reasoning_mode': reasoning_mode
                        }

            if messages is None:
                messages = []
                if system_prompt:
                    messages.append({'role': 'system', 'content': system_prompt})
                messages.append({'role': 'user', 'content': prompt})

            # Pick the right client based on inference backend
            use_openai = (
                self.inference_backend == "openai"
                and self.openai_client is not None
            )
            client = self.openai_client if use_openai else self.ollama_client
            # For OpenAI backend, override model_name with the configured model
            effective_model = (
                self.openai_client.model if use_openai else model_name
            )

            logger.debug(
                f"Querying {effective_model} via {'openai' if use_openai else 'ollama'} "
                f"with temperature {temperature}"
            )

            if stream:
                response_text = ""
                if update_callback:
                    update_callback('start', entity_name or model_name)
                async for chunk in await client.chat(
                    model=effective_model,
                    messages=messages,
                    options={'temperature': temperature},
                    stream=True
                ):
                    content = chunk['message']['content']
                    response_text += content
                    if update_callback:
                        update_callback('chunk', entity_name or model_name, content, response_text)
                if update_callback:
                    update_callback('complete', entity_name or model_name, response_text)
                return {
                    'model': model_name,
                    'response': response_text,
                    'temperature': temperature,
                    'success': True,
                    'backend': 'openai' if use_openai else 'ollama',
                }
            else:
                if update_callback:
                    update_callback('start', entity_name or model_name)
                response = await client.chat(
                    model=effective_model,
                    messages=messages,
                    options={'temperature': temperature},
                    stream=False
                )
                if update_callback:
                    update_callback('complete', entity_name or model_name, response['message']['content'])
                return {
                    'model': model_name,
                    'response': response['message']['content'],
                    'temperature': temperature,
                    'success': True,
                    'backend': 'openai' if use_openai else 'ollama',
                }

        except Exception as e:
            logger.error(f"Error querying {model_name}: {e}")
            return {
                'model': model_name,
                'response': '',
                'temperature': temperature,
                'success': False,
                'error': str(e)
            }

    async def multi_model_swarm(
        self,
        models: List[str],
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        reasoning_mode: Optional[str] = None,
        voting_mode: str = "classic",
    ) -> Dict:
        """
        Query multiple different models concurrently

        Returns:
            {
                'responses': List[Dict],
                'final_answer': str,
                'confidence': float,
                'vote_breakdown': Dict
            }
        """
        logger.info(f"Starting multi-model swarm with {len(models)} models")

        # Create concurrent tasks
        tasks = [
            self.query_model(model, prompt, temperature, system_prompt, reasoning_mode=reasoning_mode)
            for model in models
        ]

        # Execute all queries concurrently
        responses = await asyncio.gather(*tasks)

        # Filter successful responses
        successful = [r for r in responses if r['success']]

        if not successful:
            logger.error("No successful responses from swarm")
            return {
                'responses': responses,
                'final_answer': "Error: No models responded successfully",
                'confidence': 0.0,
                'vote_breakdown': {}
            }

        # Get model weights
        weights = [
            self.model_registry.get_model_weight(r['model'])
            for r in successful
        ]

        # Vote on best response
        texts = [r['response'] for r in successful]
        model_names = [r['model'] for r in successful]

        if voting_mode == "semantic":
            return await self._semantic_vote(
                prompt, successful, weights, model_names, texts, responses, temperature
            )

        final_answer, confidence, vote_breakdown = self.voting_engine.weighted_majority_vote(
            texts, weights, model_names
        )

        # Compute semantic confidence
        semantic_confidence = await self.semantic_confidence(texts, model_names, prompt)

        logger.info(f"Multi-model swarm completed. String Confidence: {confidence:.2f}, Semantic Confidence: {semantic_confidence:.2f}")

        return {
            'responses': responses,
            'final_answer': final_answer,
            'confidence': confidence,
            'semantic_confidence': semantic_confidence,
            'vote_breakdown': vote_breakdown,
            'models_used': model_names,
            'weights': weights
        }

    async def _semantic_vote(self, prompt, successful, weights, model_names, texts, all_responses, temperature):
        """Run semantic voting with optional debate escalation."""
        sve = SemanticVotingEngine(
            ollama_client=self.ollama_client,
            consensus_threshold=config.voting.consensus_threshold,
        )
        model_responses = [
            ModelResponse(name, text, weight, temperature)
            for name, text, weight in zip(model_names, texts, weights)
        ]

        result = await sve.vote(model_responses)

        if result is not None:
            return {
                "responses": all_responses,
                "final_answer": result.winner,
                "confidence": result.consensus,
                "semantic_confidence": result.consensus,
                "semantic_vote": result.to_dict(),
                "models_used": model_names,
                "weights": weights,
            }

        # No consensus — run debate
        debate_cfg_raw = config.voting.debate
        debate_cfg = DebateConfig(
            max_rounds=debate_cfg_raw.max_rounds,
            consensus_threshold=config.voting.consensus_threshold,
            temp_schedule=debate_cfg_raw.temp_schedule,
            conviction_bonus=debate_cfg_raw.conviction_bonus,
        )
        clusters, analysis = await sve.judge_group(model_responses)
        dm = DebateManager(
            ollama_client=self.ollama_client,
            voting_engine=sve,
            config=debate_cfg,
        )
        debate_result = await dm.run_debate(prompt, model_responses, clusters, analysis)
        return {
            "responses": all_responses,
            "final_answer": debate_result.winner,
            "confidence": debate_result.consensus,
            "semantic_confidence": debate_result.consensus,
            "semantic_vote": debate_result.to_dict(),
            "models_used": model_names,
            "weights": weights,
        }

    async def multi_request_swarm(
        self,
        model: str,
        prompt: str,
        temperatures: List[float] = [0.3, 0.7, 1.0, 1.2],
        system_prompt: Optional[str] = None
    ) -> Dict:
        """
        Query same model multiple times with different temperatures

        Returns similar structure to multi_model_swarm
        """
        logger.info(f"Starting multi-request swarm with model {model}, {len(temperatures)} temperatures")

        # Create concurrent tasks with different temperatures
        tasks = [
            self.query_model(model, prompt, temp, system_prompt)
            for temp in temperatures
        ]

        # Execute all queries concurrently
        responses = await asyncio.gather(*tasks)

        # Filter successful
        successful = [r for r in responses if r['success']]

        if not successful:
            return {
                'responses': responses,
                'final_answer': "Error: No successful responses",
                'confidence': 0.0
            }

        # Use temperature-based weighting
        texts = [r['response'] for r in successful]
        temps = [r['temperature'] for r in successful]

        final_answer = self.voting_engine.temperature_ensemble(texts, temps)

        return {
            'responses': responses,
            'final_answer': final_answer,
            'temperatures_used': temps,
            'model': model
        }

    async def hybrid_swarm(
        self,
        models: Optional[List[str]] = None,
        prompt: str = "",
        temperatures: Optional[List[float]] = None,
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> Dict:
        """
        Hybrid: Multiple models, each queried with multiple temperatures

        This is the most comprehensive swarm mode
        """
        # Load config for defaults
        if models is None:
            models = self.model_registry.get_top_models(config.swarm.hybrid['top_models'])
        if temperatures is None:
            temperatures = config.swarm.hybrid['temperatures']

        info_message("SWARM_START", f"Hybrid Swarm ({len(models)} models × {len(temperatures)} temps)", f"Using top {len(models)} models", config.logging.verbosity)

        # Create entity names for display (model@temp combinations)
        entities = []
        for model in models:
            for temp in temperatures:
                entities.append(f"{model}@{temp}")

        # Show swarm status panel
        status_items = [[entity, "Queued"] for entity in entities]
        swarm_status_panel("Hybrid Swarm Status", status_items, ["Model/Temp", "Status"], config.logging.verbosity)

        # Create all combinations
        task_list = []
        for model in models:
            for temp in temperatures:
                task_list.append((model, temp))

        # Execute all concurrently with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def limited_query(model, temp):
            async with semaphore:
                return await self.query_model(model, prompt, temp, system_prompt, stream=stream)

        # Execute all concurrently
        tasks = [limited_query(model, temp) for model, temp in task_list]
        responses = await asyncio.gather(*tasks)

        # Add entity names to responses for display
        for i, response in enumerate(responses):
            if i < len(entities):
                response['entity_name'] = entities[i]

        # Filter successful responses
        successful = [r for r in responses if r['success']]

        if not successful:
            error_message("SWARM_ERROR", "Hybrid Swarm", "No successful responses from any model-temperature combination", config.logging.verbosity)
            return {
                'responses': responses,
                'final_answer': "Error: No models responded successfully",
                'confidence': 0.0,
                'semantic_confidence': 0.0,
                'vote_breakdown': {}
            }

        # Log successful generations
        for resp in successful:
            entity_name = resp.get('entity_name', f'Unknown@{resp.get("temperature", 0.7)}')
            debug_action("GENERATED", entity_name, f"Response: {truncate_text(resp['response'], 80)}", config.logging.verbosity)

        # Weight by both model performance and temperature
        weights = []
        for r in successful:
            model_weight = self.model_registry.get_model_weight(r['model'])
            temp_weight = 1.0 / (r['temperature'] + 0.1)  # Lower temp = higher weight
            combined_weight = model_weight * temp_weight
            weights.append(combined_weight)

        # Vote on best response
        texts = [r['response'] for r in successful]
        entity_names = [r.get('entity_name', f"{r['model']}@{r['temperature']}") for r in successful]

        debug_action("VOTING", f"Hybrid Swarm ({len(successful)} responses)", "Aggregating responses with weighted majority vote", config.logging.verbosity)

        final_answer, confidence, vote_breakdown = self.voting_engine.weighted_majority_vote(
            texts, weights, entity_names
        )

        # Compute semantic confidence
        semantic_confidence = await self.semantic_confidence(texts, entity_names, prompt)

        info_message("SWARM_COMPLETE", "Hybrid Swarm", f"Confidence: {confidence:.2f}, Semantic: {semantic_confidence:.2f}", config.logging.verbosity)

        # Update status panel with results
        result_items = [[name, f"✓ ({vote_breakdown.get(name, 0):.1f} votes)"] for name in entity_names]
        swarm_status_panel("Hybrid Swarm Results", result_items, ["Model/Temp", "Result"], config.logging.verbosity)

        return {
            'responses': responses,
            'final_answer': final_answer,
            'confidence': confidence,
            'semantic_confidence': semantic_confidence,
            'vote_breakdown': vote_breakdown,
            'models_used': models,
            'temperatures_used': temperatures,
            'total_queries': len(tasks),
            'successful_queries': len(successful),
            'weights': weights
        }

    async def personality_swarm(
        self,
        personalities: List[Dict[str, str]],
        prompt: str,
        base_model: str,
        temperature: float = 0.7,
        history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
        update_callback: Optional[callable] = None
    ) -> Dict:
        """
        Query the same base model multiple times with different personality system prompts

        Args:
            personalities: List of {'name': str, 'system_prompt': str}
            prompt: User prompt
            base_model: Base model to use for all personalities
            history: Optional conversation history as list of {'role': 'user'|'assistant', 'content': str}

        Returns:
            Similar structure to multi_model_swarm
        """
        info_message("SWARM_START", f"Personality Swarm ({len(personalities)} entities)", f"Using {base_model} at T={temperature}")

        # Show swarm status panel
        status_items = [[p['name'], "Queued"] for p in personalities]
        swarm_status_panel("Personality Swarm Status", status_items, ["Personality", "Status"], config.logging.verbosity)

        # Send initial status if callback provided
        if update_callback:
            for p in personalities:
                update_callback('init', p['name'], 'Queued')

        # Create concurrent tasks, each with different system prompt and full message history
        async def query_personality(personality: Dict[str, str]):
            # Build full messages: system + history + current user prompt
            messages = [{"role": "system", "content": personality['system_prompt']}]
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": prompt})

            pname = personality['name']
            entity_callback = (lambda type, name, content, full, _pname=pname: update_callback(type, _pname, content, full)) if update_callback else None

            return await self.query_model(
                base_model,
                temperature=temperature,
                messages=messages,
                stream=stream,
                update_callback=entity_callback,
                entity_name=personality['name']
            )

        # Create concurrent tasks
        tasks = [query_personality(personality) for personality in personalities]

        # Execute all queries concurrently with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def limited_query(task):
            async with semaphore:
                return await task

        # Execute all queries concurrently and collect results
        results = await asyncio.gather(
            *(limited_query(task) for task in tasks),
            return_exceptions=True
        )

        responses = []
        for personality, result in zip(personalities, results):
            if isinstance(result, Exception):
                responses.append({
                    'success': False,
                    'response': str(result),
                    'personality_name': personality['name']
                })
            else:
                result['personality_name'] = personality['name']
                responses.append(result)

        # Filter successful responses
        successful = [r for r in responses if r['success']]

        if not successful:
            error_message("SWARM_ERROR", "Personality Swarm", "No successful responses from any personality")
            return {
                'responses': responses,
                'final_answer': "Error: No personalities responded successfully",
                'confidence': 0.0,
                'semantic_confidence': 0.0,
                'vote_breakdown': {}
            }

        # Log successful generations
        for i, resp in enumerate(successful):
            personality_name = resp.get('model', f'Personality {i+1}')
            debug_action("GENERATED", personality_name, f"Response: {truncate_text(resp['response'], 80)}", config.logging.verbosity)

        # Use equal weights for all personalities
        weights = [1.0] * len(successful)

        # Vote on best response
        texts = [r['response'] for r in successful]
        personality_names = [r.get('personality_name', f'Personality {i+1}') for i, r in enumerate(successful)]

        debug_action("VOTING", f"Personality Swarm ({len(successful)} responses)", "Aggregating responses with equal weights", config.logging.verbosity)

        final_answer, confidence, vote_breakdown = self.voting_engine.weighted_majority_vote(
            texts, weights, personality_names
        )

        # Compute semantic confidence
        semantic_confidence = await self.semantic_confidence(texts, personality_names, prompt)

        info_message("SWARM_COMPLETE", "Personality Swarm", f"Confidence: {confidence:.2f}, Semantic: {semantic_confidence:.2f}")

        # Update status panel with results
        result_items = [[name, f"✓ ({vote_breakdown.get(name, 0):.1f} votes)"] for name in personality_names]
        swarm_status_panel("Personality Swarm Results", result_items, ["Personality", "Result"], config.logging.verbosity)

        return {
            'responses': responses,
            'final_answer': final_answer,
            'confidence': confidence,
            'semantic_confidence': semantic_confidence,
            'vote_breakdown': vote_breakdown,
            'personalities_used': personality_names,
            'base_model': base_model,
            'weights': weights
        }
