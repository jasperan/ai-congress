"""
Swarm Orchestrator - Coordinates concurrent LLM requests and aggregates responses
"""
import asyncio
from typing import List, Dict, Optional
from .voting_engine import VotingEngine
from .model_registry import ModelRegistry
from .ollama_client import OllamaClient
from ..utils.config_loader import OllamaConfig, load_config
from ..utils.logger import debug_action, info_message, swarm_status_panel, truncate_text, error_message
import logging

logger = logging.getLogger(__name__)
config = load_config()


class SwarmOrchestrator:
    """Orchestrates LLM swarm with concurrent model querying"""

    def __init__(
        self,
        model_registry: ModelRegistry,
        voting_engine: VotingEngine,
        ollama_config: OllamaConfig
    ):
        self.model_registry = model_registry
        self.voting_engine = voting_engine
        self.ollama_client = OllamaClient(
            base_url=ollama_config.base_url,
            timeout=ollama_config.timeout,
            max_retries=ollama_config.max_retries
        )
        self.max_concurrent = 10

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
            summarizer_model = "phi3:3.8b"  # TODO: load from config
            messages = [{'role': 'user', 'content': prompt}]

            response = await self.ollama_client.chat(
                model=summarizer_model,
                messages=messages,
                options={'temperature': 0.1}  # Low temperature for consistent scoring
            )

            content = response['message']['content'].strip()

            # Extract float from response
            import re
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
        prompt: str,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> Dict:
        """Query a single model asynchronously"""
        try:
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': prompt})

            logger.debug(f"Querying {model_name} with temperature {temperature}")

            response = await self.ollama_client.chat(
                model=model_name,
                messages=messages,
                options={'temperature': temperature},
                stream=stream
            )

            return {
                'model': model_name,
                'response': response['message']['content'],
                'temperature': temperature,
                'success': True
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
        temperature: float = 0.7
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
            self.query_model(model, prompt, temperature, system_prompt)
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

    async def stream_swarm_response(
        self,
        models: List[str],
        prompt: str,
        websocket = None
    ):
        """
        Stream responses from swarm in real-time
        Useful for WebSocket connections
        """
        async def stream_model(model_name: str):
            try:
                messages = [{'role': 'user', 'content': prompt}]

                async for chunk in await self.ollama_client.chat(
                    model=model_name,
                    messages=messages,
                    stream=True
                ):
                    content = chunk['message']['content']

                    if websocket:
                        await websocket.send_json({
                            'type': 'model_chunk',
                            'model': model_name,
                            'content': content
                        })

                    yield content

            except Exception as e:
                logger.error(f"Streaming error for {model_name}: {e}")

        # Stream from all models concurrently
        # Implementation depends on specific WebSocket library
        pass

    async def personality_swarm(
        self,
        personalities: List[Dict[str, str]],
        prompt: str,
        base_model: str,
        temperature: float = 0.7
    ) -> Dict:
        """
        Query the same base model multiple times with different personality system prompts

        Args:
            personalities: List of {'name': str, 'system_prompt': str}
            prompt: User prompt
            base_model: Base model to use for all personalities

        Returns:
            Similar structure to multi_model_swarm
        """
        info_message("SWARM_START", f"Personality Swarm ({len(personalities)} entities)", f"Using {base_model} at T={temperature}")

        # Show swarm status panel
        status_items = [[p['name'], "Queued"] for p in personalities]
        swarm_status_panel("Personality Swarm Status", status_items, ["Personality", "Status"], config.logging.verbosity)

        # Create concurrent tasks, each with different system prompt
        tasks = [
            self.query_model(base_model, prompt, temperature, personality['system_prompt'])
            for personality in personalities
        ]

        # Execute all queries concurrently
        responses = await asyncio.gather(*tasks)

        # Add personality names to responses for display
        for i, response in enumerate(responses):
            if i < len(personalities):
                response['personality_name'] = personalities[i]['name']

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
        personality_names = [p['name'] for p in personalities[:len(successful)]]  # Match successful order

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
