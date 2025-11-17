"""
Swarm Orchestrator - Coordinates concurrent LLM requests and aggregates responses
"""
import asyncio
from typing import List, Dict, Optional
from .voting_engine import VotingEngine
from .model_registry import ModelRegistry
from .ollama_client import OllamaClient
from ..utils.config_loader import OllamaConfig
import logging

logger = logging.getLogger(__name__)


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

    async def query_model(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
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
                options={'temperature': temperature}
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

        logger.info(f"Multi-model swarm completed. Confidence: {confidence:.2f}")

        return {
            'responses': responses,
            'final_answer': final_answer,
            'confidence': confidence,
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
        models: List[str],
        prompt: str,
        temperatures: List[float] = [0.5, 0.9],
        system_prompt: Optional[str] = None
    ) -> Dict:
        """
        Hybrid: Multiple models, each queried with multiple temperatures

        This is the most comprehensive swarm mode
        """
        logger.info(f"Starting hybrid swarm: {len(models)} models x {len(temperatures)} temps")

        # Create all combinations
        tasks = []
        for model in models:
            for temp in temperatures:
                tasks.append(self.query_model(model, prompt, temp, system_prompt))

        # Execute all concurrently with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def limited_query(task):
            async with semaphore:
                return await task

        responses = await asyncio.gather(*[limited_query(task) for task in tasks])

        # Process results
        successful = [r for r in responses if r['success']]

        if not successful:
            return {
                'responses': responses,
                'final_answer': "Error: No successful responses",
                'confidence': 0.0
            }

        # Weight by both model performance and temperature
        weights = []
        for r in successful:
            model_weight = self.model_registry.get_model_weight(r['model'])
            temp_weight = 1.0 / (r['temperature'] + 0.1)  # Lower temp = higher weight
            combined_weight = model_weight * temp_weight
            weights.append(combined_weight)

        texts = [r['response'] for r in successful]
        model_names = [f"{r['model']}@{r['temperature']}" for r in successful]

        final_answer, confidence, vote_breakdown = self.voting_engine.weighted_majority_vote(
            texts, weights, model_names
        )

        logger.info(f"Hybrid swarm completed. Confidence: {confidence:.2f}")

        return {
            'responses': responses,
            'final_answer': final_answer,
            'confidence': confidence,
            'vote_breakdown': vote_breakdown,
            'total_queries': len(tasks),
            'successful_queries': len(successful)
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
