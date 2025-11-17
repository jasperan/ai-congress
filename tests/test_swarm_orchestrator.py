"""
Tests for SwarmOrchestrator
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.ai_congress.core.swarm_orchestrator import SwarmOrchestrator
from src.ai_congress.core.voting_engine import VotingEngine
from src.ai_congress.core.model_registry import ModelRegistry
from src.ai_congress.utils.config_loader import OllamaConfig


class TestSwarmOrchestrator:
    def setup_method(self):
        # Mock dependencies
        self.mock_registry = MagicMock(spec=ModelRegistry)
        self.mock_voting_engine = MagicMock(spec=VotingEngine)
        self.mock_config = MagicMock(spec=OllamaConfig)
        self.mock_config.base_url = "http://localhost:11434"
        self.mock_config.timeout = 120
        self.mock_config.max_retries = 3

        # Create orchestrator
        self.orchestrator = SwarmOrchestrator(
            self.mock_registry,
            self.mock_voting_engine,
            self.mock_config
        )

    @pytest.mark.asyncio
    async def test_semantic_confidence_perfect_agreement(self):
        """Test semantic confidence with identical responses"""
        responses = ["The answer is 42", "The answer is 42", "The answer is 42"]
        model_names = ["model1", "model2", "model3"]
        prompt = "What is the meaning of life?"

        # Mock the ollama client response
        mock_response = {
            'message': {'content': '1.0'}
        }

        with patch.object(self.orchestrator.ollama_client, 'chat', new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = mock_response

            confidence = await self.orchestrator.semantic_confidence(responses, model_names, prompt)

            assert confidence == 1.0
            mock_chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_semantic_confidence_no_agreement(self):
        """Test semantic confidence with completely different responses"""
        responses = ["Yes", "No", "Maybe"]
        model_names = ["model1", "model2", "model3"]
        prompt = "Is the sky blue?"

        mock_response = {
            'message': {'content': '0.2'}
        }

        with patch.object(self.orchestrator.ollama_client, 'chat', new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = mock_response

            confidence = await self.orchestrator.semantic_confidence(responses, model_names, prompt)

            assert confidence == 0.2

    @pytest.mark.asyncio
    async def test_semantic_confidence_parsing_error(self):
        """Test semantic confidence when summarizer returns non-numeric response"""
        responses = ["Paris", "London", "Tokyo"]
        model_names = ["model1", "model2", "model3"]
        prompt = "What is the capital of France?"

        mock_response = {
            'message': {'content': 'I cannot provide a numerical score for this.'}
        }

        with patch.object(self.orchestrator.ollama_client, 'chat', new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = mock_response

            confidence = await self.orchestrator.semantic_confidence(responses, model_names, prompt)

            # Should return fallback value of 0.5
            assert confidence == 0.5

    @pytest.mark.asyncio
    async def test_semantic_confidence_exception_handling(self):
        """Test semantic confidence when summarizer throws exception"""
        responses = ["Answer A", "Answer B"]
        model_names = ["model1", "model2"]
        prompt = "Choose an answer"

        with patch.object(self.orchestrator.ollama_client, 'chat', side_effect=Exception("Network error")):
            confidence = await self.orchestrator.semantic_confidence(responses, model_names, prompt)

            # Should return fallback value of 0.5
            assert confidence == 0.5

    @pytest.mark.asyncio
    async def test_personality_swarm_success(self):
        """Test personality swarm with successful responses"""
        personalities = [
            {'name': 'Trump', 'system_prompt': 'You are Donald Trump...'},
            {'name': 'Biden', 'system_prompt': 'You are Joe Biden...'}
        ]
        prompt = "What is your opinion on AI?"
        base_model = "qwen2.5:7b"

        # Mock the query_model method
        mock_responses = [
            {
                'model': base_model,
                'response': 'AI is tremendous!',
                'temperature': 0.7,
                'success': True
            },
            {
                'model': base_model,
                'response': 'AI is concerning but promising.',
                'temperature': 0.7,
                'success': True
            }
        ]

        # Mock voting engine
        self.mock_voting_engine.weighted_majority_vote.return_value = (
            'AI is tremendous!',
            0.6,
            {'AI is tremendous!': {'weight': 0.6}}
        )

        with patch.object(self.orchestrator, 'query_model', new_callable=AsyncMock) as mock_query, \
             patch.object(self.orchestrator, 'semantic_confidence', new_callable=AsyncMock) as mock_semantic:

            mock_query.side_effect = mock_responses
            mock_semantic.return_value = 0.7

            result = await self.orchestrator.personality_swarm(personalities, prompt, base_model)

            assert result['final_answer'] == 'AI is tremendous!'
            assert result['confidence'] == 0.6
            assert result['semantic_confidence'] == 0.7
            assert result['personalities_used'] == ['Trump', 'Biden']
            assert result['base_model'] == base_model

            # Verify query_model was called twice with correct system prompts
            assert mock_query.call_count == 2
            calls = mock_query.call_args_list
            # call_args_list stores positional args in [0] and kwargs in [1]
            # query_model(base_model, prompt, temperature, system_prompt)
            assert calls[0][0][3] == 'You are Donald Trump...'  # 4th positional arg
            assert calls[1][0][3] == 'You are Joe Biden...'     # 4th positional arg

    @pytest.mark.asyncio
    async def test_personality_swarm_no_successful_responses(self):
        """Test personality swarm when all responses fail"""
        personalities = [
            {'name': 'Trump', 'system_prompt': 'You are Donald Trump...'}
        ]
        prompt = "What is your opinion?"
        base_model = "deepseek-r1"

        mock_response = {
            'model': base_model,
            'response': '',
            'temperature': 0.7,
            'success': False,
            'error': 'Connection failed'
        }

        with patch.object(self.orchestrator, 'query_model', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = mock_response

            result = await self.orchestrator.personality_swarm(personalities, prompt, base_model)

            assert 'Error' in result['final_answer']
            assert result['confidence'] == 0.0
            assert result['semantic_confidence'] == 0.0
            assert len(result['responses']) == 1
            assert not result['responses'][0]['success']


class TestPersonalityLists:
    """Test personality list loading functionality"""

    def test_load_hollywood_personalities(self):
        """Test loading Hollywood personalities from JSON"""
        import json
        import os

        file_path = "config/hollywood_personalities.json"
        assert os.path.exists(file_path), "Hollywood personalities file should exist"

        with open(file_path, 'r') as f:
            data = json.load(f)

        assert isinstance(data, list), "Should be a list"
        assert len(data) > 0, "Should have personalities"

        for personality in data:
            assert "name" in personality, "Each personality should have a name"
            assert "system_prompt" in personality, "Each personality should have a system_prompt"
            assert isinstance(personality["name"], str), "Name should be string"
            assert isinstance(personality["system_prompt"], str), "System prompt should be string"

    def test_load_youtubers_personalities(self):
        """Test loading YouTubers personalities from JSON"""
        import json
        import os

        file_path = "config/youtubers.json"
        assert os.path.exists(file_path), "YouTubers personalities file should exist"

        with open(file_path, 'r') as f:
            data = json.load(f)

        assert isinstance(data, list), "Should be a list"
        assert len(data) > 0, "Should have personalities"

        for personality in data:
            assert "name" in personality, "Each personality should have a name"
            assert "system_prompt" in personality, "Each personality should have a system_prompt"
            assert isinstance(personality["name"], str), "Name should be string"
            assert isinstance(personality["system_prompt"], str), "System prompt should be string"

    def test_load_us_congress_personalities(self):
        """Test loading US Congress personalities from JSON (should not be empty)"""
        import json
        import os

        file_path = "config/us_congress.json"
        assert os.path.exists(file_path), "US Congress personalities file should exist"

        with open(file_path, 'r') as f:
            data = json.load(f)

        assert isinstance(data, list), "Should be a list"
        # Can be empty for now
