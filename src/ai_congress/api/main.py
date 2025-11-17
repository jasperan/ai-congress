"""
FastAPI Main Application
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import asyncio
import logging
import json
import os
from pydantic import BaseModel

from ..core.model_registry import ModelRegistry
from ..core.voting_engine import VotingEngine
from ..core.swarm_orchestrator import SwarmOrchestrator
from ..utils.config_loader import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Congress API",
    description="LLM Swarm with Ensemble Decision Making",
    version="0.1.0"
)

# Load config
config = load_config()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
model_registry = ModelRegistry(config.ollama)
voting_engine = VotingEngine()
swarm = SwarmOrchestrator(model_registry, voting_engine, config.ollama)


# Pydantic models
class ChatRequest(BaseModel):
    prompt: str
    models: List[str]
    mode: str = config.swarm.default_mode
    temperature: float = 0.7
    temperatures: Optional[List[float]] = None
    system_prompt: Optional[str] = None
    personalities: Optional[List[str]] = None  # For personality mode


class ModelInfo(BaseModel):
    name: str
    size: int
    weight: float


class Personality(BaseModel):
    name: str
    system_prompt: str


class PersonalityCreate(BaseModel):
    name: str
    system_prompt: str


async def load_personalities() -> List[dict]:
    """Load predefined and custom personalities"""
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


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting AI Congress API...")
    await model_registry.list_available_models()
    await model_registry.load_benchmark_weights("config/models_benchmark.json")
    logger.info("Startup complete")


@app.get("/")
async def root():
    return {"message": "AI Congress API", "status": "running"}


@app.get("/api/models", response_model=List[ModelInfo])
async def list_models():
    """List all available Ollama models"""
    models = await model_registry.list_available_models()

    return [
        ModelInfo(
            name=m['name'],
            size=m.get('size', 0),
            weight=model_registry.get_model_weight(m['name'])
        )
        for m in models
    ]


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Process chat request through swarm"""
    try:
        if request.mode == "multi_model":
            result = await swarm.multi_model_swarm(
                models=request.models,
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                temperature=request.temperature
            )
        elif request.mode == "multi_request":
            temps = request.temperatures or [0.3, 0.7, 1.0]
            result = await swarm.multi_request_swarm(
                model=request.models[0] if request.models else "mistral:7b",
                prompt=request.prompt,
                temperatures=temps,
                system_prompt=request.system_prompt
            )
        elif request.mode == "hybrid":
            temps = request.temperatures or [0.5, 0.9]
            result = await swarm.hybrid_swarm(
                models=request.models,
                prompt=request.prompt,
                temperatures=temps,
                system_prompt=request.system_prompt
            )
        elif request.mode == "personality":
            if not request.personalities:
                raise HTTPException(status_code=400, detail="Personalities required for personality mode")
            # Load personalities
            personalities_data = await load_personalities()
            selected_personalities = [
                p for p in personalities_data
                if p['name'] in request.personalities
            ]
            if not selected_personalities:
                raise HTTPException(status_code=400, detail="No valid personalities selected")
            result = await swarm.personality_swarm(
                personalities=selected_personalities,
                prompt=request.prompt,
                base_model=config.agents.base_model,
                temperature=request.temperature
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid mode")

        return result

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat"""
    await websocket.accept()
    logger.info("WebSocket connection established")

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()

            prompt = data.get('prompt')
            models = data.get('models', ['mistral:7b'])
            mode = data.get('mode', 'multi_model')

            # Send acknowledgment
            await websocket.send_json({
                'type': 'start',
                'message': f'Processing with {len(models)} models...'
            })

            # Process swarm request
            if mode == "multi_model":
                result = await swarm.multi_model_swarm(
                    models=models,
                    prompt=prompt
                )
            else:
                result = await swarm.multi_request_swarm(
                    model=models[0],
                    prompt=prompt
                )

            # Send individual model responses
            for response in result.get('responses', []):
                if response['success']:
                    await websocket.send_json({
                        'type': 'model_response',
                        'model': response['model'],
                        'content': response['response']
                    })
                    await asyncio.sleep(0.1)  # Small delay for readability

            # Send final answer
            await websocket.send_json({
                'type': 'final_answer',
                'content': result['final_answer'],
                'confidence': result.get('confidence', 0),
                'vote_breakdown': result.get('vote_breakdown', {})
            })

            await websocket.send_json({'type': 'end'})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({
            'type': 'error',
            'message': str(e)
        })


@app.post("/api/models/pull/{model_name}")
async def pull_model(model_name: str):
    """Pull a new model from Ollama"""
    success = await model_registry.pull_model(model_name)
    if success:
        return {"message": f"Model {model_name} pulled successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to pull model")


@app.get("/api/personalities", response_model=List[Personality])
async def list_personalities():
    """List all available personalities (predefined and custom)"""
    personalities = await load_personalities()
    return [Personality(**p) for p in personalities]


@app.post("/api/personalities", response_model=Personality)
async def create_personality(personality: PersonalityCreate):
    """Create a new custom personality"""
    custom_file = "personalities/custom_personalities.json"

    # Ensure directory exists
    os.makedirs("personalities", exist_ok=True)

    # Load existing custom personalities
    custom_personalities = []
    if os.path.exists(custom_file):
        try:
            with open(custom_file, 'r') as f:
                custom_personalities = json.load(f)
        except Exception as e:
            logger.error(f"Error loading custom personalities: {e}")

    # Check for duplicate name
    if any(p['name'] == personality.name for p in custom_personalities):
        raise HTTPException(status_code=400, detail="Personality name already exists")

    # Add new personality
    new_personality = {"name": personality.name, "system_prompt": personality.system_prompt}
    custom_personalities.append(new_personality)

    # Save back to file
    try:
        with open(custom_file, 'w') as f:
            json.dump(custom_personalities, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving custom personalities: {e}")
        raise HTTPException(status_code=500, detail="Failed to save personality")

    return Personality(**new_personality)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
