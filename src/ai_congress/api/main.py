"""
FastAPI Main Application
Enhanced with comprehensive logging and verbosity
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Request, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict
import asyncio
import logging
import json
import os
import tempfile
from pathlib import Path
from pydantic import BaseModel
import time

from ..core.model_registry import ModelRegistry
from ..core.voting_engine import VotingEngine
from ..core.swarm_orchestrator import SwarmOrchestrator
from ..core.rag_engine import get_rag_engine
from ..integrations.voice import get_voice_transcriber
from ..integrations.web_search import get_web_search_engine
from ..integrations.web_browser import get_web_browser
from ..integrations.image_gen import get_image_generator
from ..core.enhanced_orchestrator import EnhancedOrchestrator
from ..core.personality.profile import ModelPersonalityLoader
from ..utils.config_loader import load_config
from ..utils.logger import info_message, error_message, truncate_text
from ..datalake.connection import OraclePoolManager
from ..datalake.schema import init_schema
from ..datalake.logger import EventLogger
from ..datalake.middleware import DataLakeMiddleware
from ..integrations.embeddings import get_embedding_generator
from ..core.precedent.precedent_store import PrecedentStore

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Congress API",
    description="LLM Swarm with Ensemble Decision Making + RAG + Voice + Image Generation",
    version="0.2.0"
)

# Load config
config = load_config()

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with timing and details"""
    start_time = time.time()

    # Log request
    logger.info(f"→ {request.method} {request.url.path}")
    logger.debug(f"  Headers: {dict(request.headers)}")
    logger.debug(f"  Query params: {dict(request.query_params)}")

    try:
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Log response
        logger.info(f"← {request.method} {request.url.path} - Status: {response.status_code} - Duration: {duration:.3f}s")

        return response
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"✗ {request.method} {request.url.path} - Error: {str(e)} - Duration: {duration:.3f}s")
        raise

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for generated images
os.makedirs(config.image_gen.output_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize components
model_registry = ModelRegistry(config.ollama)
voting_engine = VotingEngine()
swarm = SwarmOrchestrator(model_registry, voting_engine, config.ollama)

# Enhanced orchestrator (lazy init)
enhanced_orchestrator = None

def get_enhanced_orchestrator():
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

# Initialize new components (lazy loading)
rag_engine = None
voice_transcriber = None
web_search_engine = None
web_browser = None
image_generator = None

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
app.add_middleware(DataLakeMiddleware, event_logger=event_logger)


# Pydantic models
class Personality(BaseModel):
    name: str
    system_prompt: str


class PersonalityCreate(BaseModel):
    name: str
    system_prompt: str


class PersonalityList(BaseModel):
    name: str
    count: int


class ChatRequest(BaseModel):
    prompt: str
    models: List[str]
    mode: str = config.swarm.default_mode
    temperature: float = 0.7
    temperatures: Optional[List[float]] = None
    system_prompt: Optional[str] = None
    personalities: Optional[List[Personality]] = None  # For personality mode
    history: Optional[List[Dict[str, str]]] = None  # Conversation history for context
    use_rag: bool = False  # Enable RAG
    search_web: bool = False  # Enable web search
    document_ids: Optional[List[str]] = None  # Specific documents for RAG
    voting_mode: str = "classic"  # classic | semantic


class EnhancedChatRequest(BaseModel):
    prompt: str
    models: List[str]
    temperature: float = 0.7
    enable_decomposition: bool = True
    enable_debate: bool = True
    use_rag: bool = False
    document_ids: Optional[List[str]] = None
    search_web: bool = False

class FeedbackRequest(BaseModel):
    session_id: str
    model: str
    feedback: str  # "positive" or "negative"
    response_text: Optional[str] = None


class ModelInfo(BaseModel):
    name: str
    size: int
    weight: float


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
    logger.info("=" * 80)
    logger.info("🚀 Starting AI Congress API")
    logger.info("=" * 80)

    # List available models
    logger.info("📋 Discovering Ollama models...")
    models = await model_registry.list_available_models()
    logger.info(f"   ✓ Found {len(models)} models available")
    for model in models[:5]:  # Show first 5
        logger.info(f"     - {model['name']}")
    if len(models) > 5:
        logger.info(f"     ... and {len(models) - 5} more")

    # Load benchmark weights
    logger.info("📊 Loading model benchmark weights...")
    await model_registry.load_benchmark_weights("config/models_benchmark.json")
    logger.info("   ✓ Weights loaded")

    # Log configuration summary
    logger.info("")
    logger.info("⚙️  Configuration Summary:")
    logger.info(f"   • RAG Enabled: {config.rag.enabled}")
    logger.info(f"   • Adaptive Chunking: {config.rag.adaptive_chunking}")
    logger.info(f"   • Vector Cache: {config.oracle_db.enable_cache}")
    logger.info(f"   • Web Search Engine: {config.web_search.default_engine}")
    logger.info(f"   • Advanced Extractors: {config.document_extraction.use_advanced_extractors}")
    logger.info(f"   • Max Concurrent Requests: {config.swarm.max_concurrent_requests}")

    # Initialize data lake (Oracle 26ai Free)
    logger.info("")
    logger.info("🗄️  Initializing Data Lake (Oracle 26ai Free)...")
    try:
        await oracle_pool.start()
        await init_schema(oracle_pool)
        event_logger.start()
        logger.info("   ✓ Data lake initialized")
    except Exception as e:
        logger.warning(f"   ⚠ Data lake unavailable (app continues without it): {e}")

    logger.info("")
    logger.info("✅ AI Congress API started successfully!")
    logger.info("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down AI Congress API...")
    await event_logger.stop()
    await oracle_pool.stop()
    logger.info("Shutdown complete")


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
    global rag_engine, web_search_engine

    # Data lake session
    dl_session = event_logger.new_session()
    chat_start = time.time()

    try:
        # Log the incoming request
        if request.mode == "personality":
            personality_count = len(request.personalities) if request.personalities else 0
            info_message("CHAT_REQUEST", f"{request.mode.upper()} Mode", f"Prompt: {truncate_text(request.prompt, 50)}... with {personality_count} personalities")
        else:
            model_count = len(request.models)
            info_message("CHAT_REQUEST", f"{request.mode.upper()} Mode", f"Prompt: {truncate_text(request.prompt, 50)}... with {model_count} models")

        # Log session to data lake
        await event_logger.log_session(
            dl_session, request.prompt, request.mode,
            request.voting_mode, request.models or [],
        )
        event_logger.log("chat_request", dl_session,
            mode=request.mode, voting_mode=request.voting_mode,
            model_count=len(request.models or []),
            use_rag=request.use_rag, search_web=request.search_web,
        )

        # Initialize prompt (may be augmented with RAG/web search)
        augmented_prompt = request.prompt
        context_sources = []
        web_search_results = []

        # Add web search context if requested
        if request.search_web:
            if web_search_engine is None:
                web_search_engine = get_web_search_engine(
                    max_results=config.web_search.max_results,
                    timeout=config.web_search.timeout,
                    default_engine=config.web_search.default_engine,
                    searxng_url=config.web_search.searxng_url if config.web_search.searxng_url else None,
                    yacy_url=config.web_search.yacy_url if config.web_search.yacy_url else None
                )

            logger.info("Performing web search for context...")
            search_results = await web_search_engine.search(request.prompt)

            if search_results:
                web_search_results = search_results  # Store for response
                web_context = web_search_engine.format_results_for_context(search_results)
                augmented_prompt = web_context + "\n\nUser Question: " + request.prompt
                context_sources.append({"type": "web_search", "count": len(search_results)})

        # Add RAG context if requested or if documents are specified
        if request.use_rag or request.document_ids:
            if rag_engine is None:
                rag_engine = get_rag_engine()

            logger.info("Retrieving RAG context...")

            # If specific documents, search within them
            if request.document_ids:
                all_chunks = []
                for doc_id in request.document_ids:
                    chunks = await rag_engine.retrieve_context(
                        augmented_prompt,
                        top_k=config.rag.top_k // len(request.document_ids),
                        document_id=doc_id
                    )
                    all_chunks.extend(chunks)
                rag_chunks = all_chunks
            else:
                # Search across all documents
                rag_chunks = await rag_engine.retrieve_context(augmented_prompt)

            if rag_chunks:
                # Format RAG context
                rag_context = "\n\nRelevant Context from Documents:\n\n"
                for i, chunk in enumerate(rag_chunks, 1):
                    rag_context += f"[{i}] {chunk['content']}\n"
                    rag_context += f"   (Source: {chunk['document_id']}, Similarity: {chunk['similarity']:.2f})\n\n"

                augmented_prompt = rag_context + "\n\nUser Question: " + request.prompt
                context_sources.append({"type": "rag", "count": len(rag_chunks)})

        if request.mode == "multi_model":
            result = await swarm.multi_model_swarm(
                models=request.models,
                prompt=augmented_prompt,
                system_prompt=request.system_prompt,
                temperature=request.temperature,
                voting_mode=request.voting_mode
            )
        elif request.mode == "multi_request":
            temps = request.temperatures or [0.3, 0.7, 1.0]
            result = await swarm.multi_request_swarm(
                model=request.models[0] if request.models else "mistral:7b",
                prompt=augmented_prompt,
                temperatures=temps,
                system_prompt=request.system_prompt
            )
        elif request.mode == "hybrid":
            temps = request.temperatures or [0.5, 0.9]
            result = await swarm.hybrid_swarm(
                models=request.models,
                prompt=augmented_prompt,
                temperatures=temps,
                system_prompt=request.system_prompt
            )
        elif request.mode == "personality":
            if not request.personalities:
                raise HTTPException(status_code=400, detail="Personalities required for personality mode")

            # Always use the provided personalities as full objects (convert Pydantic to dict)
            selected_personalities = [p.dict() for p in request.personalities]

            result = await swarm.personality_swarm(
                personalities=selected_personalities,
                prompt=augmented_prompt,
                base_model=config.agents.base_model,
                temperature=request.temperature,
                history=request.history
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid mode")

        # Add context sources to result
        if context_sources:
            result['context_sources'] = context_sources

        # Add web search results to response if available
        if web_search_results:
            result['web_search_results'] = web_search_results

        # Log result to data lake
        latency_ms = int((time.time() - chat_start) * 1000)
        event_logger.log("chat_response", dl_session,
            latency_ms=latency_ms,
            confidence=result.get("confidence", 0),
            models_used=result.get("models_used", []),
        )
        # Log vote data if present
        semantic_vote = result.get("semantic_vote")
        if semantic_vote:
            await event_logger.log_vote(
                dl_session, request.voting_mode,
                semantic_vote.get("winning_model", ""),
                semantic_vote.get("consensus", 0),
                len(semantic_vote.get("clusters", [])),
                semantic_vote,
            )
        elif result.get("vote_breakdown"):
            await event_logger.log_vote(
                dl_session, "classic", "",
                result.get("confidence", 0), 0,
                result.get("vote_breakdown"),
            )

        return result

    except Exception as e:
        logger.error(f"Chat error: {e}")
        event_logger.log("chat_error", dl_session, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/simulation")
async def websocket_simulation(websocket: WebSocket):
    """WebSocket endpoint for real-time congressional simulation"""
    await websocket.accept()
    logger.info("Simulation WebSocket connected")

    try:
        # Receive simulation config
        data = await websocket.receive_json()
        topic = data.get("topic", "Should AI systems be regulated by federal law?")
        num_agents = min(data.get("num_agents", 10), 10)
        num_ticks = data.get("num_ticks", 100)
        model = data.get("model", "qwen3.5:9b")

        # Import and run simulation
        from ..core.simulation import CongressSimulation
        sim = CongressSimulation(
            topic=topic,
            num_agents=num_agents,
            num_ticks=num_ticks,
            model=model,
        )

        async for event in sim.run():
            await websocket.send_json(event)

    except WebSocketDisconnect:
        logger.info("Simulation WebSocket disconnected")
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    event_logger.log("ws_connect")

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()

            prompt = data.get('prompt')
            models = data.get('models', ['mistral:7b'])
            mode = data.get('mode', 'multi_model')
            stream = data.get('stream', False)
            temperatures = data.get('temperatures', None)
            voting_mode = data.get('voting_mode', 'classic')

            # Data lake session for this WS message
            ws_session = event_logger.new_session()
            ws_start = time.time()
            await event_logger.log_session(ws_session, prompt, mode, voting_mode, models)
            event_logger.log("chat_request", ws_session,
                mode=mode, voting_mode=voting_mode,
                model_count=len(models), stream=True,
            )

            # Send acknowledgment
            if mode == "hybrid":
                temp_count = len(temperatures) if temperatures else len(config.swarm.hybrid.temperatures)
                await websocket.send_json({
                    'type': 'start',
                    'message': f'Processing hybrid swarm with {len(models)} models × {temp_count} temps...'
                })
            else:
                await websocket.send_json({
                    'type': 'start',
                    'message': f'Processing with {len(models)} models...'
                })

            # Process swarm request
            if mode == "multi_model":
                result = await swarm.multi_model_swarm(
                    models=models,
                    prompt=prompt,
                    voting_mode=voting_mode
                )
            elif mode == "multi_request":
                temps = temperatures or [0.3, 0.7, 1.0]
                result = await swarm.multi_request_swarm(
                    model=models[0],
                    prompt=prompt,
                    temperatures=temps
                )
            elif mode == "hybrid":
                temps = temperatures or config.swarm.hybrid.temperatures
                result = await swarm.hybrid_swarm(
                    models=models,
                    prompt=prompt,
                    temperatures=temps,
                    stream=stream
                )
            elif mode == "personality":
                personalities = data.get('personalities', [])
                if not personalities:
                    await websocket.send_json({
                        'type': 'error',
                        'message': 'Personalities required for personality mode'
                    })
                    continue

                # Define callback to send live status updates
                def status_callback(event_type, entity_name, content=None, full_response=None):
                    if event_type == 'init':
                        # Send initial status table
                        status_list = [{'name': entity_name, 'status': content}]
                        websocket.send_json({
                            'type': 'status_init',
                            'personalities': status_list
                        })
                    elif event_type == 'start':
                        websocket.send_json({
                            'type': 'status_update',
                            'name': entity_name,
                            'status': 'Generating...'
                        })
                    elif event_type == 'chunk':
                        websocket.send_json({
                            'type': 'chunk',
                            'name': entity_name,
                            'content': content
                        })
                    elif event_type == 'complete':
                        websocket.send_json({
                            'type': 'status_update',
                            'name': entity_name,
                            'status': 'Complete',
                            'response': full_response
                        })

                result = await swarm.personality_swarm(
                    personalities=personalities,
                    prompt=prompt,
                    base_model=config.agents.base_model,
                    temperature=data.get('temperature', 0.7),
                    history=data.get('history', []),
                    stream=stream,
                    update_callback=status_callback
                )
            else:
                await websocket.send_json({
                    'type': 'error',
                    'message': f'Unsupported mode: {mode}'
                })
                continue

            # Send individual model responses
            for response in result.get('responses', []):
                if response['success']:
                    entity_name = response.get('entity_name', response['model'])
                    await websocket.send_json({
                        'type': 'model_response',
                        'model': entity_name,
                        'content': response['response']
                    })
                    await asyncio.sleep(0.1)  # Small delay for readability

            # Send final answer
            await websocket.send_json({
                'type': 'final_answer',
                'content': result['final_answer'],
                'confidence': result.get('confidence', 0),
                'semantic_confidence': result.get('semantic_confidence', 0),
                'vote_breakdown': result.get('vote_breakdown', {}),
                'semantic_vote': result.get('semantic_vote'),
            })

            # Log to data lake
            ws_latency = int((time.time() - ws_start) * 1000)
            event_logger.log("chat_response", ws_session,
                latency_ms=ws_latency,
                confidence=result.get("confidence", 0),
                models_used=result.get("models_used", []),
            )
            semantic_vote = result.get("semantic_vote")
            if semantic_vote:
                await event_logger.log_vote(
                    ws_session, voting_mode,
                    semantic_vote.get("winning_model", ""),
                    semantic_vote.get("consensus", 0),
                    len(semantic_vote.get("clusters", [])),
                    semantic_vote,
                )

            await websocket.send_json({'type': 'end'})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        event_logger.log("ws_disconnect")
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


@app.get("/api/personality-lists", response_model=List[str])
async def list_personality_lists():
    """List available personality lists"""
    return ["hollywood", "us_congress", "youtubers"]


@app.get("/api/personality-list/{list_name}", response_model=List[Personality])
async def get_personality_list(list_name: str):
    """Get personalities from a specific list"""
    # Handle filename mapping: hollywood has _personalities suffix, others don't
    if list_name == "hollywood":
        file_path = f"config/{list_name}_personalities.json"
    else:
        file_path = f"config/{list_name}.json"

    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading personality list {list_name}: {e}")
        return []

    # Normalize to Personality format (name, system_prompt)
    personalities = []
    for item in data:
        if "name" in item and "system_prompt" in item:
            personalities.append(Personality(
                name=item["name"],
                system_prompt=item["system_prompt"]
            ))

    return personalities


@app.post("/api/chat/enhanced")
async def enhanced_chat(request: EnhancedChatRequest):
    """Process chat through the Enhanced Orchestrator with all 35 AI improvements."""
    dl_session = event_logger.new_session()
    chat_start = time.time()

    try:
        orch = get_enhanced_orchestrator()

        # Wire RAG engine if requested
        if request.use_rag or request.document_ids:
            global rag_engine
            if rag_engine is None:
                rag_engine = get_rag_engine()
            orch.rag_engine = rag_engine

        # Wire web search if requested
        if request.search_web:
            global web_search_engine
            if web_search_engine is None:
                web_search_engine = get_web_search_engine(
                    max_results=config.web_search.max_results,
                    timeout=config.web_search.timeout,
                    default_engine=config.web_search.default_engine,
                )
            orch.web_search_engine = web_search_engine

        result = await orch.enhanced_swarm(
            prompt=request.prompt,
            models=request.models,
            temperature=request.temperature,
            enable_decomposition=request.enable_decomposition,
            enable_debate=request.enable_debate,
        )

        # Log to data lake
        await event_logger.log_session(
            dl_session, request.prompt, "enhanced",
            "ensemble", request.models,
        )
        latency_ms = int((time.time() - chat_start) * 1000)
        event_logger.log("enhanced_chat_response", dl_session,
            latency_ms=latency_ms,
            confidence=result.get("confidence", 0),
            run_id=result.get("run_id", ""),
        )

        # Log precedent citation if applicable
        precedent = result.get("precedent")
        if precedent and precedent.get("cited"):
            await event_logger.log_precedent_cited(
                dl_session,
                precedent["cited"].get("id", ""),
                precedent.get("action", ""),
                precedent["cited"].get("similarity", 0),
                precedent.get("disposition", ""),
            )

        return result
    except Exception as e:
        logger.error(f"Enhanced chat error: {e}")
        event_logger.log("enhanced_chat_error", dl_session, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback on a model response."""
    try:
        orch = get_enhanced_orchestrator()
        orch.record_feedback(request.session_id, request.model, request.feedback)
        event_logger.log("user_feedback",
            session_id=request.session_id,
            model=request.model,
            feedback=request.feedback,
        )
        return {"success": True, "message": "Feedback recorded"}
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/enhanced/stats")
async def enhanced_stats():
    """Get Enhanced Orchestrator performance statistics."""
    try:
        orch = get_enhanced_orchestrator()
        return orch.get_performance_stats()
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/enhanced/runs/{run_id}")
async def get_run(run_id: str):
    """Get details of a specific Enhanced Orchestrator run."""
    try:
        orch = get_enhanced_orchestrator()
        run = orch.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get run error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# ==================== NEW ENDPOINTS ====================

# Voice Transcription
@app.post("/api/audio/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio file to text"""
    global voice_transcriber

    try:
        if voice_transcriber is None:
            voice_transcriber = get_voice_transcriber(
                model_size=config.voice.model,
                device=config.voice.device,
                compute_type=config.voice.compute_type,
                language=config.voice.language
            )

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Transcribe
        result = voice_transcriber.transcribe_file(tmp_path)

        # Clean up
        os.unlink(tmp_path)

        event_logger.log("audio_transcribe", language=result.get('language', ''))
        return {
            "success": True,
            "text": result['text'],
            "language": result['language'],
            "segments": result['segments']
        }

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Document Upload and RAG
@app.post("/api/documents/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process document for RAG"""
    global rag_engine

    try:
        if rag_engine is None:
            rag_engine = get_rag_engine()

        # Save uploaded file
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        # Generate document ID
        document_id = Path(file_path).stem

        # Process document in background
        background_tasks.add_task(rag_engine.process_document, file_path)

        event_logger.log("doc_upload", document_id=document_id, filename=file.filename)
        return {
            "success": True,
            "document_id": document_id,
            "message": "Upload successful, processing in background"
        }

    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/list")
async def list_documents():
    """List all uploaded documents"""
    global rag_engine

    try:
        if rag_engine is None:
            rag_engine = get_rag_engine()

        documents = await rag_engine.list_documents()
        return {"documents": documents}

    except Exception as e:
        logger.error(f"List documents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from vector store"""
    global rag_engine

    try:
        if rag_engine is None:
            rag_engine = get_rag_engine()

        success = await rag_engine.delete_document(document_id)

        if success:
            return {"success": True, "message": f"Document {document_id} deleted"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")

    except Exception as e:
        logger.error(f"Delete document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Web Search
class WebSearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = None
    search_type: str = "web"  # web or news


@app.post("/api/search/web")
async def search_web(request: WebSearchRequest):
    """Search the web"""
    global web_search_engine

    try:
        if web_search_engine is None:
            web_search_engine = get_web_search_engine(
                max_results=config.web_search.max_results,
                timeout=config.web_search.timeout,
                default_engine=config.web_search.default_engine,
                searxng_url=config.web_search.searxng_url if config.web_search.searxng_url else None,
                yacy_url=config.web_search.yacy_url if config.web_search.yacy_url else None
            )

        if request.search_type == "news":
            results = await web_search_engine.search_news(
                request.query,
                max_results=request.max_results
            )
        else:
            results = await web_search_engine.search(
                request.query,
                max_results=request.max_results
            )

        return {"success": True, "results": results, "query": request.query}

    except Exception as e:
        logger.error(f"Web search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Web Browsing
class BrowseRequest(BaseModel):
    url: str
    extract_clean_text: bool = True


@app.post("/api/browse")
async def browse_url(request: BrowseRequest):
    """Fetch and parse URL content"""
    global web_browser

    try:
        if web_browser is None:
            web_browser = get_web_browser(timeout=config.web_search.timeout)

        result = await web_browser.fetch_url(
            request.url,
            extract_clean_text=request.extract_clean_text
        )

        return result

    except Exception as e:
        logger.error(f"Browse error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Image Generation
class ImageGenRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    steps: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None


@app.post("/api/images/generate")
async def generate_image(request: ImageGenRequest):
    """Generate image from text prompt"""
    global image_generator

    try:
        if image_generator is None:
            image_generator = get_image_generator(
                model=config.image_gen.model,
                output_dir=config.image_gen.output_dir,
                device=config.image_gen.device
            )

        result = await image_generator.generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            steps=request.steps,
            width=request.width,
            height=request.height,
            seed=request.seed
        )

        event_logger.log("image_generate", prompt_length=len(request.prompt))
        return result

    except Exception as e:
        logger.error(f"Image generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== PRECEDENT ENDPOINTS (Stare Decisis) ====================

class PrecedentSearchRequest(BaseModel):
    query: str
    domain: Optional[str] = None
    top_k: int = 5
    min_similarity: float = 0.75


@app.get("/api/precedents")
async def list_precedents(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    domain: Optional[str] = None,
):
    """List stored precedent rulings."""
    try:
        orch = get_enhanced_orchestrator()
        if orch.precedent_store is None:
            return {"precedents": [], "message": "Precedent store not available"}
        precedents = await orch.precedent_store.list_precedents(
            limit=limit, offset=offset, domain=domain,
        )
        return {"precedents": [p.to_dict() for p in precedents]}
    except Exception as e:
        logger.error(f"List precedents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/precedents/search")
async def search_precedents(request: PrecedentSearchRequest):
    """Search for similar precedent rulings via vector similarity."""
    try:
        orch = get_enhanced_orchestrator()
        if orch.precedent_store is None:
            return {"precedents": [], "message": "Precedent store not available"}
        precedents = await orch.precedent_store.search_precedents(
            query_text=request.query,
            domain=request.domain,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
        )
        return {"precedents": [p.to_dict() for p in precedents], "query": request.query}
    except Exception as e:
        logger.error(f"Search precedents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/precedents/{precedent_id}")
async def get_precedent(precedent_id: str):
    """Get a specific precedent by ID, including supersession chain."""
    try:
        orch = get_enhanced_orchestrator()
        if orch.precedent_store is None:
            raise HTTPException(status_code=503, detail="Precedent store not available")
        precedent = await orch.precedent_store.get_precedent(precedent_id)
        if precedent is None:
            raise HTTPException(status_code=404, detail="Precedent not found")
        return precedent.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get precedent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
