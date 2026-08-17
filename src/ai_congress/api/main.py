"""
FastAPI Main Application (composition root)
-------------------------------------------
Owns the app object, middleware, startup/shutdown lifecycle, and WebSocket
endpoints. HTTP route handlers live in api/routes/*.py; shared singletons
live in api/state.py.
"""
import logging
import os
from typing import List

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import time

from ..datalake.schema import init_schema
from ..datalake.middleware import DataLakeMiddleware
from ..utils.logger import info_message, error_message

from .state import (
    config,
    event_logger,
    model_registry,
    oracle_pool,
    swarm,
    default_models,
)
from .routes import chat as chat_routes
from .routes import deliberation as deliberation_routes
from .routes import documents as documents_routes
from .routes import images as images_routes
from .routes import models as models_routes
from .routes import personalities as personalities_routes
from .routes import precedents as precedents_routes
from .routes import search as search_routes
from .routes import voice as voice_routes

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Congress API",
    description="LLM Swarm with Ensemble Decision Making + RAG + Voice + Image Generation",
    version="0.2.0",
)

# ── Middleware ──────────────────────────────────────────────────────────

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


# CORS — env-overridable (comma-separated origins)
_cors_env = os.getenv("CORS_ORIGINS")
if _cors_env:
    _cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
else:
    _cors_origins = config.api.cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for generated images
os.makedirs(config.image_gen.output_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Data lake middleware (batched event logging)
app.add_middleware(DataLakeMiddleware, event_logger=event_logger)

# ── Routers ─────────────────────────────────────────────────────────────

app.include_router(models_routes.router)
app.include_router(chat_routes.router)
app.include_router(deliberation_routes.router)
app.include_router(personalities_routes.router)
app.include_router(documents_routes.router)
app.include_router(search_routes.router)
app.include_router(images_routes.router)
app.include_router(precedents_routes.router)
app.include_router(voice_routes.router)

# ── Lifecycle ───────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("=" * 80)
    logger.info("🚀 Starting AI Congress API")
    logger.info("=" * 80)

    # List available models
    logger.info("📋 Discovering Ollama models...")
    try:
        models = await model_registry.list_available_models()
        logger.info(f"   ✓ Found {len(models)} models available")
        for model in models[:5]:  # Show first 5
            logger.info(f"     - {model['name']}")
        if len(models) > 5:
            logger.info(f"     ... and {len(models) - 5} more")
    except Exception as e:
        logger.warning(f"   ⚠ Model discovery failed (app continues): {e}")

    # Load benchmark weights
    logger.info("📊 Loading model benchmark weights...")
    try:
        await model_registry.load_benchmark_weights("config/models_benchmark.json")
        logger.info("   ✓ Weights loaded")
    except Exception as e:
        logger.warning(f"   ⚠ Benchmark weights unavailable (defaults used): {e}")

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
    """Graceful shutdown: flush events and close the Oracle pool."""
    try:
        await event_logger.stop()
    except Exception as e:
        logger.warning(f"Event logger shutdown warning: {e}")
    try:
        await oracle_pool.stop()
    except Exception as e:
        logger.warning(f"Oracle pool shutdown warning: {e}")


# ── Base endpoints ──────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "AI Congress API", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# ── WebSockets ──────────────────────────────────────────────────────────

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
            models = data.get('models') or default_models()
            mode = data.get('mode', 'multi_model')
            stream = data.get('stream', False)
            temperatures = data.get('temperatures', None)
            voting_mode = data.get('voting_mode', 'classic')
            inference_backend = data.get('inference_backend', 'ollama')

            # Apply inference backend for this WS request
            swarm.inference_backend = inference_backend

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
                async def status_callback(event_type, entity_name, content=None, full_response=None):
                    if event_type == 'init':
                        status_list = [{'name': entity_name, 'status': content}]
                        await websocket.send_json({
                            'type': 'status_init',
                            'personalities': status_list
                        })
                    elif event_type == 'start':
                        await websocket.send_json({
                            'type': 'status_update',
                            'name': entity_name,
                            'status': 'Generating...'
                        })
                    elif event_type == 'chunk':
                        await websocket.send_json({
                            'type': 'chunk',
                            'name': entity_name,
                            'content': content
                        })
                    elif event_type == 'complete':
                        await websocket.send_json({
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
            elif mode == "deliberation":
                # Flagship mode over WS: resolve a council from the triad
                # (or raw models) and run the 3-round protocol.
                from .routes.deliberation import _build_agents
                agents = await _build_agents(data.get('triad'), models)
                result = await swarm.deliberation_swarm(
                    agents=agents,
                    prompt=prompt,
                    temperature=data.get('temperature', 0.7),
                    evidence=data.get('evidence'),
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

            await websocket.send_json({
                'type': 'final_answer',
                'content': result['final_answer'],
                'confidence': result.get('confidence', 0),
                'semantic_confidence': result.get('semantic_confidence', 0),
                'vote_breakdown': result.get('vote_breakdown', {}),
                'semantic_vote': result.get('semantic_vote'),
                # 4.3.2: verdict payload for the deliberation UI
                'mode': result.get('mode'),
                'verdict': result.get('verdict'),
                'data': {
                    'rounds': result.get('rounds'),
                    'restate': result.get('restate'),
                    'dissent_report': result.get('dissent_report'),
                    'steelman': result.get('steelman'),
                    'final_positions': result.get('responses'),
                    'agents_used': result.get('agents_used'),
                    'engagement_compliance': result.get('engagement_compliance'),
                    'metadata': result.get('metadata'),
                },
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
        try:
            await websocket.send_json({
                'type': 'error',
                'message': str(e)
            })
        except Exception:
            pass
