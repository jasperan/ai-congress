"""
FastAPI Main Application
Enhanced with comprehensive logging and verbosity
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Request
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
from ..utils.config_loader import load_config
from ..utils.logger import info_message, error_message, truncate_text

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

# Initialize new components (lazy loading)
rag_engine = None
voice_transcriber = None
web_search_engine = None
web_browser = None
image_generator = None


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
    
    logger.info("")
    logger.info("✅ AI Congress API started successfully!")
    logger.info("=" * 80)


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
    
    try:
        # Log the incoming request
        if request.mode == "personality":
            personality_count = len(request.personalities) if request.personalities else 0
            info_message("CHAT_REQUEST", f"{request.mode.upper()} Mode", f"Prompt: {truncate_text(request.prompt, 50)}... with {personality_count} personalities")
        else:
            model_count = len(request.models)
            info_message("CHAT_REQUEST", f"{request.mode.upper()} Mode", f"Prompt: {truncate_text(request.prompt, 50)}... with {model_count} models")

        # Initialize prompt (may be augmented with RAG/web search)
        augmented_prompt = request.prompt
        context_sources = []
        
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
                temperature=request.temperature
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
            stream = data.get('stream', False)
            temperatures = data.get('temperatures', None)

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
                    prompt=prompt
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
async def upload_document(file: UploadFile = File(...)):
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
        
        # Process document
        result = await rag_engine.process_document(file_path)
        
        return result
        
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
                output_dir=config.image_gen.output_dir
            )
        
        result = await image_generator.generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            steps=request.steps,
            width=request.width,
            height=request.height,
            seed=request.seed
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
