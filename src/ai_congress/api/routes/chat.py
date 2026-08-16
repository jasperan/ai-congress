"""Chat / enhanced chat / feedback / runs routes."""
import logging
import time

from fastapi import APIRouter, HTTPException
from typing import Optional

from ..schemas import ChatRequest, EnhancedChatRequest, FeedbackRequest
from ..state import (
    config, event_logger, get_enhanced_orchestrator, rag_engine,
    web_search_engine, swarm,
)
from ...integrations.web_search import get_web_search_engine
from ...core.rag_engine import get_rag_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat")
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
            from ...utils.logger import info_message
            info_message("CHAT_REQUEST", f"{request.mode.upper()} Mode", f"Prompt: {request.prompt[:50]}... with {personality_count} personalities")
        else:
            model_count = len(request.models)
            from ...utils.logger import info_message
            info_message("CHAT_REQUEST", f"{request.mode.upper()} Mode", f"Prompt: {request.prompt[:50]}... with {model_count} models")

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

        # Apply inference backend for this request
        swarm.inference_backend = request.inference_backend

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
                model=request.models[0] if request.models else "qwen3.5:9b",
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
            selected_personalities = [p.model_dump() for p in request.personalities]

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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        event_logger.log("chat_error", dl_session, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/enhanced")
async def enhanced_chat(request: EnhancedChatRequest):
    """Process chat through the Enhanced Orchestrator with all 35 AI improvements."""
    dl_session = event_logger.new_session()
    chat_start = time.time()

    try:
        orch = get_enhanced_orchestrator()
        orch.inference_backend = request.inference_backend

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


@router.post("/feedback")
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


@router.get("/enhanced/stats")
async def enhanced_stats():
    """Get Enhanced Orchestrator performance statistics."""
    try:
        orch = get_enhanced_orchestrator()
        return orch.get_performance_stats()
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/enhanced/runs/{run_id}")
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
