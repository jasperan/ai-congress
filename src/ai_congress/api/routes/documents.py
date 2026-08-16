"""Document upload / RAG routes."""
import logging
import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from ..state import event_logger, rag_engine
from ...core.rag_engine import get_rag_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["documents"])

# Allowed upload extensions + size cap (defense-in-depth for the public endpoint)
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".xlsx", ".pptx"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB


@router.post("/documents/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process document for RAG"""
    global rag_engine

    try:
        if rag_engine is None:
            rag_engine = get_rag_engine()

        # Validate extension + size before persisting
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{suffix or '(none)'}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
            )
        content = await file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {len(content)} bytes (max {MAX_UPLOAD_BYTES})",
            )

        # Save uploaded file with a safe name (no path traversal)
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        safe_name = Path(file.filename or "upload").name
        file_path = os.path.join(upload_dir, safe_name)
        with open(file_path, 'wb') as f:
            f.write(content)

        # Generate document ID
        document_id = Path(file_path).stem

        # Process document in background
        background_tasks.add_task(rag_engine.process_document, file_path)

        event_logger.log("doc_upload", document_id=document_id, filename=safe_name)
        return {
            "success": True,
            "document_id": document_id,
            "message": "Upload successful, processing in background"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/list")
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


@router.delete("/documents/{document_id}")
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
