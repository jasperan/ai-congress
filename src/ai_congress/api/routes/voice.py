"""Voice transcription routes."""
import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..state import config, event_logger, voice_transcriber
from ...integrations.voice import get_voice_transcriber

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["voice"])


@router.post("/audio/transcribe")
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

    except RuntimeError as e:
        # faster-whisper not installed — feature unavailable, not a server bug
        logger.warning(f"Transcription unavailable: {e}")
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
