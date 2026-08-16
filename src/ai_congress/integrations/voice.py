"""
Voice Transcription Module
Provides voice-to-text using Faster-Whisper
"""
import logging
from typing import Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

# faster-whisper is a heavy optional dependency. Mirror the graceful pattern
# from consensus_detector.py: import lazily, cache the failure, and let the
# caller surface a clear error instead of breaking app/test import.
_WHISPER_IMPORT_FAILED = False
_WhisperModel = None


def _load_whisper():
    """Lazily import WhisperModel. Returns the class or None on failure."""
    global _WhisperModel, _WHISPER_IMPORT_FAILED
    if _WhisperModel is not None:
        return _WhisperModel
    if _WHISPER_IMPORT_FAILED:
        return None
    try:
        from faster_whisper import WhisperModel as _WM  # type: ignore
        _WhisperModel = _WM
        return _WhisperModel
    except Exception as exc:  # pragma: no cover - depends on optional dep
        _WHISPER_IMPORT_FAILED = True
        logger.info("faster-whisper unavailable (%s); voice transcription disabled", exc)
        return None


class VoiceTranscriber:
    """Voice transcription using Faster-Whisper"""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en"
    ):
        """
        Initialize voice transcriber

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to use ('cuda' or 'cpu')
            compute_type: Computation type (int8, float16, float32)
            language: Default language for transcription
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language

        WhisperModel = _load_whisper()
        if WhisperModel is None:
            raise RuntimeError(
                "Voice transcription unavailable: 'faster-whisper' is not installed. "
                "Install it with: pip install faster-whisper"
            )

        logger.info(f"Loading Whisper model: {model_size} on {device}")

        try:
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe_file(
        self,
        audio_file: Union[str, Path],
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> dict:
        """
        Transcribe audio file

        Args:
            audio_file: Path to audio file (wav, mp3, etc.)
            language: Language code (if None, uses default)
            task: 'transcribe' or 'translate' (to English)

        Returns:
            Dictionary with transcription results
        """
        try:
            if language is None:
                language = self.language

            logger.info(f"Transcribing audio file: {audio_file}")

            segments, info = self.model.transcribe(
                str(audio_file),
                language=language,
                task=task,
                beam_size=5,
                vad_filter=True,  # Voice activity detection
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            # Collect all segments
            transcription_segments = []
            full_text = ""

            for segment in segments:
                segment_data = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip(),
                    'confidence': segment.avg_logprob
                }
                transcription_segments.append(segment_data)
                full_text += segment.text.strip() + " "

            result = {
                'text': full_text.strip(),
                'segments': transcription_segments,
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': info.duration
            }

            logger.info(f"Transcription completed: {len(transcription_segments)} segments")
            return result

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> dict:
        """
        Transcribe audio from bytes

        Args:
            audio_bytes: Audio data as bytes
            language: Language code (if None, uses default)
            task: 'transcribe' or 'translate'

        Returns:
            Dictionary with transcription results
        """
        try:
            # Save bytes to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name

            # Transcribe
            result = self.transcribe_file(tmp_path, language, task)

            # Clean up
            import os
            os.unlink(tmp_path)

            return result

        except Exception as e:
            logger.error(f"Error transcribing audio bytes: {e}")
            raise


# Global singleton instance
_voice_transcriber = None


def get_voice_transcriber(
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "en"
) -> VoiceTranscriber:
    """
    Get or create singleton voice transcriber

    Args:
        model_size: Whisper model size
        device: Device to use
        compute_type: Computation type
        language: Default language

    Returns:
        VoiceTranscriber instance
    """
    global _voice_transcriber

    if _voice_transcriber is None:
        _voice_transcriber = VoiceTranscriber(model_size, device, compute_type, language)

    return _voice_transcriber
