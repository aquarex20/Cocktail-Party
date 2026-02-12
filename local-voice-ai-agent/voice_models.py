"""
TTS and STT models: fastrtc (Moonshine/Kokoro), optional Kokoro Italian, optional Whisper for Italian.
"""

import asyncio
import os
from pathlib import Path

import numpy as np
from fastrtc import get_stt_model, get_tts_model
from loguru import logger

# -----------------------------------------------------------------------------
# Models (loaded at import)
# -----------------------------------------------------------------------------
stt_model = get_stt_model()   # Moonshine (English; no Italian)
tts_model = get_tts_model()   # Kokoro

WHISPER_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "whisper")
_kokoro_italian = None
_stt_italian_model = None


# -----------------------------------------------------------------------------
# Kokoro Italian TTS (lazy)
# -----------------------------------------------------------------------------
def _find_kokoro_models():
    """Find kokoro model files (same logic as local_party)."""
    current_dir = Path(__file__).resolve().parent
    party_dir = current_dir.parent / "party"
    for model_path, voices_path in [
        (current_dir / "kokoro-v1.0.onnx", current_dir / "voices-v1.0.bin"),
        (party_dir / "kokoro-v1.0.onnx", party_dir / "voices-v1.0.bin"),
        (Path("kokoro-v1.0.onnx"), Path("voices-v1.0.bin")),
    ]:
        if model_path.exists() and voices_path.exists():
            return str(model_path), str(voices_path)
    return None, None


def _get_kokoro_italian():
    """Lazy-load Kokoro for Italian TTS. Returns None if models not found."""
    global _kokoro_italian
    if _kokoro_italian is not None:
        return _kokoro_italian
    model_path, voices_path = _find_kokoro_models()
    if not model_path or not voices_path:
        return None
    try:
        from kokoro_onnx import Kokoro
        _kokoro_italian = Kokoro(model_path, voices_path)
        return _kokoro_italian
    except Exception as e:
        logger.warning(f"Could not load Kokoro for Italian TTS: {e}")
        return None


def stream_tts_sync(text, language):
    """Yield (sample_rate, audio_array) chunks. English: default TTS; Italian: Kokoro if available."""
    if not text or not text.strip():
        return
    if language and str(language).lower() == "it":
        kokoro = _get_kokoro_italian()
        if kokoro is None:
            logger.error("Italian TTS requested but Kokoro not available; skipping audio.")
            return
        async def _collect():
            chunks = []
            stream = kokoro.create_stream(text.strip(), voice="if_sara", lang="it")
            async for samples, sample_rate in stream:
                chunks.append((sample_rate, samples))
            return chunks
        try:
            chunks = asyncio.run(_collect())
        except Exception as e:
            logger.error(f"Italian TTS failed: {e}")
            return
        for sr, audio in chunks:
            yield (sr, audio)
    else:
        for chunk in tts_model.stream_tts_sync(text):
            yield chunk


# -----------------------------------------------------------------------------
# STT: default Moonshine; Italian uses Whisper when available (lazy)
# -----------------------------------------------------------------------------
def _get_stt_for_language(lang):
    """Return the STT model to use for the given language. Italian uses Whisper if available."""
    global _stt_italian_model
    if lang and str(lang).lower() == "it":
        if _stt_italian_model is None:
            try:
                import torch
                from transformers import pipeline
                os.makedirs(WHISPER_CACHE_DIR, exist_ok=True)
                device = 0 if torch.cuda.is_available() else -1
                _stt_italian_model = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-small",
                    device=device,
                    model_kwargs={"cache_dir": WHISPER_CACHE_DIR},
                )
                logger.info(f"Loaded Whisper STT for Italian (language=it). Cache: {WHISPER_CACHE_DIR}")
            except Exception as e:
                logger.warning(
                    f"Could not load Whisper for Italian STT: {e}. "
                    "Install with: pip install '.[italian-stt]' (adds torch, torchaudio, transformers). Using default STT (English only)."
                )
                _stt_italian_model = False
        if _stt_italian_model is not False and _stt_italian_model is not None:
            return _stt_italian_model
    return stt_model


def stt_transcribe(audio, language):
    """Run STT on audio. Uses Italian Whisper when language is 'it' and available, else default STT."""
    model = _get_stt_for_language(language)
    if model is stt_model:
        logger.debug("STT: using default (Moonshine) model.")
        return model.stt(audio)
    # Whisper pipeline
    sample_rate, audio_array = audio
    if hasattr(audio_array, "dtype") and audio_array.dtype == np.int16:
        audio_array = audio_array.astype(np.float32) / 32768.0
    audio_array = np.asarray(audio_array).flatten()
    inp = {"array": audio_array, "sampling_rate": sample_rate}
    out = model(inp, generate_kwargs={"language": "it", "task": "transcribe"})
    if isinstance(out, dict):
        text = out.get("text")
    elif isinstance(out, list) and out:
        text = out[0].get("text") if isinstance(out[0], dict) else str(out[0])
    else:
        text = None
    return (text or "").strip()
