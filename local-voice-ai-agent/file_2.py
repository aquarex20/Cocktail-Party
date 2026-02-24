from __future__ import annotations

from typing import Any

import base64
import os
import threading
import time
import numpy as np
from pydantic import BaseModel, Field

from diarization import IncrementalDiarizationSession

try:
    from fastapi import FastAPI

    app = FastAPI()
except Exception:
    # Allows importing `diarize_audio_array()` from scripts even if FastAPI
    # isn't installed in that environment.
    app = None


_pipeline: Any | None = None
_sessions: dict[str, IncrementalDiarizationSession] = {}
_session_locks: dict[str, threading.Lock] = {}

_segments_store: dict[str, dict[str, Any]] = {}
_store_lock = threading.Lock()


def _get_hf_token() -> str | None:
    # Prefer your existing .env name, but support standard HuggingFace vars too.
    return (
        os.environ.get("HUGGING_FACE_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )


def _get_pipeline() -> Any | None:
    global _pipeline
    if _pipeline is None:
        try:
            from whisperx.diarize import DiarizationPipeline  # type: ignore
        except Exception:
            return None

        token = _get_hf_token()
        if not token:
            raise RuntimeError(
                "Missing Hugging Face token for WhisperX diarization. "
                "Set `HUGGING_FACE_TOKEN` (or `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`)."
            )

        device = "cpu"
        try:
            import torch  # type: ignore

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

        # WhisperX diarization (pyannote) pipeline. No VAD here.
        _pipeline = DiarizationPipeline(token=token, device=device)
    return _pipeline


def _get_session(name: str) -> IncrementalDiarizationSession:
    key = name or "default"
    sess = _sessions.get(key)
    if sess is None:
        pipeline = _get_pipeline()
        if pipeline is None:
            raise RuntimeError(
                "Diarization requires the `whisperx` package. Install it to enable diarization."
            )
        sess = IncrementalDiarizationSession(pipeline)
        _sessions[key] = sess
    return sess


def _get_session_lock(name: str) -> threading.Lock:
    key = name or "default"
    lock = _session_locks.get(key)
    if lock is None:
        lock = threading.Lock()
        _session_locks[key] = lock
    return lock


def diarize_audio_array(
    name: str,
    audio_array: np.ndarray,
    sample_rate: int,
) -> list[dict[str, Any]]:
    """
    Call this directly from `voice_chat_2.py` right before transcription.

    `audio_array` must be mono 1D float32. (Your `preprocess_audio()` already does this.)
    """
    try:
        sess = _get_session(name)
    except RuntimeError:
        # Keep the main app running even if whisperx isn't installed yet.
        return []

    # Protect per-session state (IncrementalDiarizationSession isn't thread-safe).
    with _get_session_lock(name):
        sess.ingest(audio_array, sample_rate)
    return [
        {"start": float(s), "end": float(e), "speaker": str(spk)}
        for (s, e, spk) in sess.global_segments
    ]


class DiarizationRequest(BaseModel):
    name: str = Field(default="default")
    sample_rate: int = Field(default=16000, ge=1)
    audio: list[float]


if app is not None:

    @app.post("/diarize")
    def diarize(request: DiarizationRequest):
        audio_np = np.asarray(request.audio, dtype=np.float32)
        segments = diarize_audio_array(request.name, audio_np, request.sample_rate)
        return {"segments": segments}

    class DiarizationIngestRequest(BaseModel):
        name: str = Field(default="default")
        sample_rate: int = Field(default=16000, ge=1)
        audio_f32_b64: str

    def _store_segments(name: str, segments: list[dict[str, Any]]):
        with _store_lock:
            _segments_store[name] = {
                "updated_at": time.time(),
                "segments": segments,
            }

    @app.post("/diarize/ingest", status_code=202)
    def diarize_ingest(request: DiarizationIngestRequest):
        # Fire-and-forget: run diarization in a background thread so we don't block the request.
        raw = base64.b64decode(request.audio_f32_b64.encode("ascii"))
        audio_np = np.frombuffer(raw, dtype=np.float32)

        def _work():
            try:
                segs = diarize_audio_array(request.name, audio_np, request.sample_rate)
                _store_segments(request.name, segs)
            except Exception as e:
                with _store_lock:
                    _segments_store[request.name] = {
                        "updated_at": time.time(),
                        "error": str(e),
                        "segments": [],
                    }

        threading.Thread(target=_work, daemon=True).start()
        return {"queued": True}

    @app.get("/diarize/segments/{name}")
    def diarize_segments(name: str):
        with _store_lock:
            return _segments_store.get(name, {"updated_at": None, "segments": []})