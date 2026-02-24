from __future__ import annotations

from typing import Any

import base64
import os
import threading
import time
import numpy as np
from pydantic import BaseModel, Field
import dotenv
dotenv.load_dotenv()
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

_align_models: dict[str, Any] = {}
_align_metadata: dict[str, Any] = {}
_align_lock = threading.Lock()

_audio_buffers: dict[str, dict[str, Any]] = {}
_audio_lock = threading.Lock()

_refine_thread_started = False


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


REFINE_INTERVAL_S = _env_int("DIARIZATION_REFINE_INTERVAL_S", 10)
REFINE_MIN_S = float(os.getenv("DIARIZATION_REFINE_MIN_S", "30"))
REFINE_MAX_S = float(os.getenv("DIARIZATION_REFINE_MAX_S", "120"))
STORE_MAX_S = float(os.getenv("DIARIZATION_STORE_MAX_S", "300"))


def _pick_device() -> str:
    """
    Machine-dependent tuning knob (CPU/GPU):
    - Set DIARIZATION_DEVICE=cpu to force CPU
    - Set DIARIZATION_DEVICE=cuda to force NVIDIA GPU
    - Set DIARIZATION_DEVICE=auto (default) to pick cuda when available
    """
    raw = str(os.getenv("DIARIZATION_DEVICE", "auto")).strip().lower()
    if raw in {"auto", "default", ""}:
        try:
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return raw


def _shift_segments_abs(payload: dict[str, Any] | None, offset_s: float) -> dict[str, Any] | None:
    if not payload:
        return payload
    for seg in payload.get("segments", []) or []:
        if "start" in seg and seg["start"] is not None:
            seg["start"] = float(seg["start"]) + offset_s
        if "end" in seg and seg["end"] is not None:
            seg["end"] = float(seg["end"]) + offset_s
        for w in seg.get("words", []) or []:
            if "start" in w and w["start"] is not None:
                w["start"] = float(w["start"]) + offset_s
            if "end" in w and w["end"] is not None:
                w["end"] = float(w["end"]) + offset_s
    return payload


def _append_audio(session_name: str, sr: int, start_sample_abs: int, audio: np.ndarray) -> tuple[int, int]:
    """
    Stores rolling audio for a session.
    Returns (buffer_t0_abs_samples, buffer_len_samples) AFTER storing/trimming.
    """
    key = session_name or "default"
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    with _audio_lock:
        buf = _audio_buffers.get(key)
        if buf is None:
            buf = {
                "sr": int(sr),
                "t0_abs_samples": int(start_sample_abs),
                "audio": audio.copy(),
                "last_refine_len": 0,
                "updated_at": time.time(),
            }
            _audio_buffers[key] = buf
        else:
            if int(buf.get("sr", sr)) != int(sr):
                # If sample rate changes, reset buffer to avoid mixing timelines.
                buf["sr"] = int(sr)
                buf["t0_abs_samples"] = int(start_sample_abs)
                buf["audio"] = audio.copy()
                buf["last_refine_len"] = 0
                buf["updated_at"] = time.time()
            else:
                expected = int(buf["t0_abs_samples"]) + int(buf["audio"].shape[0])
                if int(start_sample_abs) != expected:
                    # Timeline mismatch (dropped frames / restarted session). Reset buffer.
                    buf["t0_abs_samples"] = int(start_sample_abs)
                    buf["audio"] = audio.copy()
                    buf["last_refine_len"] = 0
                    buf["updated_at"] = time.time()
                else:
                    buf["audio"] = np.concatenate([buf["audio"], audio], axis=0)
                    buf["updated_at"] = time.time()

        # Trim to last STORE_MAX_S seconds
        max_samples = int(float(STORE_MAX_S) * int(buf["sr"]))
        if max_samples > 0 and int(buf["audio"].shape[0]) > max_samples:
            drop = int(buf["audio"].shape[0]) - max_samples
            buf["audio"] = buf["audio"][drop:]
            buf["t0_abs_samples"] = int(buf["t0_abs_samples"]) + drop

        return int(buf["t0_abs_samples"]), int(buf["audio"].shape[0])


def _refine_loop():
    while True:
        try:
            # Copy current sessions to avoid holding locks during heavy work.
            with _audio_lock:
                sessions = list(_audio_buffers.keys())

            for name in sessions:
                with _audio_lock:
                    buf = _audio_buffers.get(name)
                    if not buf:
                        continue
                    sr = int(buf["sr"])
                    audio = buf["audio"]
                    t0_abs_samples = int(buf["t0_abs_samples"])
                    last_refine_len = int(buf.get("last_refine_len", 0))

                if audio is None or int(audio.shape[0]) < int(REFINE_MIN_S * sr):
                    continue
                if int(audio.shape[0]) == last_refine_len:
                    continue  # nothing new

                # refine on last REFINE_MAX_S seconds (rolling window)
                max_samples = int(REFINE_MAX_S * sr)
                if max_samples > 0 and int(audio.shape[0]) > max_samples:
                    slice_audio = audio[-max_samples:]
                    slice_t0_abs_samples = t0_abs_samples + (int(audio.shape[0]) - max_samples)
                else:
                    slice_audio = audio
                    slice_t0_abs_samples = t0_abs_samples

                offset_s = float(slice_t0_abs_samples) / float(sr)

                pipeline = _get_pipeline()
                if pipeline is None:
                    continue

                # Run diarization over the slice, and map to stable session speaker IDs using embeddings.
                diar_df, embs = pipeline(slice_audio, return_embeddings=True)
                sess = _get_session(name)
                with _get_session_lock(name):
                    if embs is not None and len(embs) > 0:
                        mapping = sess._match_to_global(embs)  # stable global IDs
                        diar_df["speaker"] = diar_df["speaker"].map(lambda x: mapping.get(x, x))

                diar_df_abs = diar_df.copy()
                diar_df_abs["start"] = diar_df_abs["start"] + offset_s
                diar_df_abs["end"] = diar_df_abs["end"] + offset_s

                refined_segments = [
                    {"start": float(r["start"]), "end": float(r["end"]), "speaker": str(r["speaker"])}
                    for _, r in diar_df_abs.iterrows()
                ]

                # Re-assign speakers to stored aligned utterances that overlap this refine window.
                try:
                    import whisperx  # type: ignore
                except Exception:
                    whisperx = None

                win_t0 = offset_s
                win_t1 = offset_s + float(slice_audio.shape[0]) / float(sr)

                with _store_lock:
                    entry = _segments_store.get(name, {})
                    utterances = entry.get("utterances") or {}

                if whisperx is not None and utterances:
                    for uid, u in list(utterances.items()):
                        aligned_abs = u.get("aligned")
                        if not aligned_abs:
                            continue
                        u_start = u.get("start_abs")
                        u_end = u.get("end_abs")
                        if u_start is None or u_end is None:
                            continue
                        if float(u_end) < win_t0 or float(u_start) > win_t1:
                            continue

                        df_slice = diar_df_abs[
                            (diar_df_abs["end"] >= float(u_start)) & (diar_df_abs["start"] <= float(u_end))
                        ]
                        if df_slice is None or len(df_slice) == 0:
                            continue

                        assigned_refined = whisperx.assign_word_speakers(df_slice, aligned_abs)
                        utterances[uid] = {
                            **u,
                            "assigned_refined": assigned_refined,
                            "refined_at": time.time(),
                        }

                with _store_lock:
                    entry = _segments_store.get(name, {})
                    entry["refined"] = {
                        "updated_at": time.time(),
                        "window": {"start": win_t0, "end": win_t1},
                        "segments": refined_segments,
                    }
                    entry["utterances"] = utterances
                    _segments_store[name] = entry

                with _audio_lock:
                    buf = _audio_buffers.get(name)
                    if buf:
                        buf["last_refine_len"] = int(buf["audio"].shape[0])

        except Exception:
            # keep the loop alive no matter what
            pass

        time.sleep(max(1, int(REFINE_INTERVAL_S)))
        print("Refining loop")


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

        # WhisperX diarization (pyannote) pipeline. No VAD here.
        _pipeline = DiarizationPipeline(token=token, device=_pick_device())
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
    Call this from `voice_chat.py` right before transcription.

    `audio_array` must be mono 1D float32. (Your `preprocess_audio()` already does this.)
    """
    try:
        sess = _get_session(name)
    except RuntimeError:
        # Keep the main app running even if whisperx isn't installed yet.
        return []

    # Protect per-session state (IncrementalDiarizationSession isn't thread-safe).
    with _get_session_lock(name):
        sess.ingest_chunk(audio_array, sample_rate)
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
        utterance_id: str | None = None
        sample_rate: int = Field(default=16000, ge=1)
        audio_f32_b64: str
        whisper_segments: list[dict[str, Any]] | None = None
        language: str | None = None

    def _store_segments(name: str, segments: list[dict[str, Any]]):
        with _store_lock:
            _segments_store[name] = {
                "updated_at": time.time(),
                "segments": segments,
            }

    def _get_align(language: str, device: str):
        lang = (language or "en").lower()
        with _align_lock:
            if lang in _align_models and lang in _align_metadata:
                return _align_models[lang], _align_metadata[lang]

        import whisperx  # type: ignore

        model_a, metadata = whisperx.load_align_model(language_code=lang, device=device)
        with _align_lock:
            _align_models[lang] = model_a
            _align_metadata[lang] = metadata
        return model_a, metadata

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

    @app.post("/diarize/ingest_assign", status_code=202)
    def diarize_ingest_assign(request: DiarizationIngestRequest):
        """
        Like /diarize/ingest, but if `whisper_segments` is provided, the server will:
        - align to get word timestamps
        - diarize (WhisperX diarization)
        - assign speakers to words
        and store the enriched transcript under the utterance id.
        """
        raw = base64.b64decode(request.audio_f32_b64.encode("ascii"))
        audio_np = np.frombuffer(raw, dtype=np.float32)

        utterance_id = request.utterance_id or "latest"
        language = request.language or "en"

        def _work():
            try:
                device = _pick_device()

                # 1) diarize this chunk + stable speaker mapping, return local df and offset
                sess = _get_session(request.name)
                with _get_session_lock(request.name):
                    diarize_df_local, chunk_t0_abs = sess.ingest_chunk(audio_np, request.sample_rate)

                chunk_start_abs_samples = int(round(float(chunk_t0_abs) * float(request.sample_rate)))
                _append_audio(request.name, int(request.sample_rate), chunk_start_abs_samples, audio_np)

                # also keep global segments snapshot
                global_segments = [
                    {"start": float(s), "end": float(e), "speaker": str(spk)}
                    for (s, e, spk) in sess.global_segments
                ]

                # 2) if we have transcription segments, align and assign word speakers
                assigned = None
                aligned_abs = None
                if request.whisper_segments:
                    import whisperx  # type: ignore
                    import pandas as pd  # type: ignore

                    model_a, metadata = _get_align(language, device)
                    aligned = whisperx.align(
                        request.whisper_segments,
                        model_a,
                        metadata,
                        audio_np,
                        device,
                        return_char_alignments=False,
                    )

                    diar_df = diarize_df_local
                    if not hasattr(diar_df, "columns"):
                        diar_df = pd.DataFrame(diarize_df_local)

                    assigned = whisperx.assign_word_speakers(diar_df, aligned)

                    # Store absolute-time aligned + assigned payloads (useful for later refinement)
                    aligned_abs = _shift_segments_abs(aligned, float(chunk_t0_abs))
                    assigned = _shift_segments_abs(assigned, float(chunk_t0_abs))

                with _store_lock:
                    entry = _segments_store.get(request.name, {}) #request.name is the session_id
                    utterances = entry.get("utterances") or {}
                    utterances[utterance_id] = { #utterance id is the reply_id sent by the client. 
                        "updated_at": time.time(),
                        "chunk_t0_abs": chunk_t0_abs,
                        "assigned": assigned,
                        "aligned": aligned_abs,
                        "start_abs": float(chunk_t0_abs),
                        "end_abs": float(chunk_t0_abs) + float(audio_np.shape[0]) / float(request.sample_rate),
                    }
                    _segments_store[request.name] = {
                        "updated_at": time.time(),
                        "segments": global_segments,
                        "utterances": utterances,
                    }
            except Exception as e:
                with _store_lock:
                    entry = _segments_store.get(request.name, {})
                    utterances = entry.get("utterances") or {}
                    utterances[utterance_id] = {
                        "updated_at": time.time(),
                        "error": str(e),
                        "assigned": None,
                    }
                    _segments_store[request.name] = {
                        "updated_at": time.time(),
                        "error": str(e),
                        "segments": entry.get("segments", []),
                        "utterances": utterances,
                    }

        threading.Thread(target=_work, daemon=True).start()
        return {"queued": True, "utterance_id": utterance_id}

    @app.get("/diarize/segments/{name}")
    def diarize_segments(name: str):
        with _store_lock:
            return _segments_store.get(name, {"updated_at": None, "segments": []})

    @app.get("/diarize/utterances/{name}")
    def diarize_utterances(name: str):
        with _store_lock:
            entry = _segments_store.get(name, {})
            return entry.get("utterances", {})

    @app.get("/diarize/refined/{name}")
    def diarize_refined(name: str):
        with _store_lock:
            entry = _segments_store.get(name, {})
            return entry.get("refined", {"updated_at": None, "segments": [], "window": None})

    @app.on_event("startup")
    def _start_refiner():
        global _refine_thread_started
        if _refine_thread_started:
            return
        _refine_thread_started = True
        threading.Thread(target=_refine_loop, daemon=True).start()