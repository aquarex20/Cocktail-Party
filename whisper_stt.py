import numpy as np
import os
import torch
import torchaudio
import whisperx
from dotenv import load_dotenv

load_dotenv()

def _get_env(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or str(v).strip() == "" else str(v).strip()


def _pick_device() -> str:
    """
    WhisperX uses PyTorch. Device choice is machine-dependent:
    - cpu: most compatible, slower
    - cuda: NVIDIA GPU (fastest, if available)
    - mps: Apple Silicon (sometimes supported depending on your stack)
    """
    raw = _get_env("WHISPERX_DEVICE", "auto").lower()
    if raw in {"auto", "default"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw


def _pick_compute_type(device: str) -> str:
    """
    WhisperX compute_type is a major CPU/GPU tuning knob.
    Common values: int8 (CPU-friendly), float16 (GPU-friendly).
    """
    raw = _get_env("WHISPERX_COMPUTE_TYPE", "").lower()
    if raw:
        return raw
    if device == "cuda":
        return "float16"
    return "int8"


WHISPERX_MODEL = _get_env("WHISPERX_MODEL", "small")
device = _pick_device()
compute_type = _pick_compute_type(device)

# Load once at import time (heavy). Tune with env vars above.
asr = whisperx.load_model(WHISPERX_MODEL, device, compute_type=compute_type)

TARGET_SR = 16000
MIN_AUDIO_S = float(_get_env("WHISPER_MIN_AUDIO_S", "0.12"))

def to_mono_f32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:  # (n, ch)
        x = x[:, 0]
    x = x.reshape(-1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / 32768.0
    else:
        x = x.astype(np.float32)
    return x

def resample(x: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    if sr == target_sr:
        return x
    t = torch.from_numpy(x).unsqueeze(0)  # (1, n)
    t = torchaudio.functional.resample(t, sr, target_sr)
    return t.squeeze(0).cpu().numpy()

def transcribe_on_pause(
    audio: tuple[int, np.ndarray],
    language: str,
    return_segments: bool = False,
):
    sr, x = audio
    #flip x dimensions
    x = x.reshape(-1)
    x = to_mono_f32(x)

    if len(x) / sr < MIN_AUDIO_S:
        print("Audio too short")
        return None

    x16 = x#resample(x, sr, TARGET_SR)

    # OPTIONAL: if you always know the language, set it to avoid detect + the “<30s” warning
    result = asr.transcribe(x16, language=language)  # or "it"
    segments = result.get("segments", []) or []
    text = " ".join(seg.get("text", "").strip() for seg in segments).strip()

    if return_segments:
        return text, segments, (result.get("language") or language)

    return text