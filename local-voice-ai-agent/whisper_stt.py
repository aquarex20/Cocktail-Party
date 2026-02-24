import numpy as np
import torch
import torchaudio
import whisperx
from fastrtc import ReplyOnPause
from whisperx.diarize import DiarizationPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
asr = whisperx.load_model("small", device, compute_type="int8")

TARGET_SR = 16000
MIN_AUDIO_S = 0.12

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