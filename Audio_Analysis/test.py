# stt_check.py
import sounddevice as sd
import numpy as np
from scipy.signal import resample_poly
from faster_whisper import WhisperModel

IN_SR = 44100
TARGET_SR = 16000
DUR = 4.0

STT_MODEL   = "medium.en"   # or "small.en" / "small" / "large-v3"
STT_DEVICE  = "cpu"
STT_COMPUTE = "int8"
STT_LANGUAGE = None  # set "en" or "it" to force language

print("Recording 4s… Speak clearly.")
data = sd.rec(int(DUR*IN_SR), samplerate=IN_SR, channels=1, dtype="float32")
sd.wait()
x44 = (data[:,0] * 32768.0).astype(np.float32)
x16 = resample_poly(x44, TARGET_SR, IN_SR).astype(np.float32)
audio_np = x16 / 32768.0

print("Loading model…")
stt = WhisperModel(STT_MODEL, device=STT_DEVICE, compute_type=STT_COMPUTE)

print("Transcribing…")
segs, info = stt.transcribe(
    audio_np,
    language=STT_LANGUAGE,
    beam_size=3,
    vad_filter=True,
    condition_on_previous_text=False,
    temperature=[0.0, 0.2],
    no_speech_threshold=0.3,
)
text = " ".join(s.text.strip() for s in segs).strip()
print("== TRANSCRIPT ==")
print(text if text else "(no speech recognized)")
