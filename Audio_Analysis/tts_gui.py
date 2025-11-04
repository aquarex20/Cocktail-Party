import threading
import queue
import time
import collections
import numpy as np
import tkinter as tk
from tkinter import ttk

import sounddevice as sd
import webrtcvad

# --- TTS (coqui-tts) ---
from TTS.api import TTS  # provided by 'coqui-tts'

# --- STT (faster-whisper) ---
from faster_whisper import WhisperModel


# =========================
# Configuration
# =========================
# Microphone capture (VAD works best at 16 kHz mono)
IN_SAMPLE_RATE = 16000
FRAME_MS = 30                   # 10/20/30 ms supported by webrtcvad
FRAME_SAMPLES = IN_SAMPLE_RATE * FRAME_MS // 1000  # 480 at 16 kHz
VAD_AGGRESSIVENESS = 2          # 0..3 (3 = most aggressive)
START_VOICE_RATIO = 0.6         # fraction voiced frames in padding to trigger speech
END_SILENCE_RATIO = 0.3         # when voiced fraction in padding drops below this, stop
PADDING_MS = 300                # size of ring buffer for VAD lookback
END_SILENCE_MS = 500            # how much trailing silence to decide end of utterance
DRAIN_MS_AFTER_TTS = 250        # small cushion before re-enabling mic

# TTS model (coqui-tts)
TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"  # change if you prefer another
TTS_USE_GPU = False

# STT model (faster-whisper)
# Options: "tiny", "base", "small", "medium", "large-v3" (bigger = better & heavier)
STT_MODEL = "small"             # good balance on Apple Silicon / modern CPUs
STT_DEVICE = "cpu"              # "auto", "cpu", "cuda", "metal" (Apple Silicon)
STT_COMPUTE = "auto"            # "auto", "int8", "int8_float16", "float16", "float32"
STT_LANGUAGE = None             # None = auto-detect; e.g. "en", "fr", "it", "es" to force


# =========================
# Audio + Models Init
# =========================
# Coqui TTS (downloads model on first run, then cached)
tts = TTS(model_name=TTS_MODEL, progress_bar=True, gpu=TTS_USE_GPU)
TTS_SR = getattr(getattr(tts, "synthesizer", None), "output_sample_rate", 22050)

# faster-whisper STT (downloads model on first run, then cached)
stt_model = WhisperModel(STT_MODEL, device=STT_DEVICE, compute_type=STT_COMPUTE)

# WebRTC VAD
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

# Sounddevice output: we’ll use high-level play()/wait(), so no explicit OutputStream needed
in_q: "queue.Queue[bytes]" = queue.Queue(maxsize=50)

state = {
    "listening": False,
    "playing": False,
    "stream": None,
    "worker_thread": None,
}

# =========================
# Tkinter UI
# =========================
root = tk.Tk()
root.title("Mic → STT → Coqui TTS (offline)")
root.geometry("820x420")
root.minsize(600, 360)

frm = ttk.Frame(root, padding=12)
frm.pack(fill="both", expand=True)

ttk.Label(frm, text="Text (auto-filled from mic; you can also edit and press Speak):").pack(anchor="w")

txt = tk.Text(frm, height=10, wrap="word")
txt.pack(fill="both", expand=True, pady=(4, 8))

status_var = tk.StringVar(value="Idle")
ttk.Label(frm, textvariable=status_var).pack(anchor="w", pady=(0, 8))

btn_row = ttk.Frame(frm)
btn_row.pack(fill="x", pady=(0, 6))

btn_listen = ttk.Button(btn_row, text="🎤 Start Listening")
btn_stop_listen = ttk.Button(btn_row, text="🛑 Stop Listening")
btn_speak = ttk.Button(btn_row, text="🔊 Speak Text")
btn_clear = ttk.Button(btn_row, text="🧹 Clear")

btn_listen.pack(side="left", padx=(0, 8))
btn_stop_listen.pack(side="left", padx=(0, 8))
btn_speak.pack(side="left", padx=(0, 8))
btn_clear.pack(side="left")

# =========================
# Helpers
# =========================
def set_status(msg: str):
    status_var.set(msg)

def play_tts(text: str):
    """Synthesize with coqui-tts and play via sounddevice, blocking until finished."""
    if not text.strip():
        return
    state["playing"] = True
    set_status("🔊 Speaking…")
    try:
        audio = tts.tts(text)
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1).astype(np.float32)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        sd.stop()
        sd.play(audio, samplerate=TTS_SR, blocking=False)
        sd.wait()  # returns when last sample is out
        time.sleep(DRAIN_MS_AMOUNT := (DRAIN_MS_AFTER_TTS / 1000.0))
    except Exception as e:
        set_status(f"⚠️ TTS error: {e}")
    finally:
        state["playing"] = False
        set_status("✅ Finished" if text.strip() else "Idle")

def speak_text_now():
    text_val = txt.get("1.0", "end").strip()
    threading.Thread(target=play_tts, args=(text_val,), daemon=True).start()

def clear_text():
    txt.delete("1.0", "end")
    set_status("Cleared.")

def start_listening():
    if state["listening"]:
        return
    # (Re)start input stream
    try:
        if state["stream"] is None:
            stream = sd.RawInputStream(
                samplerate=IN_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=FRAME_SAMPLES,
                callback=audio_callback,
            )
            state["stream"] = stream
            stream.start()
        state["listening"] = True
        set_status("🎤 Listening… (speak)")
        # spin worker thread
        if not state.get("worker_thread") or not state["worker_thread"].is_alive():
            t = threading.Thread(target=vad_worker, daemon=True)
            t.start()
            state["worker_thread"] = t
    except Exception as e:
        set_status(f"⚠️ Mic error: {e}")

def stop_listening():
    state["listening"] = False
    set_status("🛑 Listening stopped")
    # drain queue
    try:
        while not in_q.empty():
            in_q.get_nowait()
    except Exception:
        pass

def audio_callback(indata, frames, time_info, status):
    # Avoid enqueuing while TTS is playing or listening is off
    if status:
        # You may log status here if needed
        pass
    if not state["listening"] or state["playing"]:
        return
    try:
        # indata is a bytes-like object since we use RawInputStream with dtype='int16'
        in_q.put(bytes(indata), block=False)
    except queue.Full:
        pass  # drop if backpressure

def vad_worker():
    """VAD-based segmenter. When it detects an utterance, runs STT + TTS."""
    padding_frames = int(PADDING_MS / FRAME_MS)
    end_needed = int(END_SILENCE_MS / FRAME_MS)
    ring = collections.deque(maxlen=padding_frames)
    triggered = False
    voiced_frames = []

    while True:
        if not state["listening"]:
            time.sleep(0.05)
            continue
        try:
            frame = in_q.get(timeout=0.2)
        except queue.Empty:
            continue

        # skip if currently playing (avoid echo)
        if state["playing"]:
            continue

        is_speech = False
        try:
            is_speech = vad.is_speech(frame, IN_SAMPLE_RATE)
        except Exception:
            # On very short or malformed frames, just treat as silence
            is_speech = False

        if not triggered:
            ring.append((frame, is_speech))
            num_voiced = sum(1 for _, s in ring if s)
            if len(ring) == 0:
                continue
            if num_voiced / len(ring) >= START_VOICE_RATIO:
                # Start of speech
                triggered = True
                voiced_frames = [f for f, _ in ring]
                ring.clear()
                root.after(0, lambda: set_status("🎙️ Detected speech…"))
        else:
            # Already triggered, keep collecting
            voiced_frames.append(frame)
            ring.append((frame, is_speech))
            num_voiced = sum(1 for _, s in ring if s)
            if len(ring) >= end_needed and (num_voiced / len(ring)) <= END_SILENCE_RATIO:
                # End of utterance
                audio_bytes = b"".join(voiced_frames)
                voiced_frames = []
                triggered = False
                ring.clear()

                # Pause listening while we process + speak to avoid feedback
                state["listening"] = False
                root.after(0, lambda: set_status("📝 Transcribing…"))

                # Run STT and TTS in a dedicated thread to keep this worker responsive
                threading.Thread(
                    target=transcribe_and_speak,
                    args=(audio_bytes,),
                    daemon=True
                ).start()

def transcribe_and_speak(audio_bytes: bytes):
    # Convert int16 PCM bytes to float32 numpy, 16 kHz
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Run STT (faster-whisper); collect text from segments
    try:
        segments, info = stt_model.transcribe(
            audio_np,
            language=STT_LANGUAGE,   # None = auto-detect
            beam_size=1,
            vad_filter=False,        # we already did VAD
            word_timestamps=False
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as e:
        text = ""
        root.after(0, lambda: set_status(f"⚠️ STT error: {e}"))

    # Put transcript in the box
    def update_ui_and_speak():
        if text:
            # append text to box
            existing = txt.get("1.0", "end").strip()
            if existing:
                txt.insert("end", ("\n" if not existing.endswith("\n") else "") + text + "\n")
            else:
                txt.insert("end", text + "\n")
            txt.see("end")
            # Speak it back
            play_tts(text)
        else:
            set_status("⚠️ No speech recognized.")
        # resume listening if user hasn't pressed Stop
        if not state["playing"]:
            state["listening"] = True
            set_status("🎤 Listening…")
    root.after(0, update_ui_and_speak)


# =========================
# Wire up buttons
# =========================
btn_listen.config(command=start_listening)
btn_stop_listen.config(command=stop_listening)
btn_speak.config(command=speak_text_now)
btn_clear.config(command=clear_text)

# Start the GUI loop
set_status("Idle – ready.")
root.mainloop()
