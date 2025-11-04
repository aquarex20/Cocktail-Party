import threading
import queue
import time
import collections
import numpy as np
import tkinter as tk
from tkinter import ttk
from scipy.signal import resample_poly

import sounddevice as sd
import webrtcvad

# --- TTS (coqui-tts) ---
from TTS.api import TTS  # from 'coqui-tts'

# --- STT (faster-whisper) ---
from faster_whisper import WhisperModel

# ============ Config ============
DEBUG = True

# VAD settings
DEFAULT_AGGR = 2       # 0..3 (3 = most aggressive)
VAD_AGGRESSIVENESS = 2
PADDING_MS      = 800   # look-back context (must be >= END_SILENCE_MS)
END_SILENCE_MS  = 100   # ~400 ms of non-speech to mark end
START_VOICE_RATIO = 0.35
END_SILENCE_RATIO = 0.1

# Audio I/O
IN_SAMPLE_RATE = 44100        # your real mic rate
TARGET_SAMPLE_RATE = 16000    # what VAD & Whisper expect
FRAME_MS = 30
FRAME_SAMPLES = int(IN_SAMPLE_RATE * FRAME_MS / 1000)

# TTS
DRAIN_MS_AFTER_TTS = 250
TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"
TTS_USE_GPU = False

# STT
STT_MODEL   = "medium.en"   # good balance; try "small.en" for speed, "large-v3" for max accuracy
STT_DEVICE  = "cpu"
STT_COMPUTE = "int8"
STT_LANGUAGE = None         # set "en"/"it" for better accuracy on short clips

# ================================

# ---- Models init ----
tts = TTS(model_name=TTS_MODEL, progress_bar=True, gpu=TTS_USE_GPU)
TTS_SR = getattr(getattr(tts, "synthesizer", None), "output_sample_rate", 22050)

stt_model = WhisperModel(STT_MODEL, device=STT_DEVICE, compute_type=STT_COMPUTE)

# ---- State / Queues ----
in_q: "queue.Queue[bytes]" = queue.Queue(maxsize=60)
state = {
    "listening": False,
    "playing": False,
    "stream": None,
    "worker_thread": None,
    "selected_input_index": None,
}

# Push-to-Talk buffer
rec_state = {"rec": False, "buf": []}

# ---- Tk UI ----
root = tk.Tk()
root.title("Mic → STT → Coqui TTS (offline)")
root.geometry("960x600")
root.minsize(760, 520)

frm = ttk.Frame(root, padding=12)
frm.pack(fill="both", expand=True)

top = ttk.Frame(frm)
top.pack(fill="x", pady=(0, 10))

# Device selector
ttk.Label(top, text="Microphone:").pack(side="left")
all_devices = sd.query_devices()
device_names = [
    f"{i}: {d['name']} ({int(d['max_input_channels'])}ch, {int(d.get('default_samplerate', 0))} Hz)"
    for i, d in enumerate(all_devices) if d["max_input_channels"] > 0
]
device_indices = [i for i, d in enumerate(all_devices) if d["max_input_channels"] > 0]

device_var = tk.StringVar()
device_combo = ttk.Combobox(top, values=device_names, textvariable=device_var, state="readonly", width=60)
if device_names:
    device_combo.current(0)
    state["selected_input_index"] = device_indices[0]
device_combo.pack(side="left", padx=(6, 12))

def set_status(s: str):
    status_var.set(s)

def on_device_change(event=None):
    sel = device_combo.get()
    if not sel:
        return
    idx = int(sel.split(":")[0])
    state["selected_input_index"] = idx
    set_status(f"Selected input device index: {idx}")

device_combo.bind("<<ComboboxSelected>>", on_device_change)

# VAD aggressiveness slider
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

def set_vad_aggr(val: int):
    try:
        vad.set_mode(val)
        aggr_var.set(val)
        if DEBUG: print(f"[VAD] aggressiveness -> {val}")
    except Exception as e:
        if DEBUG: print("[VAD] set_mode error:", e)

aggr_var = tk.IntVar(value=DEFAULT_AGGR)
ttk.Label(top, text="VAD aggressiveness (0–3):").pack(side="left", padx=(6, 4))
aggr_scale = ttk.Scale(top, from_=0, to=3, orient="horizontal",
                       command=lambda v: set_vad_aggr(int(float(v))))
aggr_scale.set(DEFAULT_AGGR)
aggr_scale.pack(side="left", padx=(0, 12))

# Buttons row
btn_row = ttk.Frame(frm)
btn_row.pack(fill="x", pady=(0, 6))
btn_listen = ttk.Button(btn_row, text="🎤 Start Listening")
btn_stop_listen = ttk.Button(btn_row, text="🛑 Stop Listening")
btn_ptt = ttk.Button(btn_row, text="🎙️ Push-to-Talk (hold)")
btn_speak = ttk.Button(btn_row, text="🔊 Speak Text")
btn_test = ttk.Button(btn_row, text="🎙️ Test Record 2s")
btn_clear = ttk.Button(btn_row, text="🧹 Clear Text")

btn_listen.pack(side="left", padx=(0, 8))
btn_stop_listen.pack(side="left", padx=(0, 8))
btn_ptt.pack(side="left", padx=(0, 8))
btn_speak.pack(side="left", padx=(0, 8))
btn_test.pack(side="left", padx=(0, 8))
btn_clear.pack(side="left", padx=(0, 8))

# Text area
ttk.Label(frm, text="Transcript / Input:").pack(anchor="w")
txt = tk.Text(frm, height=16, wrap="word")
txt.pack(fill="both", expand=True, pady=(4, 8))

# Status + VU meter
status_row = ttk.Frame(frm)
status_row.pack(fill="x")
status_var = tk.StringVar(value="Idle – choose your mic, then click Start Listening or hold PTT.")
status_lbl = ttk.Label(status_row, textvariable=status_var)
status_lbl.pack(side="left")

vu = ttk.Progressbar(status_row, orient="horizontal", length=240, mode="determinate", maximum=100)
vu.pack(side="right")

# ---- Audio helpers ----
def audio_callback(indata, frames, time_info, status):
    if status and DEBUG:
        print("Audio status:", status)
    # Always update VU if we get audio
    try:
        payload = bytes(indata)  # int16 bytes @ 44.1k
        x = np.frombuffer(payload, dtype=np.int16).astype(np.float32)
        if x.size:
            rms = np.sqrt(np.mean((x/32768.0)**2) + 1e-12)
            db = max(-60.0, min(0.0, 20*np.log10(rms + 1e-12)))
            pct = int((db + 60)/60 * 100)
            root.after(0, lambda v=pct: vu.config(value=v))
    except Exception:
        pass

    # Push-to-Talk capture
    if rec_state.get("rec"):
        rec_state["buf"].append(bytes(indata))

    # VAD capture
    if not state["listening"] or state["playing"]:
        return
    try:
        in_q.put_nowait(bytes(indata))
    except queue.Full:
        pass

def start_listening():
    if state["listening"]:
        return
    if state["selected_input_index"] is None:
        set_status("Select a microphone first.")
        return
    if state["stream"] is None or not state["stream"].active:
        try:
            state["stream"] = sd.RawInputStream(
                device=state["selected_input_index"],
                samplerate=IN_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=FRAME_SAMPLES,
                callback=audio_callback,
            )
            state["stream"].start()
        except Exception as e:
            set_status(f"⚠️ Mic error: {e}")
            return
    state["listening"] = True
    set_status("🎤 Listening (VAD)… speak")
    if not state["worker_thread"] or not state["worker_thread"].is_alive():
        t = threading.Thread(target=vad_worker, daemon=True)
        t.start()
        state["worker_thread"] = t

def stop_listening():
    state["listening"] = False
    set_status("🛑 Listening stopped")
    try:
        while not in_q.empty():
            in_q.get_nowait()
    except Exception:
        pass
    vu.config(value=0)

def test_record():
    if state["selected_input_index"] is None:
        set_status("Select a microphone first.")
        return
    set_status("⏺️ Recording 2s test…")
    try:
        data = sd.rec(int(2 * IN_SAMPLE_RATE), samplerate=IN_SAMPLE_RATE,
                      channels=1, dtype="float32", device=state["selected_input_index"])
        sd.wait()
        set_status("▶️ Playing back test…")
        sd.play(data, samplerate=IN_SAMPLE_RATE, blocking=True)
        set_status("✅ Mic OK. Now hold PTT or click Start Listening.")
    except Exception as e:
        set_status(f"⚠️ Test record error: {e}")

def play_tts(text: str):
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
        sd.wait()
        time.sleep(DRAIN_MS_AFTER_TTS / 1000.0)
    except Exception as e:
        set_status(f"⚠️ TTS error: {e}")
    finally:
        state["playing"] = False
        set_status("✅ Finished")
        if not state["listening"]:
            set_status("🎤 Ready (hold PTT to speak)")

def speak_text_now():
    text_val = txt.get("1.0", "end").strip()
    threading.Thread(target=play_tts, args=(text_val,), daemon=True).start()

# ---- STT paths ----
def transcribe_and_speak_44k(audio_bytes_44k: bytes):
    """PTT & VAD both call this with int16@44.1k bytes."""
    x44 = np.frombuffer(audio_bytes_44k, dtype=np.int16).astype(np.float32)
    if x44.size == 0:
        text = ""
    else:
        x16 = resample_poly(x44, TARGET_SAMPLE_RATE, IN_SAMPLE_RATE).astype(np.float32)
        audio_np = x16 / 32768.0
        try:
            segments, info = stt_model.transcribe(
                audio_np,
                language=STT_LANGUAGE,        # set "en"/"it" to improve accuracy
                beam_size=3,
                vad_filter=True,
                condition_on_previous_text=False,
                temperature=[0.0, 0.2],
                no_speech_threshold=0.3,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            if DEBUG:
                print(f"[STT] lang={info.language} p={getattr(info,'language_probability',1.0):.2f} len={len(text)}")
        except Exception as e:
            text = ""
            root.after(0, lambda: set_status(f"⚠️ STT error: {e}"))

    def after():
        if text:
            current = txt.get("1.0", "end").strip()
            txt.insert("end", ("" if not current else "\n") + text + "\n")
            txt.see("end")
            threading.Thread(target=play_tts, args=(text,), daemon=True).start()
        else:
            set_status("⚠️ No speech recognized.")
        if state["listening"]:
            set_status("🎤 Listening (VAD)… speak")
    root.after(0, after)

def vad_worker():
    padding_frames = max(1, int(PADDING_MS / FRAME_MS))
    end_needed     = max(1, int(END_SILENCE_MS / FRAME_MS))
    ring = collections.deque(maxlen=padding_frames)
    triggered = False
    voiced_frames_44k = []

    # NEW: timers
    MAX_UTT_MS     = 7000   # hard cap on an utterance (7 s)
    NO_ACTIVITY_MS = 1000   # if 1 s with no voiced frames after start -> end
    first_voiced_t = None
    last_voiced_t  = None

    if DEBUG:
        print(f"[VAD] padding={padding_frames} end_needed={end_needed} frame={FRAME_MS}ms @ {IN_SAMPLE_RATE}Hz")

    while True:
        try:
            frame44 = in_q.get(timeout=0.2)  # ~30 ms @ 44.1k
        except queue.Empty:
            continue

        if state["playing"]:
            continue

        # resample 44.1k -> 16k for VAD decision
        f16 = resample_poly(np.frombuffer(frame44, dtype=np.int16), TARGET_SAMPLE_RATE, IN_SAMPLE_RATE).astype(np.int16)
        try:
            is_speech = vad.is_speech(f16.tobytes(), TARGET_SAMPLE_RATE)
        except Exception:
            is_speech = False

        now = time.time() * 1000.0  # ms

        if not triggered:
            ring.append((frame44, is_speech))
            if len(ring) < min(padding_frames, end_needed):
                continue
            voiced_frac = sum(1 for _, s in ring if s) / len(ring)
            if voiced_frac >= START_VOICE_RATIO:
                triggered = True
                voiced_frames_44k = [f for f, _ in ring]
                ring.clear()
                first_voiced_t = now
                last_voiced_t  = now
                if DEBUG: print("[VAD] START speech")
                root.after(0, lambda: set_status("🎙️ Detected speech…"))
        else:
            voiced_frames_44k.append(frame44)
            ring.append((frame44, is_speech))
            if is_speech:
                last_voiced_t = now

            # 1) Normal end condition: trailing-silence window
            if len(ring) >= end_needed:
                voiced_frac = sum(1 for _, s in ring if s) / len(ring)
                trailing_silence = voiced_frac <= END_SILENCE_RATIO
            else:
                trailing_silence = False

            # 2) Safety: no activity for a while
            no_activity = last_voiced_t and (now - last_voiced_t) >= NO_ACTIVITY_MS

            # 3) Safety: hard max utterance length
            too_long = first_voiced_t and (now - first_voiced_t) >= MAX_UTT_MS

            if trailing_silence or no_activity or too_long:
                audio_bytes_44k = b"".join(voiced_frames_44k)
                voiced_frames_44k = []
                triggered = False
                ring.clear()
                first_voiced_t = last_voiced_t = None
                state["listening"] = False
                if DEBUG:
                    reason = "silence" if trailing_silence else ("no-activity" if no_activity else "max-len")
                    print(f"[VAD] END speech -> transcribe ({reason})")
                root.after(0, lambda: set_status("📝 Transcribing…"))
                threading.Thread(target=transcribe_and_speak_44k, args=(audio_bytes_44k,), daemon=True).start()
# ---- PTT handlers ----
def start_ptt():
    rec_state["rec"] = True
    rec_state["buf"] = []
    set_status("⏺️ Recording… release to transcribe")

def stop_ptt():
    rec_state["rec"] = False
    if not rec_state["buf"]:
        set_status("⚠️ No audio captured.")
        return
    set_status("📝 Transcribing…")
    audio_bytes_44k = b"".join(rec_state["buf"])
    threading.Thread(target=transcribe_and_speak_44k, args=(audio_bytes_44k,), daemon=True).start()

# Wire up buttons
btn_listen.config(command=start_listening)
btn_stop_listen.config(command=stop_listening)
btn_ptt.bind("<ButtonPress-1>",  lambda e: start_ptt())
btn_ptt.bind("<ButtonRelease-1>", lambda e: stop_ptt())
btn_speak.config(command=lambda: threading.Thread(target=speak_text_now, daemon=True).start())
btn_test.config(command=test_record)
btn_clear.config(command=lambda: (txt.delete("1.0", "end"), set_status("Cleared.")))

def set_default_device():
    """Initialize default mic index to system default input if available."""
    try:
        inp_idx, _ = sd.default.device
    except Exception:
        inp_idx = None
    if isinstance(inp_idx, (int, float)) and inp_idx is not None and inp_idx >= 0:
        try:
            pos = device_indices.index(int(inp_idx))
            device_combo.current(pos)
            state["selected_input_index"] = int(inp_idx)
        except Exception:
            pass

def on_close():
    stop_listening()
    try:
        if state["stream"] is not None:
            state["stream"].stop()
            state["stream"].close()
    except Exception:
        pass
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

# init
set_vad_aggr(DEFAULT_AGGR)
set_default_device()
set_status("Select a microphone, then hold PTT or click ‘Start Listening’.")
root.mainloop()
