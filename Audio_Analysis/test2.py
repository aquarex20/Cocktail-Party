# tts_continuous.py
import threading, queue, time, collections
import numpy as np
import tkinter as tk
from tkinter import ttk
from scipy.signal import resample_poly

import sounddevice as sd
from TTS.api import TTS
from faster_whisper import WhisperModel

# =================== Config ===================
DEBUG = True

# Audio I/O
IN_SR = 44100              # your mic's native rate
TARGET_SR = 16000          # what Whisper prefers
FRAME_MS = 30
FRAME_SAMPLES = int(IN_SR * FRAME_MS / 1000)

# Rolling buffer (in seconds)
BUFFER_SEC        = 12.0   # how much audio we keep
PROCESS_INTERVAL  = 2.0    # how often we try to transcribe
OVERLAP_SEC       = 0.5    # keep small overlap to avoid cutting words
TAIL_RESERVE_SEC  = 1.0    # don't finalize the last 1s; let it grow next cycle

# TTS
TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"
DRAIN_MS_AFTER_TTS = 250

# STT
STT_MODEL   = "medium.en"     # or "small.en"/"small" for speed, "large-v3" for max accuracy
STT_DEVICE  = "cpu"           # keep cpu unless you know a GPU/Metal backend is available
STT_COMPUTE = "int8"          # fastest CPU-friendly choice
STT_LANGUAGE = None           # set "en"/"it" to force language and improve accuracy

# ==============================================

# Models
tts = TTS(model_name=TTS_MODEL, progress_bar=True, gpu=False)
TTS_SR = getattr(getattr(tts, "synthesizer", None), "output_sample_rate", 22050)
stt_model = WhisperModel(STT_MODEL, device=STT_DEVICE, compute_type=STT_COMPUTE)

# State
state = {
    "playing": False,
    "continuous": False,
    "stream": None,
    "selected_input_index": None,
}
status_lock = threading.Lock()

# Rolling buffer @ 44.1k (int16 bytes)
max_bytes = int(BUFFER_SEC * IN_SR) * 2  # int16 -> 2 bytes/sample
audio_ring = collections.deque()         # deque of bytes chunks (from callback)
ring_bytes_len = 0                       # running length in bytes
ring_lock = threading.Lock()

# Progress markers (in 16 kHz samples, relative to full resampled stream)
processed_cursor_16 = 0   # how much we’ve considered (with overlap)
emitted_until_16    = 0   # up to which time (sample) we’ve emitted text

# Tk UI
root = tk.Tk()
root.title("Hands-free: Rolling Buffer → Whisper (vad_filter) → Coqui-TTS")
root.geometry("960x580")
root.minsize(760, 520)

frm = ttk.Frame(root, padding=12)
frm.pack(fill="both", expand=True)

top = ttk.Frame(frm)
top.pack(fill="x", pady=(0, 10))

# Device picker
ttk.Label(top, text="Microphone:").pack(side="left")
all_dev = sd.query_devices()
dev_names = [
    f"{i}: {d['name']} ({int(d['max_input_channels'])}ch, {int(d.get('default_samplerate', 0))} Hz)"
    for i, d in enumerate(all_dev) if d["max_input_channels"] > 0
]
dev_indices = [i for i, d in enumerate(all_dev) if d["max_input_channels"] > 0]
dev_var = tk.StringVar()
dev_combo = ttk.Combobox(top, values=dev_names, textvariable=dev_var, state="readonly", width=60)
if dev_names:
    dev_combo.current(0)
    state["selected_input_index"] = dev_indices[0]
dev_combo.pack(side="left", padx=(6, 12))

status_var = tk.StringVar(value="Idle — pick your mic and click Start.")
ttk.Label(frm, textvariable=status_var).pack(anchor="w", pady=(0, 8))

btn_row = ttk.Frame(frm); btn_row.pack(fill="x", pady=(0, 6))
btn_start = ttk.Button(btn_row, text="▶ Start")
btn_stop  = ttk.Button(btn_row, text="⏹ Stop")
btn_say   = ttk.Button(btn_row, text="🔊 Speak Text")
btn_clear = ttk.Button(btn_row, text="🧹 Clear Text")
btn_start.pack(side="left", padx=(0,8))
btn_stop.pack(side="left", padx=(0,8))
btn_say.pack(side="left", padx=(0,8))
btn_clear.pack(side="left", padx=(0,8))

ttk.Label(frm, text="Transcript / Input:").pack(anchor="w")
txt = tk.Text(frm, height=16, wrap="word")
txt.pack(fill="both", expand=True, pady=(4, 8))

status_row = ttk.Frame(frm); status_row.pack(fill="x")
vu = ttk.Progressbar(status_row, orient="horizontal", length=240, mode="determinate", maximum=100)
vu.pack(side="right")

def set_status(s: str):
    with status_lock:
        status_var.set(s)

def on_device_change(event=None):
    sel = dev_combo.get()
    if not sel: return
    idx = int(sel.split(":")[0])
    state["selected_input_index"] = idx
    set_status(f"Selected input device index: {idx}")
dev_combo.bind("<<ComboboxSelected>>", on_device_change)

def audio_callback(indata, frames, time_info, status):
    # VU meter
    try:
        payload = bytes(indata)
        x = np.frombuffer(payload, dtype=np.int16).astype(np.float32)
        if x.size:
            rms = np.sqrt(np.mean((x/32768.0)**2) + 1e-12)
            db = max(-60.0, min(0.0, 20*np.log10(rms + 1e-12)))
            pct = int((db + 60)/60 * 100)
            root.after(0, lambda v=pct: vu.config(value=v))
    except Exception:
        pass

    if not state["continuous"] or state["playing"]:
        return
    # Append chunk into rolling ring
    chunk = bytes(indata)  # int16 @ 44.1k
    with ring_lock:
        # Work around: use global-like via dict
        append_chunk_to_ring(chunk)

def append_chunk_to_ring(chunk: bytes):
    global ring_bytes_len
    audio_ring.append(chunk)
    ring_bytes_len += len(chunk)
    # Trim head to keep <= max_bytes
    while ring_bytes_len > max_bytes and audio_ring:
        old = audio_ring.popleft()
        ring_bytes_len -= len(old)

def copy_ring_bytes() -> bytes:
    with ring_lock:
        return b"".join(audio_ring)

def play_tts(text: str):
    if not text.strip(): return
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
        if state["continuous"]:
            set_status("🎤 Listening (continuous)…")

def speak_text_now():
    text_val = txt.get("1.0","end").strip()
    threading.Thread(target=play_tts, args=(text_val,), daemon=True).start()

def processor_loop():
    global processed_cursor_16, emitted_until_16
    set_status("🎤 Listening (continuous)…")
    last_run = 0.0
    while state["continuous"]:
        if state["playing"]:
            time.sleep(0.05)
            continue
        now = time.time()
        if now - last_run < PROCESS_INTERVAL:
            time.sleep(0.05)
            continue
        last_run = now

        # Copy ring -> resample to 16k
        b = copy_ring_bytes()
        if not b:
            continue

        x44 = np.frombuffer(b, dtype=np.int16).astype(np.float32)
        if x44.size == 0:
            continue
        x16 = resample_poly(x44, TARGET_SR, IN_SR).astype(np.float32)
        n16 = x16.shape[0]

        # Decide region to process: from (processed_cursor_16 - overlap) to (n16 - tail_reserve)
        start16 = max(0, processed_cursor_16 - int(OVERLAP_SEC * TARGET_SR))
        end16   = max(0, n16 - int(TAIL_RESERVE_SEC * TARGET_SR))
        if end16 <= start16:
            # Not enough new audio yet
            continue

        segment = x16[start16:end16] / 32768.0
        if DEBUG:
            print(f"[PROC] seg {start16}->{end16} (len {end16-start16} samples)")

        try:
            segments, info = stt_model.transcribe(
                segment,
                language=STT_LANGUAGE,         # set for accuracy
                beam_size=3,
                vad_filter=True,               # Whisper’s own VAD
                condition_on_previous_text=False,
                temperature=[0.0, 0.2],
                no_speech_threshold=0.3,
            )
        except Exception as e:
            set_status(f"⚠️ STT error: {e}")
            continue

        # Emit only segments that go beyond what we already emitted
        new_text_parts = []
        global_offset_sec = start16 / TARGET_SR
        latest_end_global = emitted_until_16 / TARGET_SR

        max_emitted_end_16 = emitted_until_16
        for seg in segments:
            seg_start_g = seg.start + global_offset_sec
            seg_end_g   = seg.end   + global_offset_sec
            # Only take segments with new tail
            if seg_end_g > latest_end_global + 1e-3:
                new_text_parts.append(seg.text.strip())
                # track furthest end
                cand_end_16 = int(round(seg_end_g * TARGET_SR))
                if cand_end_16 > max_emitted_end_16:
                    max_emitted_end_16 = cand_end_16

        if new_text_parts:
            out = " ".join(new_text_parts).strip()
            if out:
                root.after(0, lambda t=out: append_text_and_tts(t))

        # Move cursors forward (keep 1s tail unfinalized)
        processed_cursor_16 = end16
        emitted_until_16    = max_emitted_end_16

def append_text_and_tts(t: str):
    cur = txt.get("1.0","end").strip()
    txt.insert("end", ("" if not cur else "\n") + t + "\n")
    txt.see("end")
    threading.Thread(target=play_tts, args=(t,), daemon=True).start()

def start_continuous():
    if state["continuous"]:
        return
    if state["selected_input_index"] is None:
        set_status("Select a microphone first.")
        return

    # Start stream
    if state["stream"] is None or not state["stream"].active:
        try:
            state["stream"] = sd.RawInputStream(
                device=state["selected_input_index"],
                samplerate=IN_SR,
                channels=1,
                dtype="int16",
                blocksize=FRAME_SAMPLES,
                callback=audio_callback,
            )
            state["stream"].start()
        except Exception as e:
            set_status(f"⚠️ Mic error: {e}")
            return

    # Reset buffers/cursors
    with ring_lock:
        audio_ring.clear()
    global ring_bytes_len, processed_cursor_16, emitted_until_16
    ring_bytes_len = 0
    processed_cursor_16 = 0
    emitted_until_16 = 0

    state["continuous"] = True
    set_status("🎤 Listening (continuous)…")

    # Spin processor thread
    threading.Thread(target=processor_loop, daemon=True).start()

def stop_continuous():
    state["continuous"] = False
    set_status("⏹ Stopped")
    try:
        if state["stream"] is not None:
            state["stream"].stop()
            state["stream"].close()
            state["stream"] = None
    except Exception:
        pass
    # clear ring
    with ring_lock:
        audio_ring.clear()
    global ring_bytes_len
    ring_bytes_len = 0

def clear_text():
    txt.delete("1.0","end")
    set_status("Cleared.")

# Wire up buttons
btn_start.config(command=start_continuous)
btn_stop.config(command=stop_continuous)
btn_say.config(command=speak_text_now)
btn_clear.config(command=clear_text)

def set_default_device():
    try:
        inp_idx, _ = sd.default.device
    except Exception:
        inp_idx = None
    if isinstance(inp_idx, (int,float)) and inp_idx is not None and inp_idx >= 0:
        try:
            pos = dev_indices.index(int(inp_idx))
            dev_combo.current(pos)
            state["selected_input_index"] = int(inp_idx)
        except Exception:
            pass

def on_close():
    stop_continuous()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
set_default_device()
root.mainloop()
