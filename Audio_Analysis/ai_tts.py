# tts_calibrated.py
"""
macOS-compatible TTS/STT application.
All heavy imports and tkinter initialization happen inside main().
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import threading
import queue
import time
import collections
import asyncio
import json

import numpy as np
from scipy.signal import resample_poly
import httpx

# ================= Config =================
DEBUG = True

# Audio I/O
IN_SR = 44100
TARGET_SR = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(IN_SR * FRAME_MS / 1000)

# VAD defaults
DEFAULT_AGGR = 2
PADDING_MS = 900
END_SILENCE_MS = 300
START_VOICE_RATIO = 0.40
END_SILENCE_RATIO = 0.35

# Safety limits
MAX_UTT_MS = 7000
NO_ACTIVITY_MS = 1000

# TTS
TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"
DRAIN_MS_AFTER_TTS = 250

# STT
STT_MODEL = "medium.en"
STT_DEVICE = "cpu"
STT_COMPUTE = "int8"
STT_LANGUAGE = None

# AI Configuration
AI_API_URL = "http://localhost:11434/api/generate"
AI_MODEL = "llama3.2:latest"
INITIAL_PROMPT = """
You are currently in a cocktail party.
You are sitting at a table with a group of people.
You are listening to the conversation.
Do not include stage directions or descriptions of physical actions. Only write text or dialogue.
You need to engage into the conversation when you see fit.
Here's the transcript of the last parts of the conversation:
"""


async def call_ai(transcript: str) -> str | None:
    """Async: call AI API with streaming response."""
    if not transcript:
        return None
    print(f"Calling AI with prompt: {INITIAL_PROMPT + transcript}")

    payload = {
        "model": AI_MODEL,
        "prompt": INITIAL_PROMPT + transcript,
        "stream": True,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream(
                "POST",
                AI_API_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                resp.raise_for_status()

                full_response = ""
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if "error" in data:
                        print(f"AI API error: {data['error']}")
                        return None

                    if "response" in data:
                        full_response += data["response"]

                    if data.get("done", False):
                        break

        return full_response.strip() or None

    except httpx.ConnectError:
        print(f"Cannot connect to AI API at {AI_API_URL}")
        return None
    except httpx.ReadTimeout:
        print("AI API timeout (>30s)")
        return None
    except Exception as e:
        print(f"AI API error: {e}")
        return None


def main():
    """Main entry point - all tkinter and heavy imports happen here."""

    # Import tkinter and audio libraries inside main
    import tkinter as tk
    from tkinter import ttk, messagebox
    import sounddevice as sd
    import webrtcvad

    # Global state for this session
    last_5_transcript = ""
    tts = None
    TTS_SR = 22050
    stt_model = None

    state = {
        "playing": False,
        "listening": False,
        "stream": None,
        "selected_input_index": None,
        "busy": False,
    }

    calib = {
        "done": False,
        "ambient_rms": 0.0,
        "speech_rms": 0.0,
        "start_ratio": START_VOICE_RATIO,
        "end_ratio": END_SILENCE_RATIO,
        "end_ms": END_SILENCE_MS,
        "aggr": DEFAULT_AGGR,
        "gain": 1.0,
    }

    in_q = queue.Queue(maxsize=120)
    ptt = {"rec": False, "buf": []}

    # Create UI
    root = tk.Tk()
    root.title("Calibrated VAD: Mic -> STT -> Coqui-TTS")
    root.geometry("980x640")
    root.minsize(820, 560)

    frm = ttk.Frame(root, padding=12)
    frm.pack(fill="both", expand=True)

    top = ttk.Frame(frm)
    top.pack(fill="x", pady=(0, 10))

    # Device selector
    ttk.Label(top, text="Microphone:").pack(side="left")
    all_dev = sd.query_devices()
    dev_names = [
        f"{i}: {d['name']} ({int(d['max_input_channels'])}ch, {int(d.get('default_samplerate', 0))} Hz)"
        for i, d in enumerate(all_dev) if d["max_input_channels"] > 0
    ]
    dev_indices = [i for i, d in enumerate(all_dev) if d["max_input_channels"] > 0]
    dev_var = tk.StringVar()
    dev_combo = ttk.Combobox(top, values=dev_names, textvariable=dev_var, state="readonly", width=64)
    if dev_names:
        dev_combo.current(0)
        state["selected_input_index"] = dev_indices[0]
    dev_combo.pack(side="left", padx=(6, 12))

    status_var = tk.StringVar(value="Loading models... please wait.")
    ttk.Label(frm, textvariable=status_var).pack(anchor="w", pady=(0, 8))

    btn_row = ttk.Frame(frm)
    btn_row.pack(fill="x", pady=(0, 6))
    btn_cal = ttk.Button(btn_row, text="Calibrate")
    btn_list = ttk.Button(btn_row, text="Start Listening (VAD)")
    btn_stop = ttk.Button(btn_row, text="Stop Listening")
    btn_ptt = ttk.Button(btn_row, text="Push-to-Talk (hold)")
    btn_say = ttk.Button(btn_row, text="Speak Text")
    btn_clr = ttk.Button(btn_row, text="Clear")
    btn_cal.pack(side="left", padx=(0, 8))
    btn_list.pack(side="left", padx=(0, 8))
    btn_stop.pack(side="left", padx=(0, 8))
    btn_ptt.pack(side="left", padx=(0, 8))
    btn_say.pack(side="left", padx=(0, 8))
    btn_clr.pack(side="left", padx=(0, 8))

    ttk.Label(frm, text="Transcript / Input:").pack(anchor="w")
    txt = tk.Text(frm, height=18, wrap="word")
    txt.pack(fill="both", expand=True, pady=(4, 8))

    status_row = ttk.Frame(frm)
    status_row.pack(fill="x")
    ttk.Label(status_row, text="VU:").pack(side="left")
    vu = ttk.Progressbar(status_row, orient="horizontal", length=240, mode="determinate", maximum=100)
    vu.pack(side="left", padx=(6, 16))

    live_lbl = ttk.Label(status_row, text="calib: N/A")
    live_lbl.pack(side="left")

    # VAD
    vad = webrtcvad.Vad(DEFAULT_AGGR)

    ttk.Label(status_row, text="VAD aggr:").pack(side="left", padx=(8, 4))
    aggr_var = tk.IntVar(value=DEFAULT_AGGR)

    def set_status(s: str):
        status_var.set(s)

    def show_live():
        live_lbl.config(text=(
            f"calib: done={calib['done']} | "
            f"start={calib['start_ratio']:.2f} | end={calib['end_ratio']:.2f} | "
            f"end_ms={calib['end_ms']} | aggr={calib['aggr']} | gain x{calib['gain']:.2f}"
        ))

    def set_vad_aggr(val: int):
        try:
            vad.set_mode(val)
            aggr_var.set(val)
            calib["aggr"] = val
            show_live()
            if DEBUG:
                print(f"[VAD] aggr -> {val}")
        except Exception as e:
            if DEBUG:
                print("[VAD] set_mode error:", e)

    aggr_scale = ttk.Scale(status_row, from_=0, to=3, orient="horizontal",
                           command=lambda v: set_vad_aggr(int(float(v))))
    aggr_scale.set(DEFAULT_AGGR)
    aggr_scale.pack(side="left", padx=(0, 12))

    def on_device_change(_=None):
        sel = dev_combo.get()
        if not sel:
            return
        idx = int(sel.split(":")[0])
        state["selected_input_index"] = idx
        set_status(f"Selected mic index: {idx}")

    dev_combo.bind("<<ComboboxSelected>>", on_device_change)

    # Audio callback
    def audio_callback(indata, frames, time_info, status):
        try:
            payload = bytes(indata)
            x = np.frombuffer(payload, dtype=np.int16).astype(np.float32)
            if x.size:
                rms = np.sqrt(np.mean((x / 32768.0) ** 2) + 1e-12)
                db = max(-60.0, min(0.0, 20 * np.log10(rms + 1e-12)))
                pct = int((db + 60) / 60 * 100)
                root.after(0, lambda v=pct: vu.config(value=v))
        except Exception:
            pass

        if state["listening"] and not state["playing"]:
            try:
                in_q.put_nowait(bytes(indata))
            except queue.Full:
                pass

        if ptt["rec"]:
            ptt["buf"].append(bytes(indata))

    def start_stream():
        if state["stream"] is None or not state["stream"].active:
            state["stream"] = sd.RawInputStream(
                device=state["selected_input_index"],
                samplerate=IN_SR,
                channels=1,
                dtype="int16",
                blocksize=FRAME_SAMPLES,
                callback=audio_callback,
            )
            state["stream"].start()

    def stop_stream():
        try:
            if state["stream"] is not None:
                state["stream"].stop()
                state["stream"].close()
        except Exception:
            pass
        state["stream"] = None

    # Model loading
    def load_models():
        nonlocal tts, TTS_SR, stt_model

        try:
            print("[INIT] Importing TTS...")
            from TTS.api import TTS as TTSClass
            print("[INIT] Loading TTS model...")
            tts = TTSClass(model_name=TTS_MODEL, progress_bar=True, gpu=False)
            TTS_SR = getattr(getattr(tts, "synthesizer", None), "output_sample_rate", 22050)
        except Exception as e:
            print(f"[INIT] TTS load error: {e}")

        try:
            print("[INIT] Importing faster_whisper...")
            from faster_whisper import WhisperModel
            print("[INIT] Loading STT model...")
            stt_model = WhisperModel(STT_MODEL, device=STT_DEVICE, compute_type=STT_COMPUTE)
        except Exception as e:
            print(f"[INIT] STT load error: {e}")

        print("[INIT] Models loaded.")
        root.after(0, lambda: set_status("Models loaded. Ready to use."))

    # TTS & STT functions
    def play_tts(text: str):
        nonlocal tts, TTS_SR
        if not text.strip():
            return
        if tts is None:
            set_status("TTS model not loaded yet")
            return
        state["playing"] = True
        set_status("Speaking...")
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
            set_status(f"TTS error: {e}")
        finally:
            state["playing"] = False
            state["busy"] = False
            start_listening()
            if state["listening"]:
                set_status("Listening (VAD)...")

    def stt_transcribe_16k_float(audio16: np.ndarray) -> str:
        nonlocal stt_model
        if stt_model is None:
            set_status("STT model not loaded yet")
            return ""
        try:
            segments, info = stt_model.transcribe(
                audio16,
                language=STT_LANGUAGE,
                beam_size=3,
                vad_filter=True,
                condition_on_previous_text=False,
                temperature=[0.0, 0.2],
                no_speech_threshold=0.3,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            if DEBUG and hasattr(info, "language"):
                print(f"[STT] lang={info.language} p={getattr(info, 'language_probability', 1.0):.2f} len={len(text)}")
            return text
        except Exception as e:
            set_status(f"STT error: {e}")
            return ""

    def transcribe_and_speak_44k(audio_bytes_44k: bytes):
        nonlocal last_5_transcript
        x44 = np.frombuffer(audio_bytes_44k, dtype=np.int16).astype(np.float32)
        if x44.size == 0:
            txt_out = ""
        else:
            x16 = resample_poly(x44, TARGET_SR, IN_SR).astype(np.float32)
            audio16 = x16 / 32768.0
            txt_out = stt_transcribe_16k_float(audio16)

        def after():
            nonlocal last_5_transcript
            if txt_out:
                cur = txt.get("1.0", "end").strip()
                txt.insert("end", ("" if not cur else "\n") + txt_out + "\n")
                txt.see("end")

                def worker():
                    nonlocal last_5_transcript
                    last_5_transcript += "Human: " + txt_out + "\n"
                    text_val = asyncio.run(call_ai(last_5_transcript))
                    if text_val:
                        last_5_transcript += "AI (You): " + text_val + "\n"
                        threading.Thread(target=play_tts, args=(text_val,), daemon=True).start()
                    else:
                        root.after(0, lambda: set_status("No AI response"))

                threading.Thread(target=worker, daemon=True).start()
            else:
                set_status("No speech recognized.")
            if state["listening"]:
                set_status("Listening (VAD)...")

        root.after(0, after)

    # Calibration
    def calibrate():
        if state["selected_input_index"] is None:
            messagebox.showwarning("Mic", "Select a microphone first.")
            return

        def run():
            set_status("Calibration: Step 1/2 - stay quiet for 1.5s...")
            try:
                data_quiet = sd.rec(int(1.5 * IN_SR), samplerate=IN_SR, channels=1, dtype="float32",
                                    device=state["selected_input_index"])
                sd.wait()
            except Exception as e:
                root.after(0, lambda: set_status(f"Calibration error: {e}"))
                return

            q = (data_quiet[:, 0] * 32768.0).astype(np.float32)
            quiet_rms = float(np.sqrt(np.mean((q / 32768.0) ** 2) + 1e-12))

            root.after(0, lambda: set_status("Calibration: Step 2/2 - speak clearly for ~2s..."))
            try:
                data_speech = sd.rec(int(2.0 * IN_SR), samplerate=IN_SR, channels=1, dtype="float32",
                                     device=state["selected_input_index"])
                sd.wait()
            except Exception as e:
                root.after(0, lambda: set_status(f"Calibration error: {e}"))
                return

            s = (data_speech[:, 0] * 32768.0).astype(np.float32)
            speech_rms = float(np.sqrt(np.mean((s / 32768.0) ** 2) + 1e-12))

            desired_peak = 30000.0
            measured_peak = float(np.max(np.abs(s)) + 1e-9)
            gain = min(3.0, max(0.5, desired_peak / measured_peak))

            noise_level = quiet_rms
            start_ratio = 0.40 if noise_level < 0.02 else 0.35
            end_ratio = 0.35 if noise_level < 0.02 else 0.40
            end_ms = 300 if noise_level < 0.02 else 350
            aggr = 2 if noise_level < 0.02 else 3

            calib.update({
                "done": True,
                "ambient_rms": noise_level,
                "speech_rms": speech_rms,
                "start_ratio": start_ratio,
                "end_ratio": end_ratio,
                "end_ms": end_ms,
                "aggr": aggr,
                "gain": gain,
            })
            set_vad_aggr(aggr)
            root.after(0, show_live)
            root.after(0, lambda: set_status("Calibration done. You can Start Listening (VAD) or use PTT."))

            if DEBUG:
                print(f"[CAL] quiet_rms={noise_level:.4f} speech_rms={speech_rms:.4f} "
                      f"gain x{gain:.2f} start={start_ratio:.2f} end={end_ratio:.2f} end_ms={end_ms} aggr={aggr}")

        threading.Thread(target=run, daemon=True).start()

    # VAD worker
    def vad_worker():
        padding_frames = max(1, int(PADDING_MS / FRAME_MS))
        ring = collections.deque(maxlen=padding_frames)
        end_needed = lambda: max(1, int(calib["end_ms"] / FRAME_MS))

        triggered = False
        voiced_frames_44k = []
        first_voiced_t = None
        last_voiced_t = None

        if DEBUG:
            print(f"[VAD] padding_frames={padding_frames}, frame={FRAME_MS}ms @ {IN_SR}Hz")

        while state["listening"]:
            try:
                frame44 = in_q.get(timeout=0.25)
            except queue.Empty:
                continue

            if state["playing"]:
                continue

            f44 = np.frombuffer(frame44, dtype=np.int16).astype(np.float32)
            f16 = resample_poly(f44, TARGET_SR, IN_SR).astype(np.float32)

            g = calib["gain"] if calib["done"] else 1.0
            peak = float(np.max(np.abs(f16)) + 1e-9)
            f16 = (f16 / max(peak, 1.0)) * (30000.0 * g)
            f16 = np.clip(f16, -32768, 32767).astype(np.int16)

            try:
                is_speech = vad.is_speech(f16.tobytes(), TARGET_SR)
            except Exception:
                is_speech = False

            now_ms = time.time() * 1000.0

            if not triggered:
                ring.append((frame44, is_speech))
                if len(ring) < max(3, min(padding_frames, end_needed())):
                    continue
                voiced_frac = sum(1 for _, s in ring if s) / len(ring)
                start_ratio = calib["start_ratio"]
                if voiced_frac >= start_ratio:
                    triggered = True
                    voiced_frames_44k = [f for f, _ in ring]
                    ring.clear()
                    first_voiced_t = last_voiced_t = now_ms
                    if DEBUG:
                        print(f"[VAD] START (voiced={voiced_frac:.2f} >= {start_ratio:.2f})")
                    root.after(0, lambda: set_status("Detected speech..."))
            else:
                voiced_frames_44k.append(frame44)
                ring.append((frame44, is_speech))
                if is_speech:
                    last_voiced_t = now_ms

                trailing_silence = False
                if len(ring) >= end_needed():
                    vf = sum(1 for _, s in ring if s) / len(ring)
                    trailing_silence = (vf <= calib["end_ratio"])

                no_activity = (last_voiced_t is not None) and ((now_ms - last_voiced_t) >= NO_ACTIVITY_MS)
                too_long = (first_voiced_t is not None) and ((now_ms - first_voiced_t) >= MAX_UTT_MS)

                if trailing_silence or no_activity or too_long:
                    reason = "silence" if trailing_silence else ("no-activity" if no_activity else "max-len")
                    if DEBUG:
                        print(f"[VAD] END -> transcribe ({reason})")
                    audio_bytes_44k = b"".join(voiced_frames_44k)
                    voiced_frames_44k = []
                    triggered = False
                    ring.clear()
                    first_voiced_t = last_voiced_t = None
                    state["listening"] = False
                    root.after(0, lambda: set_status("Transcribing..."))
                    threading.Thread(target=transcribe_and_speak_44k, args=(audio_bytes_44k,), daemon=True).start()

    # UI Handlers
    def start_listening():
        if state["selected_input_index"] is None:
            set_status("Select a microphone first.")
            return
        start_stream()
        state["listening"] = True
        set_vad_aggr(calib["aggr"] if calib["done"] else DEFAULT_AGGR)
        show_live()
        set_status("Listening (VAD)... speak")
        threading.Thread(target=vad_worker, daemon=True).start()

    def stop_listening():
        state["listening"] = False
        try:
            while not in_q.empty():
                in_q.get_nowait()
        except Exception:
            pass
        set_status("Listening stopped")

    def start_ptt(_e=None):
        start_stream()
        ptt["rec"] = True
        ptt["buf"] = []
        set_status("Recording (PTT)... release to transcribe")

    def stop_ptt(_e=None):
        ptt["rec"] = False
        if not ptt["buf"]:
            set_status("No audio captured.")
            return
        set_status("Transcribing...")
        audio_bytes_44k = b"".join(ptt["buf"])
        threading.Thread(target=transcribe_and_speak_44k, args=(audio_bytes_44k,), daemon=True).start()

    def speak_text_now():
        nonlocal last_5_transcript
        text_val = txt.get("1.0", "end").strip()
        set_status("Calling AI...")

        def worker():
            nonlocal last_5_transcript
            last_5_transcript += "Human: " + text_val + "\n"
            val = asyncio.run(call_ai(last_5_transcript))
            if val is None:
                val = ""
            last_5_transcript += "AI (You): " + val + "\n"

            def after():
                if val:
                    current = txt.get("1.0", "end").strip()
                    txt.insert("end", ("" if not current else "\n") + val + "\n")
                    txt.see("end")
                    threading.Thread(target=play_tts, args=(val,), daemon=True).start()
                set_status("AI response ready")

            root.after(0, after)

        threading.Thread(target=worker, daemon=True).start()

    def clear_text():
        txt.delete("1.0", "end")
        set_status("Cleared.")

    # Wire buttons
    btn_cal.config(command=calibrate)
    btn_list.config(command=start_listening)
    btn_stop.config(command=stop_listening)
    btn_ptt.bind("<ButtonPress-1>", start_ptt)
    btn_ptt.bind("<ButtonRelease-1>", stop_ptt)
    btn_say.config(command=speak_text_now)
    btn_clr.config(command=clear_text)

    def set_default_device():
        try:
            inp_idx, _ = sd.default.device
        except Exception:
            inp_idx = None
        if isinstance(inp_idx, (int, float)) and inp_idx is not None and inp_idx >= 0:
            try:
                pos = dev_indices.index(int(inp_idx))
                dev_combo.current(pos)
                state["selected_input_index"] = int(inp_idx)
            except Exception:
                pass

    def on_close():
        stop_listening()
        stop_stream()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    set_default_device()

    # Load models in background after UI is ready
    def init_models():
        threading.Thread(target=load_models, daemon=True).start()

    root.after(500, init_models)
    root.mainloop()


if __name__ == "__main__":
    main()
