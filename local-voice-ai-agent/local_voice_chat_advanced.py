import sys
import os
import argparse
import threading
import subprocess
import asyncio
from pathlib import Path

import gradio as gr
from fastrtc import ReplyOnPause, Stream, WebRTC, get_stt_model, get_tts_model
from loguru import logger
import numpy as np
from scipy import signal
import time as time_module  # rename to avoid conflict with callback's 'time' param
import sounddevice as sd
from utilities import extract_transcript, extract_last_replies, back_and_forth

TARGET_SAMPLE_RATE = 48000  # Mac speakers prefer 48kHz

# Language: "en" or "it". Set from UI or CLI; choose from the beginning.
current_language = "en"

# Lazy-loaded Kokoro for Italian TTS (voice if_sara, lang it)
_kokoro_italian = None


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


def _stream_tts_sync(text, language):
    """Yield (sample_rate, audio_array) chunks. Uses default TTS for English, Kokoro Italian for 'it'."""
    if not text or not text.strip():
        return
    if language and language.lower() == "it":
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


def get_device_lists():
    """Return (input_devices, output_devices) as lists of (index, name)."""
    devices = sd.query_devices()
    inputs = []
    outputs = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            inputs.append((i, d["name"]))
        if d["max_output_channels"] > 0:
            outputs.append((i, d["name"]))
    return inputs, outputs


def set_audio_devices(input_device=None, output_device=None):
    """Set sounddevice default device. Accepts name (str) or index (int). Only applies when both are set."""
    if input_device is not None and output_device is not None:
        sd.default.device = (input_device, output_device)
        logger.info(f"Audio devices set: input={input_device!r}, output={output_device!r}")

def play_audio(audio_data, sample_rate):
    """Play audio with resampling to target sample rate."""
    audio_data = np.asarray(audio_data, dtype=np.float32).flatten()
    if audio_data.size == 0:
        return
    # Resample if needed
    if sample_rate != TARGET_SAMPLE_RATE:
        audio_data = signal.resample(audio_data, int(len(audio_data) * TARGET_SAMPLE_RATE / sample_rate))
    sd.play(audio_data, samplerate=TARGET_SAMPLE_RATE, blocking=True)

from llm_client import stream_llm_response, get_llm_response

# Global stop event for the audio monitor thread
audio_monitor_stop = threading.Event()

# Serialise all playback so only one sd.play() runs at a time (avoids PortAudio -9986 / AUHAL conflicts on macOS)
_playback_lock = threading.Lock()

# Audio input/output are set from CLI (--input-device / --output-device) when provided.
# For the party setup: input = BlackHole 2ch (receives from local_party), output = your headphones/speakers.

# STT (transcript agent): default is Moonshine via fastrtc. It does NOT support Italian and has no
# language parameter. For Italian we use an optional Whisper-based STT when available (see below).
stt_model = get_stt_model()  # moonshine/base (English etc.; no Italian)
tts_model = get_tts_model()  # kokoro

# Optional Italian STT (Whisper). Lazy-loaded when current_language == "it".
# Requires: pip install ".[italian-stt]" or pip install transformers torch torchaudio
_stt_italian_model = None

# Local cache for Whisper so the 967MB model is downloaded once and reused (no re-download each run)
WHISPER_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "whisper")


def _get_stt_for_language(lang):
    """Return the STT model to use for the given language. Italian uses Whisper if available."""
    global _stt_italian_model
    if lang and str(lang).lower() == "it":
        if _stt_italian_model is None:
            try:
                import torch
                from transformers import pipeline
                os.makedirs(WHISPER_CACHE_DIR, exist_ok=True)
                # CPU (-1) on Mac; GPU (0) if CUDA available. Whisper small supports Italian.
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
                _stt_italian_model = False  # mark as "tried and failed"
        if _stt_italian_model is not False and _stt_italian_model is not None:
            return _stt_italian_model
    return stt_model


def _stt_transcribe(audio):
    """Run STT on audio. Uses Italian Whisper when current_language is 'it' and available, else default STT."""
    model = _get_stt_for_language(current_language)
    if model is stt_model:
        logger.debug("STT: using default (Moonshine) model.")
        return model.stt(audio)
    # Whisper pipeline: force Italian on every call via generate_kwargs
    sample_rate, audio_array = audio
    if hasattr(audio_array, "dtype") and audio_array.dtype == np.int16:
        audio_array = audio_array.astype(np.float32) / 32768.0
    audio_array = np.asarray(audio_array).flatten()
    inp = {"array": audio_array, "sampling_rate": sample_rate}
    # Pass language and task on each call so Whisper always transcribes in Italian
    out = model(inp, generate_kwargs={"language": "it", "task": "transcribe"})
    # Pipeline can return {"text": "..."} or list of segments
    if isinstance(out, dict):
        text = out.get("text")
    elif isinstance(out, list) and out:
        text = out[0].get("text") if isinstance(out[0], dict) else str(out[0])
    else:
        text = None
    return (text or "").strip()

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")
conversation="""\nTranscript:\n """
example_conversation=""" AI: Hello sir, how are you doing?
User: Uhh good. 
AI: Awesome, do you like the party?
User: Can't complain
AI: Glad to hear that! Wow, this cocktail party is… something.
User: Maybe.  """

someone_talking = False
strikes=0
last_voice_detected=0.0
last_summary_time=0
summary="\nSummary:\n"

def talk():
    global conversation, someone_talking, last_voice_detected, summary
    logger.debug("🧠 Starting to talk...")
    text_buffer = ""
    ai_reply="AI:"
    alone = all(r.startswith("AI:") for r in extract_last_replies(conversation, 2))
    is_back_and_forth = back_and_forth(conversation, 6)  # 3 back and forths
    
    # Determine context to send
    if alone:
        context = "\n".join(extract_last_replies(conversation, 2))
        #print("alone activated")
    elif is_back_and_forth:
        context = "\n".join(extract_last_replies(conversation, 6))
    else:
        print("summary activated")
        context = summary + conversation
    if someone_talking:
        return

    # 1. Stream text from LLM as it's generated (language from current_language)
    for chunk in stream_llm_response(
        context, alone=alone, is_back_and_forth=is_back_and_forth, language=current_language
    ):
        text_buffer += chunk
        ai_reply += chunk
        if someone_talking:
            return
        # Simple heuristic: speak when we see end of sentence or buffer big enough
        if any(p in text_buffer for p in [".", "!", "?"]) or (len(text_buffer) > 80 and text_buffer[-1] == ","):
            speak_part = text_buffer
            text_buffer = ""
            if someone_talking:
                break

            logger.debug(f"🗣️ TTS on chunk: {speak_part!r}")
            # 2. Stream TTS for that chunk (English or Italian)
            for audio_chunk in _stream_tts_sync(speak_part, current_language):
                if someone_talking:
                    break
                yield audio_chunk

    # 3. Flush any remaining text once LLM is done
    text_buffer = text_buffer.strip()
    if someone_talking:
        return False

    if text_buffer:
        ai_reply += text_buffer
        logger.debug(f"🗣️ TTS on final chunk: {text_buffer!r}")
        for audio_chunk in _stream_tts_sync(text_buffer, current_language):
            if someone_talking:
                break
            yield audio_chunk
    conversation+="\n"+ai_reply



def _echo_work(audio):
    """Runs in background: STT -> update conversation -> talk -> play. Kept off the generator to avoid timeout and 'generator already executing'."""
    global conversation, ai_is_speaking
    with _echo_lock:
        try:
            transcript = _stt_transcribe(audio)
        except Exception as e:
            logger.exception(f"STT failed: {e}")
            return
        if not (transcript and transcript.strip()):
            logger.debug("🎤 Transcript empty, skipping response.")
            return
        logger.debug(f"🎤 Transcript: {transcript}")
        conversation += "\nUser:" + transcript
        ai_is_speaking = True
        try:
            for audio_chunk in talk():
                if someone_talking:
                    break
                sample_rate, audio_data = audio_chunk
                play_audio(audio_data, sample_rate)
                if someone_talking:
                    break
        except Exception as e:
            logger.exception(f"Talk/play failed: {e}")
        finally:
            ai_is_speaking = False


def echo(audio):
    """Called by fastrtc each time the user stops speaking (ReplyOnPause). Yields immediately so fastrtc can accept the next turn; heavy work runs in a thread."""
    threading.Thread(target=_echo_work, args=(audio,), daemon=True).start()
    yield  # Must yield at least once so handler stays valid; avoids timeout and "generator already executing"

def create_stream():
    return Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")


# Flag to prevent multiple talk_direct calls
ai_is_speaking = False

# Serialize echo responses so we don't run STT+LLM+TTS concurrently from overlapping user turns
_echo_lock = threading.Lock()

def make_summary():
    global conversation, summary
    # summary = get_llm_response(summary+conversation, summarize=True, language=current_language)
    print("summary generated is " + summary)
    conversation = "\n".join(extract_last_replies(conversation, 10))

def audio_callback(indata, frames, time, status):
    global someone_talking, last_voice_detected, ai_is_speaking, conversation, last_summary_time
    """Process each audio chunk as it arrives."""
    if status:
        print(f"Status: {status}")
    
    audio_chunk = indata[:, 0]  # Get mono channel
    volume = np.sqrt(np.mean(audio_chunk**2))
    #print(f"Volume: {volume:.4f}")
    if volume >= 0.00001:
        someone_talking = True
        last_voice_detected = time_module.time()
        #summarization during user talking to not lose active speech time. 
        if last_summary_time==0 : #replace to 10 instead of 5
            last_summary_time=time_module.time()
        if time_module.time() - last_summary_time > 100 and conversation!="\nTranscript:\n ": #replace to 10 instead of 5
            last_summary_time = time_module.time()
            make_summary()
        #strikes+=1
    else:
        someone_talking = False
        # Trigger proactive speech after 2 seconds of silence
        if time_module.time() - last_voice_detected > 2 and not ai_is_speaking and conversation!="\nTranscript:\n ":
            ai_is_speaking = True
            # Run in separate thread to not block audio callback
            threading.Thread(target=proactive_speak, daemon=True).start()


def proactive_speak():
    """Wrapper to handle proactive speaking - iterates over talk() and plays each chunk."""
    global ai_is_speaking, last_voice_detected
    try:
        logger.debug("🎙️ AI proactively speaking...")
        for audio_chunk in talk():
            sample_rate, audio_data = audio_chunk
            play_audio(audio_data, sample_rate)
        logger.debug("🎙️ Finished proactive speech")
    finally:
        ai_is_speaking = False
        last_voice_detected = time_module.time()  # Reset timer after speaking

def start_audio_monitor():
    """Run audio monitoring in a separate thread."""
    def monitor_loop():
        with sd.InputStream(samplerate=16000, channels=1, callback=audio_callback):
            logger.info("🎧 Audio monitor started...")
            while not audio_monitor_stop.is_set():
                sd.sleep(100)  # Check stop flag every 100ms
            logger.info("🎧 Audio monitor stopped")
    
    thread = threading.Thread(target=monitor_loop, daemon=True)
    thread.start()
    return thread


def start_conversation_printer():
    """Print the conversation every 5 seconds in a separate thread."""
    def printer_loop():
        while not audio_monitor_stop.is_set():
            print("\n" + "="*50)
            print("📝 CONVERSATION:")
            print("="*50)
            print(conversation)
            print("="*50 + "\n")
            time_module.sleep(5)
    
    thread = threading.Thread(target=printer_loop, daemon=True)
    thread.start()
    return thread


# Only set when user clicks "Start session" in the UI (so we don't start monitor twice)
_session_started = False


def _restart_audio_monitor():
    """Stop the monitor, wait, then start it again (for after device change). Run in background."""
    time_module.sleep(1.2)
    audio_monitor_stop.clear()
    start_audio_monitor()


def _ui_apply_devices(input_name, output_name):
    """Apply selected input/output devices. Called from Gradio. Restarts monitor if session is running."""
    global _session_started
    in_dev = None
    out_dev = None
    if input_name and str(input_name).strip() and str(input_name).strip() != SYSTEM_DEFAULT_LABEL:
        in_dev = _parse_device(str(input_name).strip())
    if output_name and str(output_name).strip() and str(output_name).strip() != SYSTEM_DEFAULT_LABEL:
        out_dev = _parse_device(str(output_name).strip())
    if in_dev is not None and out_dev is not None:
        set_audio_devices(in_dev, out_dev)
        if _session_started:
            audio_monitor_stop.set()
            threading.Thread(target=_restart_audio_monitor, daemon=True).start()
            return "✅ Audio devices applied. Session was running; audio monitor is restarting with new devices."
        return "✅ Audio devices applied."
    return "⚠️ Select both input and output (not System default) to apply, or leave as-is to use system default."


def _ui_start_session():
    """Start the audio monitor and conversation printer. Called from Gradio."""
    global _session_started
    if _session_started:
        return "✅ Session already running. Use the voice chat below."
    _session_started = True
    audio_monitor_stop.clear()
    start_audio_monitor()
    start_conversation_printer()
    return "✅ Session started. You can use the voice chat below and hear the AI on your configured output."


def _ui_stop_session():
    """Stop the audio monitor and conversation printer. Called from Gradio."""
    global _session_started
    if not _session_started:
        return "⚠️ Session was not running."
    _session_started = False
    audio_monitor_stop.set()
    return "✅ Session stopped. You can change devices and start again."


LOCAL_PARTY_PORT = 7861


def _ui_apply_language(lang_choice):
    """Set current language from UI. Called from Gradio (button or dropdown change)."""
    global current_language
    if lang_choice and "Italian" in str(lang_choice):
        current_language = "it"
        return "✅ Language set to **Italian**. AI, TTS and STT (Whisper) will use Italian."
    current_language = "en"
    return "✅ Language set to **English**. (Default setup.)"


def _ui_run_local_party():
    """Start local_party.py in the background for testing. Called from Gradio. Returns link."""
    local_party_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_party.py")
    if not os.path.isfile(local_party_path):
        return f"❌ local_party.py not found at {local_party_path}"
    try:
        subprocess.Popen(
            [sys.executable, local_party_path, "--port", str(LOCAL_PARTY_PORT)],
            cwd=os.path.dirname(local_party_path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        link = f"http://127.0.0.1:{LOCAL_PARTY_PORT}"
        return f"✅ local_party.py started. **Open this link:** [{link}]({link}) — play a script there; you should hear it on the same output as this app."
    except Exception as e:
        return f"❌ Failed to start local_party.py: {e}"


CONFIG_HELP_MD = """
**Party + AI Chat setup**

- **local_party.py** (Script Player) outputs to **Script Config** (multi-output: BlackHole 2ch + your headphones/speakers).
- **This app** uses **BlackHole 2ch** as *input* and your **headphones/speakers** as *output*.

Use the **same output device** for both so you hear the script and the AI in one place.
"""

TROUBLESHOOTING_MD = """
**If you can't hear properly:**  
1. Set your system sound output to your headphones/speakers.  
2. In Audio MIDI Setup, ensure **Script Config** includes that same device.  
3. Set this app's **Output device** above to that same device.
"""


SYSTEM_DEFAULT_LABEL = "— System default —"

def build_ui():
    """Build Gradio Blocks with config section + WebRTC voice chat."""
    inputs_list, outputs_list = get_device_lists()
    input_names = [name for _, name in inputs_list]
    output_names = [name for _, name in outputs_list]
    input_choices = [SYSTEM_DEFAULT_LABEL] + (input_names or ["(no input devices)"])
    output_choices = [SYSTEM_DEFAULT_LABEL] + (output_names or ["(no output devices)"])

    with gr.Blocks(title="Local Voice Chat Advanced", css=".gradio-container { max-width: 900px !important }") as demo:
        gr.Markdown("# 🎤 Local Voice Chat Advanced")
        gr.Markdown("Configure audio devices below, then start the session and use the voice chat.")

        with gr.Accordion("🔧 Audio configuration", open=True):
            gr.Markdown("Choose where to capture audio (e.g. **BlackHole 2ch** for the party) and where to play it (e.g. your **headphones/speakers**). Leave as *System default* to use system defaults.")
            with gr.Row():
                input_dropdown = gr.Dropdown(
                    choices=input_choices,
                    value=SYSTEM_DEFAULT_LABEL,
                    label="Input device",
                    allow_custom_value=True,
                )
                output_dropdown = gr.Dropdown(
                    choices=output_choices,
                    value=SYSTEM_DEFAULT_LABEL,
                    label="Output device",
                    allow_custom_value=True,
                )
            with gr.Row():
                apply_btn = gr.Button("Apply devices", variant="secondary")
                config_status = gr.Textbox(label="Status", interactive=False, value="")
            apply_btn.click(
                fn=_ui_apply_devices,
                inputs=[input_dropdown, output_dropdown],
                outputs=[config_status],
            )

        with gr.Accordion("🌐 Language", open=True):
            gr.Markdown("Choose the language for the AI and TTS. Set this **before** starting the session.")
            with gr.Row():
                language_dropdown = gr.Dropdown(
                    choices=["English", "Italian"],
                    value="Italian" if current_language == "it" else "English",
                    label="Language",
                )
                apply_lang_btn = gr.Button("Apply language", variant="secondary")
                language_status = gr.Markdown(value="")
            apply_lang_btn.click(
                fn=_ui_apply_language,
                inputs=[language_dropdown],
                outputs=[language_status],
            )
            # Update current_language as soon as user changes dropdown (so STT/TTS/LLM use it without clicking Apply)
            language_dropdown.change(
                fn=_ui_apply_language,
                inputs=[language_dropdown],
                outputs=[language_status],
            )
            gr.Markdown("*For Italian: the AI and TTS use Italian. Transcription (STT) uses Whisper when you have installed: `pip install transformers torch`; otherwise the default STT is used and may be poor for Italian.*")

        with gr.Accordion("🧪 Testing mode", open=False):
            gr.Markdown(CONFIG_HELP_MD)
            run_party_btn = gr.Button("Run local_party.py for testing", variant="secondary")
            party_status = gr.Markdown(value="")
            run_party_btn.click(fn=_ui_run_local_party, inputs=[], outputs=[party_status])
            gr.Markdown(TROUBLESHOOTING_MD)

        gr.Markdown("---")
        gr.Markdown("### Start / Stop session")
        gr.Markdown("Start the audio monitor (so the AI can react to room audio), then use the voice chat. Stop to change devices.")
        with gr.Row():
            start_btn = gr.Button("Start session", variant="primary")
            stop_btn = gr.Button("Stop session", variant="stop")
            session_status = gr.Textbox(label="Session", interactive=False, value="")
        start_btn.click(fn=_ui_start_session, inputs=[], outputs=[session_status])
        stop_btn.click(fn=_ui_stop_session, inputs=[], outputs=[session_status])

        gr.Markdown("---")
        gr.Markdown("### Voice chat (WebRTC)")
        with gr.Column():
            audio = WebRTC(
                mode="send-receive",
                modality="audio",
            )
            audio.stream(
                fn=ReplyOnPause(echo),
                inputs=[audio],
                outputs=[audio],
            )

    return demo


def _parse_device(value):
    """Parse device arg: int string -> int, else keep as str (device name)."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value.strip()


def _print_config_explanation():
    """Print how local_party and advanced chat fit together."""
    print("""
================================================================================
  AUDIO CONFIGURATION (Party + AI Chat)
================================================================================

  • local_party.py (Script Player)
    Output: "Script Config" (multi-output device that sends audio to:
            - BlackHole 2ch  (virtual cable)
            - Your headphones/speakers (the same device you listen on)

  • local_voice_chat_advanced.py (this app)
    Input:  BlackHole 2ch (receives what the party sends + your mic if routed)
    Output: Your headphones/speakers (so you hear the AI and the party together)

  Both apps should use the SAME output device (your headphones/speakers) so you
  hear the script and the AI in one place.

  If you can't hear properly:
  → Check your Mac (or system) sound output is set to your headphones/speakers.
  → In Audio MIDI Setup, ensure "Script Config" includes that same device.
  → Set this app's output (--output-device) to that same device.

================================================================================
""")


def _run_testing_mode(args):
    """
    Interactive testing: explain config, list devices, optionally run local_party, then gate session start.
    Returns (input_device, output_device) if user entered them; otherwise (None, None) so CLI args are used.
    """
    _print_config_explanation()

    inputs, outputs = get_device_lists()
    print("Available INPUT devices (use name or index for --input-device):")
    for i, name in inputs:
        print(f"  [{i}] {name}")
    print("\nAvailable OUTPUT devices (use name or index for --output-device):")
    for i, name in outputs:
        print(f"  [{i}] {name}")

    test_input_dev = None
    test_output_dev = None
    try:
        inp = input("\nInput device (name or index, or Enter to use --input-device / default): ").strip()
        test_input_dev = _parse_device(inp) if inp else None
        out = input("Output device (name or index, or Enter to use --output-device / default): ").strip()
        test_output_dev = _parse_device(out) if out else None
    except EOFError:
        pass

    local_party_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_party.py")
    if os.path.isfile(local_party_path):
        try:
            reply = input("\nRun local_party.py in the background to test audio? [y/N]: ").strip().lower()
            if reply == "y" or reply == "yes":
                subprocess.Popen(
                    [sys.executable, local_party_path],
                    cwd=os.path.dirname(local_party_path),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print("Started local_party.py. Use it to play a script; you should hear it on your output device.")
        except EOFError:
            pass
    else:
        print(f"\n(local_party.py not found at {local_party_path}; skip running it for testing.)")

    print("\nIf you have sound issues, check:")
    print("  • System output is your headphones/speakers.")
    print("  • Script Config and this app use the SAME output device.")
    print()
    try:
        input("Press Enter to start the voice chat session (or Ctrl+C to exit)...")
    except EOFError:
        pass

    return (test_input_dev, test_output_dev)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local Voice Chat Advanced",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Audio device configuration (optional):
  Use --input-device and --output-device to set where to capture and play audio.
  For the party setup: input = "BlackHole 2ch", output = your headphones/speakers.
  Use --testing to list devices, get configuration help, and optionally run local_party.py.
        """,
    )
    parser.add_argument(
        "--phone",
        action="store_true",
        help="Launch with FastRTC phone interface (get a temp phone number)",
    )
    parser.add_argument(
        "--input-device",
        type=str,
        default=None,
        metavar="NAME_OR_INDEX",
        help="Audio input device (e.g. 'BlackHole 2ch' or index). Required for party setup.",
    )
    parser.add_argument(
        "--output-device",
        type=str,
        default=None,
        metavar="NAME_OR_INDEX",
        help="Audio output device (e.g. your headphones name or index). Same as Script Config output.",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Testing mode: show config explanation, list devices, optionally run local_party.py, then start session.",
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["en", "it"],
        default="en",
        help="Language: 'en' (English) or 'it' (Italian). Set before starting; Italian uses Italian prompts and TTS.",
    )
    args = parser.parse_args()

    # Set language from CLI (UI can override after with Apply language)
    current_language = args.language  # module-level; no global needed in __main__

    if args.testing:
        test_input, test_output = _run_testing_mode(args)
        input_dev = test_input if test_input is not None else _parse_device(args.input_device)
        output_dev = test_output if test_output is not None else _parse_device(args.output_device)
    else:
        input_dev = _parse_device(args.input_device)
        output_dev = _parse_device(args.output_device)
    set_audio_devices(input_dev, output_dev)

    if args.phone:
        # Phone mode: start monitor and printer immediately, use Stream's built-in UI
        monitor_thread = start_audio_monitor()
        printer_thread = start_conversation_printer()
        stream = create_stream()
        try:
            logger.info("Launching with FastRTC phone interface...")
            stream.fastphone()
        finally:
            audio_monitor_stop.set()
    else:
        # Web UI: use custom Gradio app with config + WebRTC; monitor starts when user clicks "Start session"
        demo = build_ui()
        try:
            logger.info("Launching with Gradio UI (config + voice chat)...")
            demo.launch()
        finally:
            audio_monitor_stop.set()
