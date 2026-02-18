"""
Local Voice Chat Advanced: voice chat with configurable devices, language, and optional party testing.
Configuration, audio utils, and TTS/STT models are in separate modules.
"""

import os
import sys
import threading
import subprocess
import time as time_module
import queue

import numpy as np
import gradio as gr
import httpx
from fastrtc import ReplyOnPause, Stream, WebRTC
from loguru import logger

from audio_utils import get_device_lists, play_audio, set_audio_devices
from llm_client import stream_llm_response
from utilities import extract_last_replies, back_and_forth
from voice_config import (
    CONFIG_HELP_MD,
    LOCAL_PARTY_PORT,
    SYSTEM_DEFAULT_LABEL,
    TROUBLESHOOTING_MD,
    build_parser,
    parse_device,
    run_testing_mode,
)
from voice_models import get_available_voices, stream_tts_sync, stt_transcribe

# Two-agent mode: sample rate used by the input device monitor (must match InputStream)
DEVICE_INPUT_SAMPLE_RATE = 16000
# Silence duration (seconds) after voice before we send the buffer to STT
DEVICE_INPUT_SILENCE_SEC = 1.5
DEVICE_INPUT_VOLUME_THRESHOLD = 0.00001

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------
current_language = "en"
# TTS voice ID for the agent (None = use language default). Only voices valid for current language should be used.
current_tts_voice = None
conversation = "\nTranscript:\n "
someone_talking = False
last_voice_detected = 0.0
last_summary_time = 0
summary = "\nSummary:\n"
ai_is_speaking = False
_session_started = False

audio_monitor_stop = threading.Event()
_playback_lock = threading.Lock()
_echo_lock = threading.Lock()

# Two-agent mode: queue of (sample_rate, audio_array) from device input
_device_input_queue = queue.Queue()
_device_input_buffer = []
_device_input_last_voice_time = 0.0
_device_input_in_capture = False
_device_input_lock = threading.Lock()


# -----------------------------------------------------------------------------
# Conversation / talk pipeline
# -----------------------------------------------------------------------------
def talk():
    global conversation, someone_talking, summary
    logger.debug("🧠 Starting to talk...")
    text_buffer = ""
    ai_reply = "AI:"
    alone = all(r.startswith("AI:") for r in extract_last_replies(conversation, 2))
    is_back_and_forth = back_and_forth(conversation, 6)

    if alone:
        context = "\n".join(extract_last_replies(conversation, 2))
    elif is_back_and_forth:
        context = "\n".join(extract_last_replies(conversation, 6))
    else:
        print("summary activated")
        context = summary + conversation
    if someone_talking:
        return

    for chunk in stream_llm_response(context, alone=alone, is_back_and_forth=is_back_and_forth, language=current_language):
        text_buffer += chunk
        ai_reply += chunk
        if someone_talking:
            return
        if any(p in text_buffer for p in [".", "!", "?"]) or (len(text_buffer) > 80 and text_buffer[-1] == ","):
            speak_part = text_buffer
            text_buffer = ""
            if someone_talking:
                break
            logger.debug(f"🗣️ TTS on chunk: {speak_part!r}")
            for audio_chunk in stream_tts_sync(speak_part, current_language, voice=current_tts_voice):
                if someone_talking:
                    break
                yield audio_chunk

    text_buffer = text_buffer.strip()
    if someone_talking:
        return
    if text_buffer:
        ai_reply += text_buffer
        logger.debug(f"🗣️ TTS on final chunk: {text_buffer!r}")
        for audio_chunk in stream_tts_sync(text_buffer, current_language, voice=current_tts_voice):
            if someone_talking:
                break
            yield audio_chunk
    conversation += "\n" + ai_reply


def _echo_work(audio):
    global conversation, ai_is_speaking
    with _echo_lock:
        try:
            transcript = stt_transcribe(audio, current_language)
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
    threading.Thread(target=_echo_work, args=(audio,), daemon=True).start()
    yield


def create_stream():
    return Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")


# -----------------------------------------------------------------------------
# Summary (stub)
# -----------------------------------------------------------------------------
def make_summary():
    global conversation, summary
    print("summary generated is " + summary)
    conversation = "\n".join(extract_last_replies(conversation, 10))


# -----------------------------------------------------------------------------
# Audio monitor & proactive speech
# -----------------------------------------------------------------------------
def audio_callback(indata, frames, time, status):
    global someone_talking, last_voice_detected, ai_is_speaking, conversation, last_summary_time
    if status:
        print(f"Status: {status}")
    audio_chunk = indata[:, 0]
    volume = (audio_chunk ** 2).mean() ** 0.5
    if volume >= 0.00001:
        someone_talking = True
        last_voice_detected = time_module.time()
        if last_summary_time == 0:
            last_summary_time = time_module.time()
        if time_module.time() - last_summary_time > 100 and conversation != "\nTranscript:\n ":
            last_summary_time = time_module.time()
            make_summary()
    else:
        someone_talking = False
        if time_module.time() - last_voice_detected > 2 and not ai_is_speaking and conversation != "\nTranscript:\n ":
            ai_is_speaking = True
            threading.Thread(target=proactive_speak, daemon=True).start()


def proactive_speak():
    global ai_is_speaking, last_voice_detected
    try:
        logger.debug("🎙️ AI proactively speaking...")
        for audio_chunk in talk():
            sample_rate, audio_data = audio_chunk
            play_audio(audio_data, sample_rate)
        logger.debug("🎙️ Finished proactive speech")
    finally:
        ai_is_speaking = False
        last_voice_detected = time_module.time()


def audio_callback_two_agent(indata, frames, time, status):
    """Accumulate input device audio when volume is high; on silence after voice, put (sr, array) in queue."""
    global _device_input_buffer, _device_input_last_voice_time, _device_input_in_capture
    if status:
        logger.warning(f"Device input status: {status}")
    audio_chunk = np.asarray(indata[:, 0].copy(), dtype=np.float32)
    volume = (audio_chunk ** 2).mean() ** 0.5
    now = time_module.time()
    with _device_input_lock:
        if volume >= DEVICE_INPUT_VOLUME_THRESHOLD:
            _device_input_buffer.append(audio_chunk)
            _device_input_last_voice_time = now
            _device_input_in_capture = True
        else:
            if _device_input_in_capture and (now - _device_input_last_voice_time) >= DEVICE_INPUT_SILENCE_SEC:
                if _device_input_buffer:
                    audio_array = np.concatenate(_device_input_buffer)
                    _device_input_queue.put((DEVICE_INPUT_SAMPLE_RATE, audio_array))
                _device_input_buffer = []
                _device_input_in_capture = False


def _device_input_worker():
    """Take (sample_rate, audio) from queue and run STT + talk + play (same as echo path)."""
    while not audio_monitor_stop.is_set():
        try:
            audio = _device_input_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if audio is None:
            break
        sr, arr = audio
        _echo_work((sr, arr))


def start_audio_monitor(two_agent=False):
    def monitor_loop():
        import sounddevice as sd
        callback = audio_callback_two_agent if two_agent else audio_callback
        with sd.InputStream(samplerate=16000, channels=1, callback=callback):
            logger.info("🎧 Audio monitor started (two_agent=%s)...", two_agent)
            while not audio_monitor_stop.is_set():
                sd.sleep(100)
            logger.info("🎧 Audio monitor stopped")
    thread = threading.Thread(target=monitor_loop, daemon=True)
    thread.start()
    return thread


def start_two_agent_mode():
    """Start monitor (device input -> queue) and worker (queue -> STT -> talk -> play). No proactive_speak."""
    audio_monitor_stop.clear()
    start_audio_monitor(two_agent=True)
    worker = threading.Thread(target=_device_input_worker, daemon=True)
    worker.start()
    start_conversation_printer()
    logger.info("Two-agent mode: listening on input device, responding to headphones + second output.")


def start_conversation_printer():
    def printer_loop():
        while not audio_monitor_stop.is_set():
            print("\n" + "=" * 50)
            print("📝 CONVERSATION:")
            print("=" * 50)
            print(conversation)
            print("=" * 50 + "\n")
            time_module.sleep(5)
    thread = threading.Thread(target=printer_loop, daemon=True)
    thread.start()
    return thread


def _restart_audio_monitor():
    time_module.sleep(1.2)
    audio_monitor_stop.clear()
    start_audio_monitor()


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def _ui_apply_devices(input_name, output_name):
    global _session_started
    in_dev = None
    out_dev = None
    if input_name and str(input_name).strip() and str(input_name).strip() != SYSTEM_DEFAULT_LABEL:
        in_dev = parse_device(str(input_name).strip())
    if output_name and str(output_name).strip() and str(output_name).strip() != SYSTEM_DEFAULT_LABEL:
        out_dev = parse_device(str(output_name).strip())
    if in_dev is not None and out_dev is not None:
        set_audio_devices(in_dev, out_dev)
        if _session_started:
            audio_monitor_stop.set()
            threading.Thread(target=_restart_audio_monitor, daemon=True).start()
            return "✅ Audio devices applied. Session was running; audio monitor is restarting with new devices."
        return "✅ Audio devices applied."
    return "⚠️ Select both input and output (not System default) to apply, or leave as-is to use system default."


def _ui_start_session():
    global _session_started
    if _session_started:
        return "✅ Session already running. Use the voice chat below."
    _session_started = True
    audio_monitor_stop.clear()
    start_audio_monitor()
    start_conversation_printer()
    return "✅ Session started. You can use the voice chat below and hear the AI on your configured output."


def _ui_stop_session():
    global _session_started
    if not _session_started:
        return "⚠️ Session was not running."
    _session_started = False
    audio_monitor_stop.set()
    return "✅ Session stopped. You can change devices and start again."


def _ui_apply_language(lang_choice):
    global current_language
    if lang_choice and "Italian" in str(lang_choice):
        current_language = "it"
        status = "✅ Language set to **Italian**. AI, TTS and STT (Whisper) will use Italian."
    else:
        current_language = "en"
        status = "✅ Language set to **English**. (Default setup.)"
    voices = get_available_voices(current_language)
    voice_choices = ["Default"] + (voices if isinstance(voices, (list, tuple)) else list(voices))
    return status, gr.update(choices=voice_choices, value="Default")


def _ui_apply_voice(voice_choice):
    """Set the TTS voice for the agent. 'Default' or empty = use language default."""
    global current_tts_voice
    if not voice_choice or str(voice_choice).strip() == "Default":
        current_tts_voice = None
    else:
        current_tts_voice = str(voice_choice).strip()


def _ui_run_local_party():
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


def build_ui():
    inputs_list, outputs_list = get_device_lists()
    input_names = [name for _, name in inputs_list]
    output_names = [name for _, name in outputs_list]
    input_choices = [SYSTEM_DEFAULT_LABEL] + (input_names or ["(no input devices)"])
    output_choices = [SYSTEM_DEFAULT_LABEL] + (output_names or ["(no output devices)"])

    with gr.Blocks(title="Local Voice Chat Advanced", css=".gradio-container { max-width: 900px !important }") as demo:
        gr.Markdown("# 🎤 Local Voice Chat Advanced")
        gr.Markdown("Configure audio devices below, then start the session and use the voice chat.")

        with gr.Accordion("🔧 Audio configuration", open=True):
            gr.Markdown("Choose where to capture audio and where to play it.")
            with gr.Row():
                input_dropdown = gr.Dropdown(choices=input_choices, value=SYSTEM_DEFAULT_LABEL, label="Input device", allow_custom_value=True)
                output_dropdown = gr.Dropdown(choices=output_choices, value=SYSTEM_DEFAULT_LABEL, label="Output device", allow_custom_value=True)
            with gr.Row():
                apply_btn = gr.Button("Apply devices", variant="secondary")
                config_status = gr.Textbox(label="Status", interactive=False, value="")
            apply_btn.click(fn=_ui_apply_devices, inputs=[input_dropdown, output_dropdown], outputs=[config_status])

        with gr.Accordion("🌐 Language & TTS voice", open=True):
            gr.Markdown("Choose the language for the AI and TTS. Set this **before** starting the session.")
            with gr.Row():
                language_dropdown = gr.Dropdown(choices=["English", "Italian"], value="Italian" if current_language == "it" else "English", label="Language")
                apply_lang_btn = gr.Button("Apply language", variant="secondary")
                language_status = gr.Markdown(value="")
            voice_choices = ["Default"] + get_available_voices(current_language)
            voice_dropdown = gr.Dropdown(
                choices=voice_choices,
                value="Default",
                label="TTS voice (for current language)",
                allow_custom_value=False,
            )
            apply_lang_btn.click(fn=_ui_apply_language, inputs=[language_dropdown], outputs=[language_status, voice_dropdown])
            language_dropdown.change(fn=_ui_apply_language, inputs=[language_dropdown], outputs=[language_status, voice_dropdown])
            voice_dropdown.change(fn=_ui_apply_voice, inputs=[voice_dropdown], outputs=[])
            gr.Markdown("*Voice list is for the selected language. Change language to see voices for English or Italian. Default uses a built-in voice for that language.*")
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
            audio = WebRTC(mode="send-receive", modality="audio")
            audio.stream(fn=ReplyOnPause(echo), inputs=[audio], outputs=[audio])

    return demo


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    current_language = args.language

    if args.testing:
        test_input, test_output = run_testing_mode(args, get_device_lists, set_audio_devices)
        input_dev = test_input if test_input is not None else parse_device(args.input_device)
        output_dev = test_output if test_output is not None else parse_device(args.output_device)
    else:
        input_dev = parse_device(args.input_device)
        output_dev = parse_device(args.output_device)
    set_audio_devices(input_dev, output_dev)

    if getattr(args, "two_agent", False):
        start_two_agent_mode()
        try:
            logger.info("Two-agent mode running. Press Ctrl+C to stop.")
            while True:
                time_module.sleep(1)
        except KeyboardInterrupt:
            audio_monitor_stop.set()
    elif args.phone:
        monitor_thread = start_audio_monitor()
        printer_thread = start_conversation_printer()
        stream = create_stream()
        try:
            logger.info("Launching with FastRTC phone interface...")
            stream.fastphone()
        finally:
            audio_monitor_stop.set()
    else:
        demo = build_ui()
        try:
            port = getattr(args, "port", None) or 7860
            user_set_port = getattr(args, "port", None) is not None

            def do_launch(p):
                logger.info("Launching with Gradio UI (config + voice chat) on port %s...", p)
                demo.launch(server_port=p)

            # On port-in-use: try next ports when user didn't set --port. On ConnectError: retry once after 2s.
            launch_err = None
            for attempt in range(2):
                try:
                    try:
                        do_launch(port)
                    except OSError as e:
                        if ("port" in str(e).lower() or "address already in use" in str(e).lower()) and not user_set_port:
                            for p in range(port + 1, 7871):
                                try:
                                    do_launch(p)
                                except OSError:
                                    continue
                                break
                            else:
                                raise RuntimeError(
                                    f"No free port in {port}-7870. Use --port <number>."
                                ) from e
                        raise
                except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                    launch_err = e
                    if attempt == 0:
                        logger.warning("Gradio startup check failed (%s), retrying in 2s...", e)
                        time_module.sleep(2)
                    else:
                        raise RuntimeError(
                            "Gradio could not start (connection refused). "
                            "Try: --port 7861, or close other apps using the port."
                        ) from e
        finally:
            audio_monitor_stop.set()
