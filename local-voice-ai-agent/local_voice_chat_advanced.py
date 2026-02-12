"""
Local Voice Chat Advanced: voice chat with configurable devices, language, and optional party testing.
Configuration, audio utils, and TTS/STT models are in separate modules.
"""

import os
import sys
import threading
import subprocess
import time as time_module

import gradio as gr
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
from voice_models import stream_tts_sync, stt_transcribe

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------
current_language = "en"
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
            for audio_chunk in stream_tts_sync(speak_part, current_language):
                if someone_talking:
                    break
                yield audio_chunk

    text_buffer = text_buffer.strip()
    if someone_talking:
        return
    if text_buffer:
        ai_reply += text_buffer
        logger.debug(f"🗣️ TTS on final chunk: {text_buffer!r}")
        for audio_chunk in stream_tts_sync(text_buffer, current_language):
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


def start_audio_monitor():
    def monitor_loop():
        import sounddevice as sd
        with sd.InputStream(samplerate=16000, channels=1, callback=audio_callback):
            logger.info("🎧 Audio monitor started...")
            while not audio_monitor_stop.is_set():
                sd.sleep(100)
            logger.info("🎧 Audio monitor stopped")
    thread = threading.Thread(target=monitor_loop, daemon=True)
    thread.start()
    return thread


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
        return "✅ Language set to **Italian**. AI, TTS and STT (Whisper) will use Italian."
    current_language = "en"
    return "✅ Language set to **English**. (Default setup.)"


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
            gr.Markdown("Choose where to capture audio (e.g. **BlackHole 2ch** for the party) and where to play it (e.g. your **headphones/speakers**). Leave as *System default* to use system defaults.")
            with gr.Row():
                input_dropdown = gr.Dropdown(choices=input_choices, value=SYSTEM_DEFAULT_LABEL, label="Input device", allow_custom_value=True)
                output_dropdown = gr.Dropdown(choices=output_choices, value=SYSTEM_DEFAULT_LABEL, label="Output device", allow_custom_value=True)
            with gr.Row():
                apply_btn = gr.Button("Apply devices", variant="secondary")
                config_status = gr.Textbox(label="Status", interactive=False, value="")
            apply_btn.click(fn=_ui_apply_devices, inputs=[input_dropdown, output_dropdown], outputs=[config_status])

        with gr.Accordion("🌐 Language", open=True):
            gr.Markdown("Choose the language for the AI and TTS. Set this **before** starting the session.")
            with gr.Row():
                language_dropdown = gr.Dropdown(choices=["English", "Italian"], value="Italian" if current_language == "it" else "English", label="Language")
                apply_lang_btn = gr.Button("Apply language", variant="secondary")
                language_status = gr.Markdown(value="")
            apply_lang_btn.click(fn=_ui_apply_language, inputs=[language_dropdown], outputs=[language_status])
            language_dropdown.change(fn=_ui_apply_language, inputs=[language_dropdown], outputs=[language_status])
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

    if args.phone:
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
            logger.info("Launching with Gradio UI (config + voice chat)...")
            demo.launch()
        finally:
            audio_monitor_stop.set()
