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
from fastrtc import ReplyOnPause, Stream
from loguru import logger

from audio_utils import get_device_lists, play_audio, set_audio_devices
from conversation_mode import ConversationMode, compute_conversation_context
from llm_client import stream_llm_response
from party_manager import (
    get_party_transcript,
    run_internal_party,
    stop_internal_party,
)
from voice_config import (
    CONFIG_HELP_MD,
    LOCAL_PARTY_PORT,
    SYSTEM_DEFAULT_LABEL,
    TROUBLESHOOTING_MD,
    build_parser,
    parse_device,
    run_testing_mode,
)
from utilities import extract_last_replies
from voice_models import get_available_voices, stream_tts_sync, stt_transcribe
from webrtc_ui import add_voice_chat_block

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
_audio_monitor_thread = None
_conversation_printer_thread = None
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
    if audio_monitor_stop.is_set() or not _session_started:
        return
    logger.debug("🧠 Starting to talk...")
    text_buffer = ""
    ai_reply = "AI:"

    ctx = compute_conversation_context(conversation, summary)
    if ctx.wait_before_speaking_sec > 0:
        time_module.sleep(ctx.wait_before_speaking_sec)
    if ctx.mode == ConversationMode.COCKTAIL_PARTY:
        print("summary activated")
    if someone_talking:
        return

    for chunk in stream_llm_response(ctx.context, mode=ctx.mode, language=current_language):
        if audio_monitor_stop.is_set() or not _session_started:
            return
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
                if someone_talking or audio_monitor_stop.is_set() or not _session_started:
                    return
                yield audio_chunk

    text_buffer = text_buffer.strip()
    if someone_talking:
        return
    if text_buffer:
        ai_reply += text_buffer
        logger.debug(f"🗣️ TTS on final chunk: {text_buffer!r}")
        for audio_chunk in stream_tts_sync(text_buffer, current_language, voice=current_tts_voice):
            if someone_talking or audio_monitor_stop.is_set() or not _session_started:
                return
            yield audio_chunk
    conversation += "\n" + ai_reply


def _echo_work(audio):
    global conversation, ai_is_speaking
    if audio_monitor_stop.is_set() or not _session_started:
        return
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
                if someone_talking or audio_monitor_stop.is_set() or not _session_started:
                    break
                sample_rate, audio_data = audio_chunk
                play_audio(audio_data, sample_rate)
                if someone_talking or audio_monitor_stop.is_set() or not _session_started:
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
    if audio_monitor_stop.is_set() or not _session_started:
        return
    try:
        logger.debug("🎙️ AI proactively speaking...")
        for audio_chunk in talk():
            if audio_monitor_stop.is_set() or not _session_started:
                break
            sample_rate, audio_data = audio_chunk
            play_audio(audio_data, sample_rate)
        logger.debug("🎙️ Finished proactive speech")
    finally:
        ai_is_speaking = False
        last_voice_detected = time_module.time()





def start_audio_monitor():
    def monitor_loop():
        import sounddevice as sd
        callback = audio_callback   
        with sd.InputStream(samplerate=16000, channels=1, callback=callback):
            logger.info("🎧 Audio monitor started")
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


def _ui_start_session(selected_input_name=None):
    global _session_started, _audio_monitor_thread, _conversation_printer_thread
    if _session_started:
        return "✅ Session already running. Use the voice chat below."
    _session_started = True
    audio_monitor_stop.clear()
    _audio_monitor_thread = start_audio_monitor()
    _conversation_printer_thread = start_conversation_printer()
    return "✅ Session started. You can use the voice chat below and hear the AI on your configured output."


def _ui_stop_session():
    global _session_started
    if not _session_started:
        return "⚠️ Session was not running."
    _session_started = False
    audio_monitor_stop.set()
    # Wait for monitor and printer threads to exit (they check audio_monitor_stop)
    for thread, name in [(_audio_monitor_thread, "audio monitor"), (_conversation_printer_thread, "conversation printer")]:
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
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


def _ui_get_transcript():
    """Return current conversation for the dashboard transcript UI."""
    return conversation.strip() or "Transcript will appear here as you speak and the AI responds."


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


def _ui_start_internal_party(num_agents, language, monitor_all, v1, v2, v3, v4):
    """Start internal party (no Blackhole). v1..v4 are voice choices per agent."""
    n = int(num_agents) if num_agents else 2
    lang = "it" if language and "Italian" in str(language) else "en"
    monitor = bool(monitor_all)

    def _voice_or_none(v):
        if not v or str(v).strip() == "Default":
            return None
        return str(v).strip()

    voices = [_voice_or_none(v1), _voice_or_none(v2), _voice_or_none(v3), _voice_or_none(v4)]
    try:
        run_internal_party(num_agents=n, language=lang, tts_voices=voices[:n], monitor_all=monitor)
        return f"✅ Internal party started with {n} agents. They will start talking to each other."
    except Exception as e:
        logger.exception(f"Failed to start internal party: {e}")
        return f"❌ Failed: {e}"


def _ui_stop_internal_party():
    """Stop the internal party."""
    stop_internal_party()
    return "✅ Party stopped."


def _ui_get_party_transcript():
    """Return party transcript for the UI."""
    return get_party_transcript()


def build_ui():
    inputs_list, outputs_list = get_device_lists()
    input_names = [name for _, name in inputs_list]
    output_names = [name for _, name in outputs_list]
    input_choices = [SYSTEM_DEFAULT_LABEL] + (input_names or ["(no input devices)"])
    output_choices = [SYSTEM_DEFAULT_LABEL] + (output_names or ["(no output devices)"])

    with gr.Blocks(title="Local Voice Chat Advanced", css=".gradio-container { max-width: 900px !important }") as demo:
        gr.Markdown("# 🎤 Local Voice Chat Advanced")
        gr.Markdown("Choose a tab: **Single Agent** for one-on-one chat, **AI Party** to run multiple AIs talking to each other, or **Testing** for local_party.")

        with gr.Tabs():
            # -----------------------------------------------------------------
            # Tab 1: Single Agent
            # -----------------------------------------------------------------
            with gr.TabItem("Single Agent", id="single"):
                gr.Markdown("### Single AI voice chat")
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

                gr.Markdown("---")
                gr.Markdown("### Start / Stop session")
                gr.Markdown("Click **Start session** to start the audio monitor and voice chat. Voice chat uses the **Input device** above if not system default (browser may ask for mic permission).")
                with gr.Row():
                    start_btn = gr.Button("Start session", variant="primary")
                    stop_btn = gr.Button("Stop session", variant="stop")
                    session_status = gr.Textbox(label="Session", interactive=False, value="")
                add_voice_chat_block(
                    echo,
                    _ui_start_session,
                    start_btn,
                    input_dropdown,
                    session_status,
                )
                stop_btn.click(fn=_ui_stop_session, inputs=[], outputs=[session_status])

                gr.Markdown("---")
                gr.Markdown("### 📝 Live transcript")
                gr.Markdown("Conversation updates here as you speak and the AI replies.")
                transcript_box = gr.Textbox(
                    label="Transcript",
                    value=_ui_get_transcript(),
                    lines=12,
                    max_lines=24,
                    interactive=False,
                    autoscroll=True,
                )
                transcript_timer = gr.Timer(value=1)
                transcript_timer.tick(fn=_ui_get_transcript, inputs=[], outputs=[transcript_box])

            # -----------------------------------------------------------------
            # Tab 2: AI Party (internal routing, no Blackhole)
            # -----------------------------------------------------------------
            with gr.TabItem("AI Party", id="party"):
                gr.Markdown("### 🎭 Create a party of AIs talking to each other")
                gr.Markdown("""
                **Internal routing** – no Blackhole or virtual audio devices. Audio frames are passed between agents in-process.

                - Each agent: STT → LLM → TTS
                - Agent A's TTS output is fed directly to Agent B's "mic" (and vice versa)
                - Docker-friendly, works on any platform
                """)

                with gr.Accordion("Start Internal Party", open=True):
                    with gr.Row():
                        party_num_agents = gr.Dropdown(
                            choices=["2", "3", "4"],
                            value="2",
                            label="Number of agents",
                        )
                        party_language = gr.Dropdown(
                            choices=["English", "Italian"],
                            value="English",
                            label="Language",
                        )
                        party_monitor = gr.Checkbox(
                            value=True,
                            label="Monitor to headphones (hear all agents)",
                        )
                    voice_choices_en = ["Default"] + get_available_voices("en")
                    voice_choices_it = ["Default"] + get_available_voices("it")

                    def _party_voice_choices(lang):
                        choices = voice_choices_it if lang and "Italian" in str(lang) else voice_choices_en
                        upd = gr.update(choices=choices, value="Default")
                        return upd, upd, upd, upd

                    with gr.Row():
                        party_voice_1 = gr.Dropdown(
                            choices=voice_choices_en,
                            value="Default",
                            label="Agent 1 voice",
                            allow_custom_value=False,
                        )
                        party_voice_2 = gr.Dropdown(
                            choices=voice_choices_en,
                            value="Default",
                            label="Agent 2 voice",
                            allow_custom_value=False,
                        )
                        party_voice_3 = gr.Dropdown(
                            choices=voice_choices_en,
                            value="Default",
                            label="Agent 3 voice",
                            allow_custom_value=False,
                        )
                        party_voice_4 = gr.Dropdown(
                            choices=voice_choices_en,
                            value="Default",
                            label="Agent 4 voice",
                            allow_custom_value=False,
                        )
                    party_language.change(
                        fn=_party_voice_choices,
                        inputs=[party_language],
                        outputs=[party_voice_1, party_voice_2, party_voice_3, party_voice_4],
                    )
                    with gr.Row():
                        start_party_btn = gr.Button("Start Internal Party", variant="primary")
                        stop_party_btn = gr.Button("Stop Party", variant="stop")
                    party_status = gr.Textbox(label="Status", interactive=False, value="")
                    start_party_btn.click(
                        fn=_ui_start_internal_party,
                        inputs=[
                            party_num_agents,
                            party_language,
                            party_monitor,
                            party_voice_1,
                            party_voice_2,
                            party_voice_3,
                            party_voice_4,
                        ],
                        outputs=[party_status],
                    )
                    stop_party_btn.click(fn=_ui_stop_internal_party, inputs=[], outputs=[party_status])

                gr.Markdown("---")
                gr.Markdown("### 📝 Party transcript")
                party_transcript_box = gr.Textbox(
                    label="Conversation",
                    value="Start the party to see the conversation.",
                    lines=14,
                    max_lines=28,
                    interactive=False,
                    autoscroll=True,
                )
                party_transcript_timer = gr.Timer(value=1)
                party_transcript_timer.tick(
                    fn=_ui_get_party_transcript,
                    inputs=[],
                    outputs=[party_transcript_box],
                )

            # -----------------------------------------------------------------
            # Tab 3: Testing
            # -----------------------------------------------------------------
            with gr.TabItem("Testing", id="testing"):
                gr.Markdown("### 🧪 Testing mode")
                with gr.Accordion("Run local_party.py for testing", open=True):
                    gr.Markdown(CONFIG_HELP_MD)
                    run_party_btn = gr.Button("Run local_party.py for testing", variant="secondary")
                    party_status = gr.Markdown(value="")
                    run_party_btn.click(fn=_ui_run_local_party, inputs=[], outputs=[party_status])
                    gr.Markdown(TROUBLESHOOTING_MD)

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
