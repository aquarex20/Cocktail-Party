"""
Local Voice Chat Advanced: voice chat with configurable devices, language, and optional party testing.
Configuration, audio utils, and TTS/STT models are in separate modules.
"""

import os
import sys

# Disable SSL verification for self-signed cert (must run before httpx/gradio imports)
if os.environ.get("USE_SSL", "").lower() in ("1", "true", "yes"):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
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
    get_agent_states,
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

# Placeholder labels when no audio devices (e.g. Docker)
NO_INPUT_PLACEHOLDER = "(no input devices)"
NO_OUTPUT_PLACEHOLDER = "(no output devices)"

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
    # Reject placeholder labels (no real devices, e.g. in Docker)
    if in_dev in (NO_INPUT_PLACEHOLDER, NO_OUTPUT_PLACEHOLDER) or out_dev in (NO_INPUT_PLACEHOLDER, NO_OUTPUT_PLACEHOLDER):
        return "⚠️ No host audio devices available (normal in Docker). Use **System default** — voice chat uses your browser's mic and speakers."
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


def _ui_start_party_with_scene(num_agents, language, monitor_all, v1, v2, v3, v4):
    """Start party and return (status, static_html, dynamic_html). Both are the full scene."""
    status = _ui_start_internal_party(num_agents, language, monitor_all, v1, v2, v3, v4)
    n = int(num_agents) if num_agents else 2
    full = _ui_render_party_full(n, get_agent_states())
    return status, full, full


def _ui_stop_party_with_scene():
    """Stop party and return (status, static_html, dynamic_html). Both are the full scene."""
    status = _ui_stop_internal_party()
    full = _ui_render_party_full(0, [])
    return status, full, full


def _ui_get_party_transcript():
    """Return party transcript for the UI."""
    return get_party_transcript()


# Cache for dynamic full scene
_party_dynamic_cache = {"state_key": None, "html": None}


def _party_char_positions(n: int):
    """Return (x, y) % for each of n chars around the table."""
    import math
    radius = 38
    return [
        (50 + radius * math.cos(math.radians(i * (360 / n) - 90)),
         50 + radius * math.sin(math.radians(i * (360 / n) - 90)))
        for i in range(n)
    ]


def _ui_render_party_full(n_agents: int, states):
    """Render the FULL party scene: background, table, avatars, names, thinking, ring. Everything."""
    if n_agents == 0:
        return """
        <div class="party-scene" style="
            width: 100%; min-height: 500px; height: 500px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            border-radius: 16px; display: flex; align-items: center; justify-content: center;
            font-family: system-ui, sans-serif; color: #94a3b8;">
            <p>Start the party to see the characters.</p>
        </div>
        """
    cocktails = ["🍸", "🍹", "🥃", "🍷"]
    avatars = ["🧑‍💼", "👩‍💼", "🧑‍🎨", "👩‍🎨"]
    names = ["Sage", "Maverick", "Luna", "Cosmo"]
    positions = _party_char_positions(n_agents)
    state_by_id = {s["agent_id"]: s for s in (states or [])}
    cocktail_row = "".join(
        f'<span style="font-size: 24px; margin: 0 8px;">{c}</span>' for c in cocktails[:n_agents]
    )
    chars_html = []
    for i in range(n_agents):
        x, y = positions[i]
        aid = i + 1
        s = state_by_id.get(aid, {})
        thinking_vis = "visible" if s.get("is_thinking") else "hidden"
        border_vis = "visible" if s.get("is_speaking") else "hidden"
        chars_html.append(f"""
            <div class="party-char" data-agent-id="{aid}" style="
                position: absolute; left: {x}%; top: {y}%; transform: translate(-50%, -50%);
                text-align: center; z-index: 5;">
                <div class="thinking-bubble" style="
                    visibility: {thinking_vis}; height: 32px; display: flex; align-items: center; justify-content: center;
                    font-size: 20px; animation: think-spin 1s linear infinite;">💭</div>
                <div class="avatar" style="
                    width: 56px; height: 56px; border-radius: 50%; background: linear-gradient(145deg, #334155, #1e293b);
                    display: flex; align-items: center; justify-content: center; font-size: 28px;
                    border: 3px solid #475569; margin: 0 auto; box-sizing: border-box;
                    box-shadow: {"0 0 20px rgba(34,197,94,0.6)" if s.get("is_speaking") else "none"};
                    border-color: {"#22c55e" if s.get("is_speaking") else "#475569"};">
                    {avatars[i % len(avatars)]}
                </div>
                <div class="name" style="
                    margin-top: 6px; font-size: 12px; font-weight: 600; color: #e2e8f0;">
                    {names[i % len(names)]}
                </div>
            </div>
        """)
    return f"""
    <style>
        @keyframes think-spin {{
            0% {{ transform: scale(1) rotate(0deg); }}
            50% {{ transform: scale(1.15) rotate(5deg); }}
            100% {{ transform: scale(1) rotate(0deg); }}
        }}
    </style>
    <div class="party-scene" style="
        width: 100%; height: 500px; min-height: 500px; position: relative;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px; overflow: hidden; font-family: system-ui, sans-serif;">
        <div style="
            position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%);
            width: 55%; height: 45%; background: linear-gradient(180deg, #422006 0%, #78350f 50%, #451a03 100%);
            border-radius: 50% / 45%; box-shadow: inset 0 4px 20px rgba(0,0,0,0.4), 0 8px 32px rgba(0,0,0,0.3);
            border: 4px solid #92400e; display: flex; align-items: center; justify-content: center;">
            <div style="text-align: center;">
                <div style="margin-bottom: 8px;">{cocktail_row}</div>
                <span style="font-size: 11px; color: #a78bfa;">Cocktails</span>
            </div>
        </div>
        {''.join(chars_html)}
    </div>
    """


def _ui_render_party_dynamic():
    """Return full scene with current states. Timer updates this. Same as static, just updated."""
    states = get_agent_states()
    state_key = tuple((s["agent_id"], s["is_speaking"], s["is_thinking"]) for s in states) if states else ()
    if state_key == _party_dynamic_cache["state_key"] and _party_dynamic_cache["html"]:
        return _party_dynamic_cache["html"]
    n = len(states) if states else 0
    html = _ui_render_party_full(n, states)
    _party_dynamic_cache["state_key"] = state_key
    _party_dynamic_cache["html"] = html
    return html


def build_ui():
    inputs_list, outputs_list = get_device_lists()
    input_names = [name for _, name in inputs_list]
    output_names = [name for _, name in outputs_list]
    input_choices = [SYSTEM_DEFAULT_LABEL] + (input_names or [NO_INPUT_PLACEHOLDER])
    output_choices = [SYSTEM_DEFAULT_LABEL] + (output_names or [NO_OUTPUT_PLACEHOLDER])
    no_devices = not input_names and not output_names

    with gr.Blocks(
        title="Local Voice Chat Advanced",
        css="""
        .gradio-container { max-width: 900px !important }
        /* Party scene: static and dynamic are identical full scenes, stacked. Dynamic overlays static. */
        #party_scene_wrapper { position: relative !important; min-height: 500px !important; height: 500px !important; overflow: hidden !important; }
        #party_scene_wrapper > div { padding: 0 !important; overflow: hidden !important; }
        #party_scene_wrapper > div:first-child { height: 500px !important; overflow: hidden !important; }
        #party_scene_wrapper > div:last-child { position: absolute !important; top: 0 !important; left: 0 !important; right: 0 !important; width: 100% !important; height: 500px !important; overflow: hidden !important; }
        /*
          WebRTC mic UI (fastrtc/gradio-webrtc) can render a full-screen overlay in some deployments
          (often in Docker / different Gradio builds). That overlay blocks clicks on the rest of the app.

          We still keep the component mounted so the Start session button can trigger mic permission
          via JS, but we neutralize the overlay behavior and hide the wave UI.
        */
        [id^="webrtc_voice"] .audio-container.full-screen,
        #webrtc_voice .audio-container.full-screen,
        .audio-container.full-screen {
            position: relative !important;
            inset: auto !important;
            width: 100% !important;
            height: 1px !important;
            max-height: 1px !important;
            min-height: 0 !important;
            overflow: hidden !important;
            margin: 0 !important;
            padding: 0 !important;
            z-index: 0 !important;
        }
        /* Ensure it can never intercept clicks even if it grows. */
        [id^="webrtc_voice"] .audio-container,
        #webrtc_voice .audio-container {
            pointer-events: none !important;
        }
        /* Hide the wave + buttons (we start/stop through the app buttons). */
        [id^="webrtc_voice"] .gradio-webrtc-waveContainer,
        [id^="webrtc_voice"] .wave-container,
        [id^="webrtc_voice"] .wave-svg,
        [id^="webrtc_voice"] .button-wrap {
            display: none !important;
        }
        """,
    ) as demo:
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
                    if no_devices:
                        gr.Markdown("""
                        **No host audio devices detected** (normal in Docker). Leave as **System default** — the voice chat uses your **browser's microphone and speakers** via WebRTC.
                        
                        **If you see "Impossible d'accéder aux périphériques multimédias" / "Unable to access media devices":**
                        - Use **https://localhost:7860** (Docker enables HTTPS by default)
                        - Accept the browser's self-signed certificate warning
                        - Grant microphone permission when the browser asks
                        """)
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
                gr.Markdown("### 🎪 Party scene")
                gr.Markdown("Characters around the table. A 💭 thinking indicator appears above their head when they have the floor and are processing (LLM). Green border = speaking.")
                with gr.Column(elem_id="party_scene_wrapper"):
                    party_scene_static = gr.HTML(
                        value=_ui_render_party_full(0, []),
                        elem_id="party_scene_static",
                    )
                    party_scene_dynamic = gr.HTML(
                        value=_ui_render_party_full(0, []),
                        elem_id="party_scene_dynamic",
                    )
                party_scene_timer = gr.Timer(value=1.2)
                party_scene_timer.tick(
                    fn=_ui_render_party_dynamic,
                    inputs=[],
                    outputs=[party_scene_dynamic],
                    show_progress="hidden",
                )

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
                    # Default voices per agent
                    DEFAULT_PARTY_VOICES = ["af_river", "bf_alice", "am_adam", "am_eric"]

                    def _pick_voice(choices, preferred):
                        return preferred if preferred in choices else "Default"

                    def _party_voice_choices(lang):
                        choices = voice_choices_it if lang and "Italian" in str(lang) else voice_choices_en
                        defaults = DEFAULT_PARTY_VOICES if "Italian" not in str(lang or "") else ["Default"] * 4
                        return (
                            gr.update(choices=choices, value=_pick_voice(choices, defaults[0])),
                            gr.update(choices=choices, value=_pick_voice(choices, defaults[1])),
                            gr.update(choices=choices, value=_pick_voice(choices, defaults[2])),
                            gr.update(choices=choices, value=_pick_voice(choices, defaults[3])),
                        )

                    with gr.Row():
                        party_voice_1 = gr.Dropdown(
                            choices=voice_choices_en,
                            value=_pick_voice(voice_choices_en, "af_river"),
                            label="Agent 1 voice",
                            allow_custom_value=False,
                        )
                        party_voice_2 = gr.Dropdown(
                            choices=voice_choices_en,
                            value=_pick_voice(voice_choices_en, "bf_alice"),
                            label="Agent 2 voice",
                            allow_custom_value=False,
                        )
                        party_voice_3 = gr.Dropdown(
                            choices=voice_choices_en,
                            value=_pick_voice(voice_choices_en, "am_adam"),
                            label="Agent 3 voice",
                            allow_custom_value=False,
                        )
                        party_voice_4 = gr.Dropdown(
                            choices=voice_choices_en,
                            value=_pick_voice(voice_choices_en, "am_eric"),
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
                        fn=_ui_start_party_with_scene,
                        inputs=[
                            party_num_agents,
                            party_language,
                            party_monitor,
                            party_voice_1,
                            party_voice_2,
                            party_voice_3,
                            party_voice_4,
                        ],
                        outputs=[party_status, party_scene_static, party_scene_dynamic],
                    )
                    stop_party_btn.click(
                        fn=_ui_stop_party_with_scene,
                        inputs=[],
                        outputs=[party_status, party_scene_static, party_scene_dynamic],
                    )

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
    port = getattr(args, "port", None) or 7860
    user_set_port = getattr(args, "port", None) is not None

    # HTTPS for browser mic access when not on localhost (e.g. Docker, remote access)
    use_ssl = os.environ.get("USE_SSL", "").lower() in ("1", "true", "yes")
    ssl_keyfile = ssl_certfile = None
    if use_ssl:
        import subprocess
        import tempfile
        cert_dir = tempfile.mkdtemp(prefix="gradio_ssl_")
        ssl_keyfile = os.path.join(cert_dir, "key.pem")
        ssl_certfile = os.path.join(cert_dir, "cert.pem")
        try:
            subprocess.run(
                [
                    "openssl", "req", "-x509", "-newkey", "rsa:2048",
                    "-keyout", ssl_keyfile, "-out", ssl_certfile,
                    "-days", "365", "-nodes", "-subj", "/CN=localhost",
                ],
                check=True,
                capture_output=True,
            )
            logger.info("Generated self-signed SSL cert for HTTPS (browser mic access)")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning("SSL generation failed, falling back to HTTP: %s", e)
            ssl_keyfile = ssl_certfile = None
            use_ssl = False

    try:
        scheme = "https" if use_ssl else "http"
        url = f"{scheme}://127.0.0.1:{port}"
        print(f"\n  Running on local URL:  {url}\n")
        if use_ssl:
            print("  Use HTTPS for browser microphone access (accept self-signed cert warning)\n")
        logger.info("Launching with Gradio UI (config + voice chat) on port %s...", port)
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
        )
    except OSError as e:
        if ("port" in str(e).lower() or "address already in use" in str(e).lower()) and not user_set_port:
            for p in range(port + 1, 7871):
                try:
                    scheme = "https" if use_ssl else "http"
                    url = f"{scheme}://127.0.0.1:{p}"
                    print(f"\n  Running on local URL:  {url}\n")
                    demo.launch(
                        server_name="0.0.0.0",
                        server_port=p,
                        ssl_keyfile=ssl_keyfile,
                        ssl_certfile=ssl_certfile,
                    )
                except OSError:
                    continue
                break
            else:
                raise RuntimeError(
                    f"No free port in {port}-7870. Use --port <number>."
                ) from e
        raise
    finally:
        audio_monitor_stop.set()
