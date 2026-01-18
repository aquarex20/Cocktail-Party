import sys
import os
import argparse
import threading

from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model
from loguru import logger
import gradio as gr
import numpy as np
import time as time_module  # rename to avoid conflict with callback's 'time' param
import sounddevice as sd
from utilities import extract_transcript, extract_last_replies, back_and_forth

from llm_client import stream_llm_response, get_llm_response

# Docker mode detection - disables direct audio device access in containers
DOCKER_MODE = os.getenv("DOCKER_CONTAINER", "").lower() == "true"

# Global stop event for the audio monitor thread
audio_monitor_stop = threading.Event()

# Set default audio device only when running locally (not in Docker)
# Comment this if you want to use your own microphone with your own party
if not DOCKER_MODE:
    sd.default.device = ("BlackHole 2ch", 1)  # (input, output)

stt_model = get_stt_model()  # moonshine/base
tts_model = get_tts_model()  # kokoro

# Track the current selected input device (None means use default)
selected_input_device = None
audio_monitor_thread = None

def get_input_devices():
    """Returns list of available input (microphone) devices."""
    devices = sd.query_devices()
    input_devices = []
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            input_devices.append(f"{i}: {d['name']}")
    return input_devices

def get_output_devices():
    """Returns list of available output (speaker) devices."""
    devices = sd.query_devices()
    output_devices = []
    for i, d in enumerate(devices):
        if d['max_output_channels'] > 0:
            output_devices.append(f"{i}: {d['name']}")
    return output_devices

def on_input_device_change(device_str):
    """Handle microphone device selection change."""
    global selected_input_device, audio_monitor_thread
    if not device_str:
        return "No device selected"

    device_id = int(device_str.split(":")[0])
    selected_input_device = device_id

    # Update sounddevice default input
    current_output = sd.default.device[1] if isinstance(sd.default.device, tuple) else sd.default.device
    sd.default.device = (device_id, current_output)

    # Restart audio monitor with new device
    audio_monitor_stop.set()
    time_module.sleep(0.2)  # Give time for the old monitor to stop
    audio_monitor_stop.clear()
    audio_monitor_thread = start_audio_monitor(device_id)

    logger.info(f"🎤 Input device changed to: {device_str}")
    return f"Input device set to: {device_str}"

def on_output_device_change(device_str):
    """Handle speaker device selection change."""
    if not device_str:
        return "No device selected"

    device_id = int(device_str.split(":")[0])

    # Update sounddevice default output
    current_input = sd.default.device[0] if isinstance(sd.default.device, tuple) else sd.default.device
    sd.default.device = (current_input, device_id)

    logger.info(f"🔊 Output device changed to: {device_str}")
    return f"Output device set to: {device_str}"

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
full_send_it = False
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
    
    # 1. Stream text from LLM as it's generated
    for chunk in stream_llm_response(context, alone=alone, is_back_and_forth=is_back_and_forth):
        text_buffer += chunk
        ai_reply+=chunk
        if someone_talking:
            break
        # Simple heuristic: speak when we see end of sentence or buffer big enough
        if any(p in text_buffer for p in [".", "!", "?"]) or (len(text_buffer) > 30 and text_buffer[-1]==","):
            speak_part = text_buffer
            text_buffer = ""
            if someone_talking:
                break

            logger.debug(f"🗣️ TTS on chunk: {speak_part!r}")
            # 2. Stream TTS for that chunk
            for audio_chunk in tts_model.stream_tts_sync(speak_part):
                if someone_talking:
                    break
                yield audio_chunk

    # 3. Flush any remaining text once LLM is done
    text_buffer = text_buffer.strip()
    if someone_talking:
        return False

    if text_buffer:
        ai_reply+=text_buffer
        logger.debug(f"🗣️ TTS on final chunk: {text_buffer!r}")
        for audio_chunk in tts_model.stream_tts_sync(text_buffer):
            if someone_talking:
                break
            yield audio_chunk
    conversation+="\n"+ai_reply



def echo(audio):
    global conversation, someone_talking, last_voice_detected, full_send_it
    if full_send_it:
        full_send_it = False
        yield from talk()
    transcript = stt_model.stt(audio)
    logger.debug(f"🎤 Transcript: {transcript}")
    conversation+="\nUser:"+transcript
    yield from talk()

def create_stream():
    return Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")


# Flag to prevent multiple talk_direct calls
ai_is_speaking = False

def make_summary(): 
    global conversation, summary
    summary= get_llm_response(summary+conversation, summarize=True)
    print("summary generated is "+ summary)
    conversation = "\n".join(extract_last_replies(conversation, 10))

def audio_callback(indata, frames, time, status):
    global someone_talking, last_voice_detected, strikes, full_send_it, ai_is_speaking, conversation, last_summary_time
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
        if time_module.time() - last_summary_time > 15 and conversation!="\nTranscript:\n ": #replace to 10 instead of 5
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
            # FastRTC returns tuples as (sample_rate, audio_data)
            sample_rate, audio_data = audio_chunk
            # Convert to float32 for sounddevice and ensure 1D
            audio_data = np.asarray(audio_data, dtype=np.float32).flatten()
            if audio_data.size > 0:
                sd.play(audio_data, samplerate=int(sample_rate), blocking=True)
        logger.debug("🎙️ Finished proactive speech")
    finally:
        ai_is_speaking = False
        last_voice_detected = time_module.time()  # Reset timer after speaking

def start_audio_monitor(device=None):
    """Run audio monitoring in a separate thread."""
    def monitor_loop():
        with sd.InputStream(samplerate=16000, channels=1, callback=audio_callback, device=device):
            logger.info(f"🎧 Audio monitor started (device={device})...")
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


def create_ui_with_settings(stream):
    """Create Gradio UI with settings accordion for device selection."""
    input_devices = get_input_devices()
    output_devices = get_output_devices()

    with gr.Blocks() as demo:
        with gr.Accordion("Settings", open=False):
            with gr.Row():
                mic_dropdown = gr.Dropdown(
                    choices=input_devices,
                    label="Microphone",
                    value=input_devices[0] if input_devices else None,
                    interactive=True
                )
                speaker_dropdown = gr.Dropdown(
                    choices=output_devices,
                    label="Speaker",
                    value=output_devices[0] if output_devices else None,
                    interactive=True
                )
            with gr.Row():
                mic_status = gr.Textbox(label="Mic Status", interactive=False)
                speaker_status = gr.Textbox(label="Speaker Status", interactive=False)

            mic_dropdown.change(
                fn=on_input_device_change,
                inputs=[mic_dropdown],
                outputs=[mic_status]
            )
            speaker_dropdown.change(
                fn=on_output_device_change,
                inputs=[speaker_dropdown],
                outputs=[speaker_status]
            )

        # Embed the FastRTC WebRTC stream UI
        stream.ui.render()

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Voice Chat Advanced")
    parser.add_argument(
        "--phone",
        action="store_true",
        help="Launch with FastRTC phone interface (get a temp phone number)",
    )
    args = parser.parse_args()

    # Start audio monitor and printer only when running locally (not in Docker)
    # In Docker, WebRTC handles all audio via browser - no direct device access needed
    if not DOCKER_MODE:
        audio_monitor_thread = start_audio_monitor()
        printer_thread = start_conversation_printer()
    else:
        logger.info("Running in Docker mode - audio monitor disabled (WebRTC handles audio)")

    stream = create_stream()

    try:
        if args.phone:
            logger.info("Launching with FastRTC phone interface...")
            stream.fastphone()
        else:
            logger.info("Launching with Gradio UI (with settings)...")
            custom_ui = create_ui_with_settings(stream)
            # SSL configuration for HTTPS (required for microphone access)
            ssl_certfile = os.getenv("GRADIO_SSL_CERTFILE")
            ssl_keyfile = os.getenv("GRADIO_SSL_KEYFILE")
            if ssl_certfile and ssl_keyfile and os.path.exists(ssl_certfile) and os.path.exists(ssl_keyfile):
                logger.info(f"Launching with SSL: {ssl_certfile}")
                custom_ui.launch(ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile, ssl_verify=False)
            else:
                custom_ui.launch()
    finally:
        # Cleanup when done
        audio_monitor_stop.set()
