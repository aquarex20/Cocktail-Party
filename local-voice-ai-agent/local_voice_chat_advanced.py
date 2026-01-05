import sys
import argparse
import threading

from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model
from loguru import logger
from ollama import chat
import requests
import json
import numpy as np
import time as time_module  # rename to avoid conflict with callback's 'time' param


OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
import sounddevice as sd

# Global stop event for the audio monitor thread
audio_monitor_stop = threading.Event()

#comment this if you want to use your own microphone with your own party
sd.default.device = ("BlackHole 2ch", 1)  # (input, output)

stt_model = get_stt_model()  # moonshine/base
tts_model = get_tts_model()  # kokoro

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")
conversation="\nTranscript:\n "
someone_talking = False
last_voice_detected=0.0
def stream_llm_response(transcript: str):
    """
    [Unverified] Streams text chunks from Ollama /api/chat with stream=true.
    Yields small pieces of text as they come.
    """
    payload = {
        "model": "gemma3:4b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a LLM in a WebRTC call simulationg a Cocktail Party. Your goal is to "
                    "be chill and answer in a cool way. the "
                    "output will be converted to audio so don't include emojis "
                    "or special characters in your answers. Respond to what the "
                    "user said in a creative and helpful way base yourself off of the conversation transcript in which AI represents you, User represents the User you have to reply to. DONT ANSWER WITH AI, directly speak what you need to speak. "
                ),
            },
            {"role": "user", "content": transcript},
        ],
        "options": {"num_predict": 150},
        "stream": True,
    }

    with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            # [Unverified] Streaming format – adjust if your actual JSON differs
            chunk = ""
            if "message" in data and "content" in data["message"]:
                chunk = data["message"]["content"].replace("*","")
            elif "delta" in data:
                chunk = data["delta"].replace("*","")

            if chunk:
                yield chunk

def echo(audio):
    global conversation, someone_talking, last_voice_detected
    transcript = stt_model.stt(audio)
    logger.debug(f"🎤 Transcript: {transcript}")
    conversation+="\nUser:"+transcript
    logger.debug("🧠 Starting streamed LLM response...")
    text_buffer = ""
    ai_reply="AI:"
    # 1. Stream text from LLM as it’s generated
    for chunk in stream_llm_response(conversation):
        text_buffer += chunk
        ai_reply+=chunk
        if someone_talking:

            return
        # Simple heuristic: speak when we see end of sentence or buffer big enough
        if any(p in text_buffer for p in [".", "!", "?"]) or (len(text_buffer) > 80 and text_buffer[-1]==","):
            speak_part = text_buffer
            text_buffer = ""

            logger.debug(f"🗣️ TTS on chunk: {speak_part!r}")
            # 2. Stream TTS for that chunk
            for audio_chunk in tts_model.stream_tts_sync(speak_part):
                yield audio_chunk

    # 3. Flush any remaining text once LLM is done
    text_buffer = text_buffer.strip()
    if someone_talking:
        return

    if text_buffer:
        ai_reply+=text_buffer
        logger.debug(f"🗣️ TTS on final chunk: {text_buffer!r}")
        for audio_chunk in tts_model.stream_tts_sync(text_buffer):
            if someone_talking:
                return
            yield audio_chunk
    conversation+="\n"+ai_reply


def create_stream():
    return Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")


def audio_callback(indata, frames, time, status):
    global someone_talking, last_voice_detected
    """Process each audio chunk as it arrives."""
    if status:
        print(f"Status: {status}")
    
    audio_chunk = indata[:, 0]  # Get mono channel
    volume = np.sqrt(np.mean(audio_chunk**2))
    #print(f"Volume: {volume:.4f}")
    if volume >= 0.0005:
        someone_talking = True
        last_voice_detected = time_module.time()
    elif time_module.time() - last_voice_detected > 3:  # 2 second cooldown
        someone_talking = False

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Voice Chat Advanced")
    parser.add_argument(
        "--phone",
        action="store_true",
        help="Launch with FastRTC phone interface (get a temp phone number)",
    )
    args = parser.parse_args()

    # Start audio monitor in background thread
    monitor_thread = start_audio_monitor()

    stream = create_stream()

    try:
        if args.phone:
            logger.info("Launching with FastRTC phone interface...")
            stream.fastphone()
        else:
            logger.info("Launching with Gradio UI...")
            stream.ui.launch()
    finally:
        # Cleanup when done
        audio_monitor_stop.set()
