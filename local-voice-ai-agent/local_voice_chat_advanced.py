import sys
import argparse
import threading

from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model
from loguru import logger
import numpy as np
import time as time_module  # rename to avoid conflict with callback's 'time' param
import sounddevice as sd
from utilities import extract_transcript, extract_last_replies

from llm_client import stream_llm_response, get_llm_response

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
    alone= all(r.startswith("AI:") for r in extract_last_replies(conversation, 2))
    # 1. Stream text from LLM as it's generated
    for chunk in stream_llm_response(summary+conversation if not alone else "\n".join(extract_last_replies(conversation, 2)), alone=alone):
        text_buffer += chunk
        ai_reply+=chunk
        if someone_talking:
            break
        # Simple heuristic: speak when we see end of sentence or buffer big enough
        if any(p in text_buffer for p in [".", "!", "?"]) or (len(text_buffer) > 80 and text_buffer[-1]==","):
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
    
    # Start conversation printer in background thread
    printer_thread = start_conversation_printer()

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
