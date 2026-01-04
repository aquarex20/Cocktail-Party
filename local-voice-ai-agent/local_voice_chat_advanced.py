import sys
import argparse
import time
import threading
import queue

from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model
from loguru import logger
from ollama import chat
import requests
import json
import numpy as np

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
import sounddevice as sd

#comment this if you want to use your own microphone with your own party
sd.default.device = ("BlackHole 2ch", 1)  # (input, output)

stt_model = get_stt_model()  # moonshine/base
tts_model = get_tts_model()  # kokoro

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")
conversation="\nTranscript:\n "

# === Smart AI Intervention Settings ===
# Timing thresholds (humans typically answer within 1-2 seconds)
QUESTION_TIMEOUT_SEC = 2.0         # Wait this long for human to answer
POST_AI_TIMEOUT_SEC = 3.0          # Longer timeout after AI just spoke (let other humans answer)
SILENCE_TIMEOUT_SEC = 8.0          # If silence this long, AI can speak
AI_COOLDOWN_SEC = 5.0              # After AI speaks, wait this long before considering new questions
MAX_QUESTION_AGE_SEC = 30.0        # Questions older than this are stale and should be discarded

# Conversation state tracking
conversation_state = {
    # Question tracking
    "pending_question": None,        # The current unanswered question text
    "pending_question_time": None,   # When it was asked

    # AI tracking
    "ai_last_response_time": None,   # When AI last spoke
    "ai_is_speaking": False,         # True when AI is playing TTS (prevents feedback loop)

    # Human tracking
    "multi_human_detected": False,   # Did we detect multiple humans?
    "last_utterance_time": None,     # When did anyone last speak
    "turn_count": 0,                 # Number of turns since last AI response
    "user_last_speech_time": None,   # When user last spoke (for don't-interrupt feature)
    "ai_finished_speaking_time": None,  # When AI finished speaking (for post-speaking cooldown)
}

# === Background Timer Thread State ===
timer_thread = None
timer_stop_event = threading.Event()
timer_lock = threading.Lock()  # Prevent race conditions


def is_question(text: str) -> bool:
    """Detect if text is a question using simple heuristics."""
    text = text.strip().lower()

    # Check for question mark
    if text.endswith("?"):
        return True

    # Check for question words at start
    question_starters = [
        "who ", "what ", "where ", "when ", "why ", "how ",
        "is ", "are ", "do ", "does ", "did ", "can ", "could ",
        "would ", "should ", "will ", "have ", "has "
    ]
    return any(text.startswith(q) for q in question_starters)


def get_timeout() -> float:
    """Get appropriate timeout based on conversation context."""
    # Group mode = longer timeout (let other humans answer)
    if conversation_state["multi_human_detected"]:
        return POST_AI_TIMEOUT_SEC  # 3s for group

    # AI recently spoke = longer timeout (give humans a chance)
    if conversation_state["ai_last_response_time"]:
        time_since_ai = time.time() - conversation_state["ai_last_response_time"]
        if time_since_ai < AI_COOLDOWN_SEC:
            return POST_AI_TIMEOUT_SEC  # 3s during cooldown

    # Solo mode = faster response
    return QUESTION_TIMEOUT_SEC  # 2s for solo


def start_question_timer():
    """Start background thread to monitor question timeout."""
    global timer_thread, timer_stop_event

    with timer_lock:
        # If timer is already running, don't start another
        if timer_thread and timer_thread.is_alive():
            logger.debug("⏱️ Timer already running")
            return

        # Clear the stop event and start new timer
        timer_stop_event.clear()
        timer_thread = threading.Thread(target=question_timeout_monitor, daemon=True)
        timer_thread.start()
        logger.debug("⏱️ Timer started!")


def question_timeout_monitor():
    """Background thread that checks for unanswered questions."""
    global conversation, conversation_state

    while not timer_stop_event.is_set():
        time.sleep(0.5)  # Check every 500ms

        with timer_lock:
            if conversation_state["pending_question"] and conversation_state["pending_question_time"]:
                time_since_question = time.time() - conversation_state["pending_question_time"]
                timeout = get_timeout()

                if time_since_question >= timeout:
                    logger.debug(f"⏰ Timer: Question timeout ({time_since_question:.1f}s >= {timeout}s) - triggering response!")
                    trigger_ai_response()
                    return  # Exit after triggering
            else:
                # No pending question, exit timer
                logger.debug("⏱️ Timer: No pending question, stopping")
                return

    logger.debug("⏱️ Timer stopped by event")


# Cache the working output device
_cached_output_device = None


def find_mac_speakers():
    """Find MacBook Air Speakers or similar built-in output device."""
    try:
        devices = sd.query_devices()
        # Priority order for finding speakers
        speaker_keywords = ['MacBook', 'Built-in', 'Speaker', 'Internal']

        for keyword in speaker_keywords:
            for i, dev in enumerate(devices):
                if keyword.lower() in dev['name'].lower() and dev['max_output_channels'] >= 1:
                    logger.info(f"🔊 Found speaker device {i}: {dev['name']}")
                    return i

        # Fallback: return first device with output channels that's not a virtual device
        for i, dev in enumerate(devices):
            if dev['max_output_channels'] >= 1 and 'BlackHole' not in dev['name']:
                return i
    except Exception as e:
        logger.error(f"❌ Error finding speakers: {e}")
    return None


def play_audio_chunk(audio_chunk):
    """Play an audio chunk, handling mono/stereo conversion if needed."""
    global _cached_output_device

    sample_rate, audio_data = audio_chunk

    # Ensure audio_data is a numpy array with float32 dtype
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.array(audio_data, dtype=np.float32)
    else:
        audio_data = audio_data.astype(np.float32)

    # If audio is 1D (mono), convert to stereo by duplicating channel
    if audio_data.ndim == 1:
        audio_data = np.column_stack([audio_data, audio_data])  # Mono to stereo

    # If audio is 2D but only 1 channel, duplicate to stereo
    elif audio_data.ndim == 2 and audio_data.shape[1] == 1:
        audio_data = np.column_stack([audio_data[:, 0], audio_data[:, 0]])

    # Find or use cached output device
    if _cached_output_device is None:
        _cached_output_device = find_mac_speakers()

    output_device = _cached_output_device
    logger.debug(f"🔊 Playing to device {output_device}, shape: {audio_data.shape}")

    try:
        sd.play(audio_data, sample_rate, device=output_device)
        sd.wait()
    except sd.PortAudioError as e:
        logger.warning(f"⚠️ Audio error with device {output_device}: {e}")
        # Reset cache and try to find another device
        _cached_output_device = None
        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if dev['max_output_channels'] >= 1 and 'BlackHole' not in dev['name']:
                    logger.debug(f"🔊 Trying device {i}: {dev['name']}")
                    try:
                        sd.play(audio_data, sample_rate, device=i)
                        sd.wait()
                        _cached_output_device = i
                        logger.info(f"✅ Now using device {i}: {dev['name']}")
                        return
                    except:
                        continue
            logger.error("❌ No working audio device found")
        except Exception as e2:
            logger.error(f"❌ Audio playback failed: {e2}")


def wait_for_user_to_stop():
    """Wait until user stops speaking before AI responds (don't interrupt feature)."""
    USER_SILENCE_THRESHOLD = 1.0  # Wait 1 second of silence before speaking

    while True:
        if conversation_state["user_last_speech_time"]:
            time_since_speech = time.time() - conversation_state["user_last_speech_time"]
            if time_since_speech < USER_SILENCE_THRESHOLD:
                logger.debug(f"⏳ Waiting for user to finish speaking ({time_since_speech:.1f}s < {USER_SILENCE_THRESHOLD}s)")
                time.sleep(0.3)
                continue
        break  # User is silent, proceed


def audio_player(audio_queue):
    """Consumer thread: plays audio chunks continuously from queue."""
    while True:
        try:
            chunk = audio_queue.get(timeout=5.0)  # 5 second timeout
            if chunk is None:  # Sentinel value signals end
                break
            play_audio_chunk(chunk)
        except queue.Empty:
            logger.debug("⏰ Audio queue timeout, stopping player")
            break


def trigger_ai_response():
    """Generate and play AI response from background thread."""
    global conversation, conversation_state

    # Feature 1: Don't interrupt user - wait for them to stop speaking
    wait_for_user_to_stop()

    current_time = time.time()

    # Reset state
    conversation_state["pending_question"] = None
    conversation_state["pending_question_time"] = None
    conversation_state["ai_last_response_time"] = current_time
    conversation_state["turn_count"] = 0

    # Set speaking flag to prevent feedback loop (AI hearing itself)
    conversation_state["ai_is_speaking"] = True
    logger.debug("🧠 AI responding (from timer)!")

    try:
        text_buffer = ""
        ai_reply = "AI:"
        audio_chunks_buffer = []  # Buffer audio for smoother playback

        # Stream LLM → TTS → buffer audio chunks
        for chunk in stream_llm_response(conversation):
            text_buffer += chunk
            ai_reply += chunk

            if any(p in text_buffer for p in [".", "!", "?"]) or (len(text_buffer) > 80 and text_buffer.endswith(",")):
                speak_part = text_buffer
                text_buffer = ""

                logger.debug(f"🗣️ TTS on chunk (timer): {speak_part!r}")
                # Collect audio chunks
                for audio_chunk in tts_model.stream_tts_sync(speak_part):
                    audio_chunks_buffer.append(audio_chunk)

                # Play buffered chunks (play while generating more)
                while audio_chunks_buffer:
                    play_audio_chunk(audio_chunks_buffer.pop(0))

        # Flush remaining text
        text_buffer = text_buffer.strip()
        if text_buffer:
            ai_reply += text_buffer
            logger.debug(f"🗣️ TTS on final chunk (timer): {text_buffer!r}")
            for audio_chunk in tts_model.stream_tts_sync(text_buffer):
                play_audio_chunk(audio_chunk)

        conversation += "\n" + ai_reply
        logger.debug("🧠 AI response complete (from timer)")

    finally:
        # Always clear speaking flag when done
        conversation_state["ai_is_speaking"] = False
        conversation_state["ai_finished_speaking_time"] = time.time()  # For post-speaking cooldown
        logger.debug("🔇 AI finished speaking")


def is_likely_answer(text: str, question: str) -> bool:
    """Detect if text is likely an answer to the pending question."""
    text = text.strip().lower()
    word_count = len(text.split())

    # Answer starters that indicate a direct response
    answer_starters = [
        "yes", "no", "yeah", "nope", "sure", "definitely",
        "probably", "maybe", "i think", "i believe", "well",
        "it's", "its", "that's", "he", "she", "they", "we"
    ]

    # If it starts with an answer word and isn't a question
    if any(text.startswith(a) for a in answer_starters) and not text.endswith("?"):
        return True

    # If it's a statement (no question mark) with reasonable length
    if not text.endswith("?") and word_count >= 3:
        return True

    return False


def should_ai_respond(current_time: float, time_since_question: float = None) -> bool:
    """Determine if AI should respond based on conversation state."""

    # Check if AI is in cooldown (just responded recently)
    ai_in_cooldown = False
    if conversation_state["ai_last_response_time"]:
        time_since_ai_spoke = current_time - conversation_state["ai_last_response_time"]
        ai_in_cooldown = time_since_ai_spoke < AI_COOLDOWN_SEC
        if ai_in_cooldown:
            logger.debug(f"⏸️ AI cooldown: {time_since_ai_spoke:.1f}s < {AI_COOLDOWN_SEC}s")

    # Determine timeout based on context
    if ai_in_cooldown or conversation_state["multi_human_detected"]:
        timeout = POST_AI_TIMEOUT_SEC  # 3s - be more patient
    else:
        timeout = QUESTION_TIMEOUT_SEC  # 2s - respond faster in solo mode

    # Rule 1: If there's a pending unanswered question
    if conversation_state["pending_question"] and time_since_question is not None:
        if time_since_question >= timeout:
            logger.debug(f"⏰ Unanswered question for {time_since_question:.1f}s (>= {timeout}s) - AI will respond")
            return True
        else:
            logger.debug(f"Question pending, waiting ({time_since_question:.1f}s < {timeout}s)")
            return False

    # Rule 2: Long silence = opportunity (no pending question)
    if conversation_state["last_utterance_time"]:
        silence_duration = current_time - conversation_state["last_utterance_time"]
        if silence_duration >= SILENCE_TIMEOUT_SEC:
            logger.debug(f"Long silence ({silence_duration:.1f}s) - AI can intervene")
            return True

    # Default: Don't respond
    logger.debug("No intervention opportunity - AI staying silent")
    return False


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
    global conversation, conversation_state

    # Skip if AI is currently speaking (prevents feedback loop where AI hears itself)
    if conversation_state["ai_is_speaking"]:
        logger.debug("🔇 Skipping - AI is speaking (preventing feedback)")
        return

    # Skip if within post-speaking cooldown (audio takes ~600ms to be transcribed)
    POST_SPEAKING_COOLDOWN = 1.5  # seconds
    if conversation_state["ai_finished_speaking_time"]:
        time_since_ai_finished = time.time() - conversation_state["ai_finished_speaking_time"]
        if time_since_ai_finished < POST_SPEAKING_COOLDOWN:
            logger.debug(f"🔇 Skipping - post-speaking cooldown ({time_since_ai_finished:.1f}s < {POST_SPEAKING_COOLDOWN}s)")
            return

    transcript = stt_model.stt(audio)
    current_time = time.time()

    # Calculate time since last question FIRST (needed for empty transcript check)
    time_since_question = None
    if conversation_state["pending_question_time"]:
        time_since_question = current_time - conversation_state["pending_question_time"]

    # Check for stale questions (older than MAX_QUESTION_AGE_SEC)
    if time_since_question is not None and time_since_question > MAX_QUESTION_AGE_SEC:
        logger.debug(f"❌ Question too old ({time_since_question:.1f}s > {MAX_QUESTION_AGE_SEC}s) - discarding")
        timer_stop_event.set()  # Stop any running timer
        conversation_state["pending_question"] = None
        conversation_state["pending_question_time"] = None
        time_since_question = None

    # Skip empty transcripts - background timer handles timeout responses
    if not transcript or not transcript.strip():
        logger.debug("🔇 Empty transcript, skipping")
        return

    logger.debug(f"🎤 Transcript: {transcript}")

    # Update conversation history
    conversation += f"\nUser: {transcript}"
    conversation_state["last_utterance_time"] = current_time
    conversation_state["user_last_speech_time"] = current_time  # Track for don't-interrupt feature
    conversation_state["turn_count"] += 1

    # === Analyze the transcript ===

    # Check if this answers a pending question FIRST
    if conversation_state["pending_question"] and time_since_question is not None:
        if is_likely_answer(transcript, conversation_state["pending_question"]):
            logger.debug(f"✅ Question answered by human in {time_since_question:.1f}s: {transcript}")

            # Detect multiple humans: if answered quickly by non-AI
            if time_since_question < POST_AI_TIMEOUT_SEC:
                conversation_state["multi_human_detected"] = True
                logger.debug("👥 Multiple humans detected! (quick answer)")

            # Clear the pending question and stop timer
            timer_stop_event.set()  # Stop the background timer
            conversation_state["pending_question"] = None
            conversation_state["pending_question_time"] = None
            return  # Don't let AI respond, humans are conversing

    # Check if this is a NEW question
    if is_question(transcript):
        # If there's already a pending question that's old enough, we'll answer it
        if conversation_state["pending_question"] and time_since_question is not None:
            logger.debug(f"❓ New question, but pending one from {time_since_question:.1f}s ago")
            # Keep the OLD question's timestamp for timing check
        else:
            # Start new question timer
            conversation_state["pending_question"] = transcript
            conversation_state["pending_question_time"] = current_time
            logger.debug(f"❓ Question detected: {transcript}")

            # Start background timer to check for timeout
            start_question_timer()

    # === Decide if AI should respond ===
    if not should_ai_respond(current_time, time_since_question):
        logger.debug("🤐 AI staying silent")
        return  # Yield nothing, stay silent

    # === AI responds ===
    timer_stop_event.set()  # Stop the background timer
    logger.debug("🧠 AI responding!")

    # Reset ALL state when AI responds
    conversation_state["pending_question"] = None
    conversation_state["pending_question_time"] = None  # THIS WAS MISSING!
    conversation_state["ai_last_response_time"] = current_time  # Track when AI spoke
    conversation_state["turn_count"] = 0

    text_buffer = ""
    ai_reply = "AI:"

    # 1. Stream text from LLM as it's generated
    for chunk in stream_llm_response(conversation):
        text_buffer += chunk
        ai_reply += chunk

        # Simple heuristic: speak when we see end of sentence or buffer big enough
        if any(p in text_buffer for p in [".", "!", "?"]) or (len(text_buffer) > 80 and text_buffer[-1] == ","):
            speak_part = text_buffer
            text_buffer = ""

            logger.debug(f"🗣️ TTS on chunk: {speak_part!r}")
            # 2. Stream TTS for that chunk
            for audio_chunk in tts_model.stream_tts_sync(speak_part):
                yield audio_chunk

    # 3. Flush any remaining text once LLM is done
    text_buffer = text_buffer.strip()
    if text_buffer:
        ai_reply += text_buffer
        logger.debug(f"🗣️ TTS on final chunk: {text_buffer!r}")
        for audio_chunk in tts_model.stream_tts_sync(text_buffer):
            yield audio_chunk

    conversation += "\n" + ai_reply


def create_stream():
    return Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Voice Chat Advanced")
    parser.add_argument(
        "--phone",
        action="store_true",
        help="Launch with FastRTC phone interface (get a temp phone number)",
    )
    args = parser.parse_args()

    stream = create_stream()

    if args.phone:
        logger.info("Launching with FastRTC phone interface...")
        stream.fastphone()
    else:
        logger.info("Launching with Gradio UI...")
        stream.ui.launch()
