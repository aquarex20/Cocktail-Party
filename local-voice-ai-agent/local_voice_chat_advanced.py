import sys
import argparse
import time

from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model
from loguru import logger
from ollama import chat
import requests
import json

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
# Timing thresholds
QUESTION_TIMEOUT_SEC = 4.0     # Wait this long for human to answer
SILENCE_TIMEOUT_SEC = 10.0     # If silence this long, AI can speak
MIN_TURNS_BEFORE_AI = 2        # AI waits at least 2 human turns before considering response

# Conversation state tracking
conversation_state = {
    "last_question_time": None,      # When was the last question asked
    "last_question_text": None,      # The question text
    "question_answered": False,      # Was it answered by a human?
    "last_utterance_time": None,     # When did anyone last speak
    "turn_count": 0,                 # Number of turns since last AI response
}


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


def should_ai_respond(current_time: float) -> bool:
    """Determine if AI should respond based on conversation state."""

    # Rule 1: Wait for minimum turns before AI can respond
    if conversation_state["turn_count"] < MIN_TURNS_BEFORE_AI:
        logger.debug(f"AI waiting: only {conversation_state['turn_count']} turns (need {MIN_TURNS_BEFORE_AI})")
        return False

    # Rule 2: If there's a pending unanswered question
    if (conversation_state["last_question_time"] and
        not conversation_state["question_answered"]):

        time_since_question = current_time - conversation_state["last_question_time"]

        # Wait for timeout before answering
        if time_since_question >= QUESTION_TIMEOUT_SEC:
            logger.debug(f"Unanswered question for {time_since_question:.1f}s - AI will respond")
            return True
        else:
            logger.debug(f"Question pending, waiting for human answer ({time_since_question:.1f}s < {QUESTION_TIMEOUT_SEC}s)")
            return False

    # Rule 3: Long silence = opportunity (checked when no pending question)
    if conversation_state["last_utterance_time"]:
        silence_duration = current_time - conversation_state["last_utterance_time"]
        if silence_duration >= SILENCE_TIMEOUT_SEC:
            logger.debug(f"Long silence ({silence_duration:.1f}s) - AI can intervene")
            return True

    # Default: Don't respond if it's just a statement and no special conditions
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

    transcript = stt_model.stt(audio)
    current_time = time.time()

    logger.debug(f"🎤 Transcript: {transcript}")

    # Update conversation history
    conversation += f"\nUser: {transcript}"
    conversation_state["last_utterance_time"] = current_time
    conversation_state["turn_count"] += 1

    # === Analyze the transcript ===

    # Check if this is a question
    if is_question(transcript):
        conversation_state["last_question_time"] = current_time
        conversation_state["last_question_text"] = transcript
        conversation_state["question_answered"] = False
        logger.debug(f"❓ Question detected: {transcript}")

    # Check if this answers a pending question
    elif conversation_state["last_question_text"]:
        if is_likely_answer(transcript, conversation_state["last_question_text"]):
            conversation_state["question_answered"] = True
            logger.debug(f"✅ Question answered by human: {transcript}")
            # Clear the pending question since it was answered
            conversation_state["last_question_text"] = None

    # === Decide if AI should respond ===
    if not should_ai_respond(current_time):
        logger.debug("🤐 AI staying silent - humans are conversing")
        return  # Yield nothing, stay silent

    # === AI responds ===
    logger.debug("🧠 AI responding - opportunity detected!")

    # Reset state for AI's turn
    conversation_state["turn_count"] = 0
    conversation_state["last_question_text"] = None
    conversation_state["question_answered"] = False

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
