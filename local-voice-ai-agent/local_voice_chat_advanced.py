import sys
import argparse
import os

from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model
from loguru import logger
from ollama import chat
import requests
import json
import numpy as np

# Add parent directory to path for speaker_diarization module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from speaker_diarization import SpeakerDiarizer

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

stt_model = get_stt_model()  # moonshine/base
tts_model = get_tts_model()  # kokoro

# Initialize speaker diarization
logger.info("Initializing speaker diarization...")
diarizer = SpeakerDiarizer(
    similarity_threshold=0.35,  # Lower = more lenient matching (same speaker recognized)
    max_speakers=10,            # For short audio, threshold is reduced by 30% automatically
    device="cpu"  # Mac uses CPU (MPS has issues with SpeechBrain)
)
logger.info(f"Speaker diarization ready: {diarizer.is_ready()}")

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")
conversation="\nTranscript:\n "

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
    global conversation

    # Unpack audio tuple (sample_rate, audio_array)
    sample_rate, audio_array = audio

    # Debug audio format
    logger.debug(f"Audio: sr={sample_rate}, shape={audio_array.shape}, dtype={audio_array.dtype}")
    logger.debug(f"Diarizer ready: {diarizer.is_ready()}")

    # Identify speaker using diarization
    try:
        # Flatten audio if needed (fastrtc may return 2D array)
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()

        speaker_id, confidence, is_new = diarizer.identify_speaker_with_confidence(audio_array, sample_rate)
        if is_new:
            logger.debug(f"NEW speaker detected: {speaker_id}")
        else:
            logger.debug(f"Matched speaker: {speaker_id} (confidence: {confidence:.3f})")
    except Exception as e:
        logger.warning(f"Diarization failed, defaulting to 'User': {e}")
        import traceback
        traceback.print_exc()
        speaker_id = "User"

    transcript = stt_model.stt(audio)
    logger.debug(f"[{speaker_id}] Transcript: {transcript}")
    conversation += f"\n{speaker_id}: {transcript}"
    logger.debug("Starting streamed LLM response...")
    text_buffer = ""
    ai_reply="AI:"
    # 1. Stream text from LLM as it’s generated
    for chunk in stream_llm_response(conversation):
        text_buffer += chunk
        ai_reply+=chunk

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
    if text_buffer:
        ai_reply+=text_buffer
        logger.debug(f"🗣️ TTS on final chunk: {text_buffer!r}")
        for audio_chunk in tts_model.stream_tts_sync(text_buffer):
            yield audio_chunk
    conversation+="\n"+ai_reply


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
