"""
Multi-speaker aware local voice AI agent.

Listens to conversations, detects multiple speakers, and provides
contextual insights + follow-up questions after 5 seconds of silence.

Usage:
    python local_voice_chat_multispeaker.py          # Launch with Gradio UI
    python local_voice_chat_multispeaker.py --phone  # Launch with phone interface

Environment Variables:
    HF_TOKEN: HuggingFace token for pyannote models (required)
              Get from: https://huggingface.co/settings/tokens
"""
import sys
import argparse
import os

from fastrtc import Stream, get_stt_model, get_tts_model
from loguru import logger
import requests
import json
import sounddevice as sd

from speaker_diarization_handler import MultiSpeakerPauseHandler
from config import DiarizationConfig, ConversationConfig, LLMConfig, AudioConfig


# Configuration
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Audio config - comment this if you want to use your own microphone
sd.default.device = ("BlackHole 2ch", 1)  # (input, output)

# Initialize STT and TTS models
stt_model = get_stt_model()  # moonshine/base
tts_model = get_tts_model()  # kokoro

# Configure logging
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Initialize configurations
diarization_config = DiarizationConfig(
    hf_token=HF_TOKEN,
    silence_threshold_seconds=5.0,  # Wait 5 seconds of silence before responding
    speaker_similarity_threshold=0.7,
)

conversation_config = ConversationConfig(
    max_history_turns=50,
)

llm_config = LLMConfig(
    model="gemma3:4b",
    max_response_tokens=150,
)


def stream_llm_response(conversation_context: str):
    """
    Stream text chunks from Ollama for contextual response.
    Generates an insight/fact + follow-up question based on conversation.
    """
    payload = {
        "model": llm_config.model,
        "messages": [
            {
                "role": "system",
                "content": llm_config.system_prompt,
            },
            {
                "role": "user",
                "content": conversation_context,
            },
        ],
        "options": {"num_predict": llm_config.max_response_tokens},
        "stream": True,
    }

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line.decode("utf-8"))
                chunk = ""
                if "message" in data and "content" in data["message"]:
                    chunk = data["message"]["content"].replace("*", "")
                elif "delta" in data:
                    chunk = data["delta"].replace("*", "")

                if chunk:
                    yield chunk
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama request error: {e}")
        yield "I apologize, I'm having trouble connecting to the language model."


def generate_contextual_response(conversation_context: str):
    """
    Generate a response with:
    1. An interesting insight/fact related to the discussion
    2. A thought-provoking follow-up question

    This function is called by the MultiSpeakerPauseHandler after
    detecting 5 seconds of silence.
    """
    logger.info("Generating contextual response...")
    logger.debug(f"Context:\n{conversation_context[:200]}...")

    text_buffer = ""
    full_response = ""

    # Stream text from LLM
    for chunk in stream_llm_response(conversation_context):
        text_buffer += chunk
        full_response += chunk

        # Speak at sentence boundaries or when buffer is large enough
        if any(p in text_buffer for p in [".", "!", "?"]) or \
           (len(text_buffer) > 80 and text_buffer[-1] == ","):
            speak_part = text_buffer
            text_buffer = ""

            logger.debug(f"TTS on chunk: {speak_part!r}")

            # Stream TTS for this chunk
            for audio_chunk in tts_model.stream_tts_sync(speak_part):
                yield audio_chunk

    # Flush any remaining text
    text_buffer = text_buffer.strip()
    if text_buffer:
        full_response += text_buffer
        logger.debug(f"TTS on final chunk: {text_buffer!r}")
        for audio_chunk in tts_model.stream_tts_sync(text_buffer):
            yield audio_chunk

    logger.info(f"Full AI response: {full_response}")


def create_stream():
    """Create the FastRTC stream with multi-speaker handler."""
    handler = MultiSpeakerPauseHandler(
        response_fn=generate_contextual_response,
        stt_model=stt_model,
        diarization_config=diarization_config,
        conversation_config=conversation_config,
    )
    return Stream(handler, modality="audio", mode="send-receive")


def check_requirements():
    """Check that all requirements are met before starting."""
    issues = []

    # Check HF token
    if not HF_TOKEN:
        issues.append(
            "HF_TOKEN environment variable not set.\n"
            "  Get a token from: https://huggingface.co/settings/tokens\n"
            "  Accept model terms at: https://huggingface.co/pyannote/segmentation-3.0\n"
            "  Then run: export HF_TOKEN='your_token_here'"
        )

    # Check Ollama
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        if response.status_code != 200:
            issues.append("Ollama is not responding properly")
    except requests.exceptions.RequestException:
        issues.append(
            "Ollama is not running.\n"
            "  Start it with: ollama serve\n"
            "  Then pull the model: ollama pull gemma3:4b"
        )

    return issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Speaker Voice AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python local_voice_chat_multispeaker.py          # Launch with Gradio UI
    python local_voice_chat_multispeaker.py --phone  # Launch with phone interface

Environment Variables:
    HF_TOKEN: HuggingFace token for pyannote speaker diarization models

Behavior:
    - Listens to conversation and identifies different speakers
    - When 2+ people are talking, AI stays silent
    - After 5 seconds of silence, AI provides:
      * An interesting insight related to the discussion
      * A follow-up question to continue the conversation
        """
    )
    parser.add_argument(
        "--phone",
        action="store_true",
        help="Launch with FastRTC phone interface",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip requirement checks",
    )
    args = parser.parse_args()

    # Check requirements
    if not args.skip_checks:
        issues = check_requirements()
        if issues:
            logger.error("Requirements not met:")
            for issue in issues:
                logger.error(f"  - {issue}")
            logger.info("Run with --skip-checks to bypass these checks")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("Multi-Speaker Voice AI Agent")
    logger.info("=" * 60)
    logger.info(f"Silence threshold: {diarization_config.silence_threshold_seconds}s")
    logger.info("Behavior: AI responds after 5s silence with insight + question")
    logger.info("=" * 60)

    # Create and launch stream
    stream = create_stream()

    if args.phone:
        logger.info("Launching with FastRTC phone interface...")
        stream.fastphone()
    else:
        logger.info("Launching with Gradio UI...")
        stream.ui.launch()
