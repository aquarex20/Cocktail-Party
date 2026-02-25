import sys
import argparse
import gradio as gr
from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model, WebRTC
from loguru import logger
from ollama import chat
from kokoro_onnx import Kokoro
stt_model = get_stt_model()  # moonshine/base
tts_model = get_tts_model()  # kokoro
kokoro_model=Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


def echo(audio):
    transcript = stt_model.stt(audio)
    logger.debug(f"🎤 Transcript: {transcript}")
    response = chat(
        model="gemma3:4b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful LLM in a Cocktail Party event. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include emojis or special characters in your answers. Respond to what the user said in a creative and helpful way in Italian.",
            },
            {"role": "user", "content": transcript},
        ],
        options={"num_predict": 200},
    )
    response_text = response["message"]["content"]
    logger.debug(f"🤖 Response: {response_text}")
    for audio_chunk in tts_model.stream_tts_sync(response_text):
        yield audio_chunk


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