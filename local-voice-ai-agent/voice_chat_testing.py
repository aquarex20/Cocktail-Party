import sys
import argparse
import gradio as gr
from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model, WebRTC
from loguru import logger
from ollama import chat
from fastrtc import StreamHandler
from queue import Queue
import numpy as np
import asyncio
import json
import httpx
from fastrtc import AlgoOptions, SileroVadOptions

OLLAMA_MODEL = "gemma3:4b"
stt_model = get_stt_model()  # moonshine/base
tts_model = get_tts_model()  # kokoro

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")



def response(audio: tuple[int, np.ndarray]): # 
    sample_rate, audio_array = audio

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

stream = Stream(
    handler=ReplyOnPause(
        response,
        algo_options=AlgoOptions(
            audio_chunk_duration=0.6,
            started_talking_threshold=0.2,
            speech_threshold=0.1
        ),
        model_options=SileroVadOptions(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100
        )
    ),
    modality="audio",
    mode="send-receive",
)

#stream.ui.launch()


with gr.Blocks() as demo:
    gr.HTML(
    """
    <h1 style='text-align: center'>
    Chat (Powered by WebRTC ⚡️)
    </h1>
    """
    )
    with gr.Column():
        with gr.Group():
            audio = WebRTC(
                mode="send-receive",
                modality="audio",
            )
        audio.stream(fn=ReplyOnPause(
        response,
        algo_options=AlgoOptions(
            audio_chunk_duration=0.6,
            started_talking_threshold=0.2,
            speech_threshold=0.1
        ),
        model_options=SileroVadOptions(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100
        )
    ),
    inputs=[audio], outputs=[audio],
    )
    
    demo.launch()

