import sys
import argparse
import gradio as gr
from fastrtc import ReplyOnPause, Stream, get_stt_model, WebRTC
from loguru import logger
from ollama import chat
from fastrtc import StreamHandler
from queue import Queue
import numpy as np
import asyncio
import json
import httpx
from fastrtc import AlgoOptions, SileroVadOptions
import gradio as gr
from kokoro_onnx import Kokoro



OLLAMA_MODEL = "gemma3:4b"

stt_model = get_stt_model()  # moonshine/base

kokoro=Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
#tts_model = get_tts_model()  # kokoro

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


language_state = gr.State("English")
voice_state = gr.State("af_nicole")

voices_choices={"English": ["af_heart", "af_bella","af_nicole", "am_michael", "am_puck"], "Italian": ["if_sara", "im_nicola"]}
def update_language(lang):
    language_state.value = lang
    print("Language updated to:", lang) #to remove later 
    voice_state.value = voices_choices[lang][0]
    print("Voice updated to:", voice_state.value) #to remove later
    return gr.update(choices=voices_choices[lang], value=voice_state.value)

def update_voice(voice):
    voice_state.value = voice
    print("Voice updated to:", voice_state.value) #to remove later

async def stream_tts(text):
    async for samples, sr in kokoro.create_stream(
        text,
        voice=voice_state.value,
        speed=1.0,
        lang=language_state.value=="Italian" and "it" or "en-us",
    ):
        # Ensure (1, N)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        elif samples.ndim == 2 and samples.shape[0] != 1:
            # if it's (N,1) etc., normalize to (1,N)
            samples = samples.reshape(1, -1)

        # IMPORTANT: return (sample_rate, numpy_array)
        yield (sr, samples.astype(np.float32))

async def response(audio: tuple[int, np.ndarray]): # 
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
    async for sr, audio_chunk in stream_tts(response_text):
        yield (sr, audio_chunk)


with gr.Blocks(css="""
.audio-container {
    position: relative !important;
    width: 100% !important;
    height: auto !important;
    inset: unset !important;
    z-index: 1 !important;
}
""") as demo:
    gr.HTML(
    """
    <h1 style='text-align: center'>
    Chat (Powered by WebRTC ⚡️)
    </h1>
    """
    )
    gr.Markdown("## Language Selector")

    language_selector = gr.Dropdown(
        choices=["English", "Italian"],
        value="English",
        label="Choose Language"
    )

    voice_selector = gr.Dropdown(
        choices=voices_choices[language_state.value],
        value=voice_state.value,
        label="Choose Voice"
    )

    # 🔹 This updates immediately when changed
    language_selector.change(
        fn=update_language,
        inputs=language_selector,
        outputs=voice_selector
    )
    voice_selector.change(
        fn=update_voice,
        inputs=voice_selector,
        outputs=None
    )

    with gr.Column(scale=1):
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
    with gr.Row():
        with gr.Column():
            gr.Textbox(label="Conversation", interactive=False, lines=10)

    demo.launch()

