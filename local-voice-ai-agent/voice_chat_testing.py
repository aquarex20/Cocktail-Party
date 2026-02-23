import sys
import argparse
import gradio as gr
from fastrtc import ReplyOnPause, Stream, get_stt_model, WebRTC
from loguru import logger
from ollama import chat
from fastrtc import StreamHandler
from fastrtc import AdditionalOutputs
from queue import Queue
import numpy as np
import asyncio
import json
import httpx
from fastrtc import AlgoOptions, SileroVadOptions
import gradio as gr
from kokoro_onnx import Kokoro
import os
from dotenv import load_dotenv
from whisper_stt import WhisperSTT
load_dotenv()

OLLAMA_MODEL = "gemma3:4b"

stt_models={"English": get_stt_model(), "Italian": WhisperSTT(model_size="large-v3", chunk_s=0.5, overlap_s=0.5, device="cpu", compute_type="int8", language="it")}
kokoro=Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
#tts_model = get_tts_model()  # kokoro

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


LANGUAGE_STATE="English"
VOICE_STATE="af_nicole"
CONVERSATION_STATE=""

voices_choices={"English": ["af_heart", "af_bella","af_nicole", "am_michael", "am_puck"], "Italian": ["if_sara", "im_nicola"]}
def update_language(lang):
    global LANGUAGE_STATE, VOICE_STATE
    global speech_config, speech_synthesizer

    LANGUAGE_STATE = lang

    print("Language updated to:", lang) #to remove later 
    VOICE_STATE = voices_choices[lang][0]
    print("Voice updated to:", VOICE_STATE) #to remove later
    return gr.update(choices=voices_choices[lang], value=VOICE_STATE)

def update_voice(voice):
    global VOICE_STATE
    VOICE_STATE = voice
    print("Voice updated to:", VOICE_STATE) #to remove later

async def stream_tts(text):
    global LANGUAGE_STATE, VOICE_STATE
    async for samples, sr in kokoro.create_stream(
        text,
        voice=VOICE_STATE,
        speed=1.0,
        lang=LANGUAGE_STATE=="Italian" and "it" or "en-us",
    ):
        # Ensure (1, N)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        elif samples.ndim == 2 and samples.shape[0] != 1:
            # if it's (N,1) etc., normalize to (1,N)
            samples = samples.reshape(1, -1)

        # IMPORTANT: return (sample_rate, numpy_array)
        yield (sr, samples.astype(np.float32))

async def response(audio: tuple[int, np.ndarray], string_identifier: str, transformers_convo: list[dict]): # 
    global CONVERSATION_STATE, LANGUAGE_STATE, VOICE_STATE
    sample_rate, audio_array = audio

    transcript = stt_models[LANGUAGE_STATE].stt(audio)
    transformers_convo.append({"role": "user", "content": transcript})
    logger.debug(f"🎤 Transcript: {transcript}")
    response = chat(
        model="gemma3:4b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful LLM in a Cocktail Party event. Your goal is to answer the user and be helpful. Your output will be converted to audio so don't include emojis or special characters in your answers. Respond to what the user said in a creative and helpful way in "+LANGUAGE_STATE+".",
            },
            {"role": "user", "content": CONVERSATION_STATE},
        ],
        options={"num_predict": 200},
    )
    response_text = response["message"]["content"]
    logger.debug(f"🤖 Response: {response_text}")
    transformers_convo.append({"role": "assistant", "content": response_text})
    CONVERSATION_STATE += "User: " + transcript + "\n"
    CONVERSATION_STATE += "AI (you are talking to the user): " + response_text + "\n"
    
    async for sr, audio_chunk in stream_tts(response_text):
        yield (sr, audio_chunk), AdditionalOutputs(transformers_convo)


with gr.Blocks(css="""
.audio-container {
    position: relative !important;
    width: 100% !important;
    height: auto !important;
    inset: unset !important;
    z-index: 1 !important;
}
""") as demo:
    language_state = gr.State("English")
    voice_state = gr.State("af_nicole")
    conversation_state = gr.State("")
    transformers_convo = gr.State(value=[])

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

    with gr.Row():
        with gr.Column():
            audio = WebRTC(
                mode="send-receive",
                modality="audio",
            )

        with gr.Column():
            transcript = gr.Chatbot(label="transcript", type="messages")

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
    inputs=[audio, transformers_convo], outputs=[audio],
    )
    audio.on_additional_outputs(lambda s: s, # 
                                outputs=[transcript],
                                queue=False, show_progress="hidden")
    demo.launch()

