import sys
import argparse
import gradio as gr
from loguru import logger
from ollama import chat
from fastrtc import StreamHandler, AdditionalOutputs, ReplyOnPause, Stream, get_stt_model, WebRTC, AlgoOptions, SileroVadOptions
from queue import Queue
import numpy as np
import asyncio
import json
import httpx
from kokoro_onnx import Kokoro
import os
from dotenv import load_dotenv
from whisper_stt import transcribe_on_pause
import numpy as np
import onnxruntime as ort
from utilities import clean_text_for_tts, preprocess_audio, split_for_tts
from fastrtc import get_tts_model, KokoroTTSOptions
load_dotenv()

OLLAMA_MODEL = "gemma3:4b"

tts_model = get_tts_model()  # kokoro

kokoro=Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


voices_choices={"English": ["af_heart", "af_bella","af_nicole", "am_michael", "am_puck"], "Italian": ["if_sara", "im_nicola"]}

def update_language(lang):
    voices_choices={"English": ["af_heart", "af_bella","af_nicole", "am_michael", "am_puck"], "Italian": ["if_sara", "im_nicola"]}


    return voices_choices[lang][0], gr.update(choices=voices_choices[lang], value=voices_choices[lang][0]), lang


def update_voice(voice):
    return gr.update(value=voice), voice

            
async def stream_tts(text, voice_value: str, language_value: str):
    async for samples, sr in kokoro.create_stream(
        text,
        voice=voice_value,
        speed=1.0,
        lang=language_value=="Italian" and "it" or "en-us",
    ):
        # Ensure (1, N)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        elif samples.ndim == 2 and samples.shape[0] != 1:
            # if it's (N,1) etc., normalize to (1,N)
            samples = samples.reshape(1, -1)

        # IMPORTANT: return (sample_rate, numpy_array)
        yield (sr, samples.astype(np.float32))

 
def response(audio: tuple[int, np.ndarray], string_identifier: str, transformers_convo: list[dict],conversation_value: str, language_value: str, voice_value: str): # 
    sample_rate, audio_array = preprocess_audio(*audio)

    transcript =transcribe_on_pause((sample_rate, audio_array), language_value=="Italian" and "it" or "en")
    if transcript is None:
        print("No transcript")
        return
    new_convo=transformers_convo+[{"role": "user", "content": transcript}]
    conversation_value += "User: " + transcript + "\n"
    logger.debug(f"🎤 Transcript: {transcript}")
    response = chat(
        model="gemma3:4b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful LLM in a Cocktail Party event. Your goal is to answer the user and be helpful. Your output will be converted to audio so don't include emojis or special characters in your answers. Respond to what the user said in a creative and helpful way in "+language_value+".",
            },
            {"role": "user", "content": conversation_value},
        ],
        options={"num_predict": 100},
    )
    response_text = clean_text_for_tts(response["message"]["content"])
    logger.debug(f"🤖 Response: {response_text}")
    new_convo= new_convo +[{"role": "assistant", "content": response_text}]
    conversation_value += "AI (you are talking to the user): " + response_text + "\n"
    
    chunks = split_for_tts(response_text)
    print("done")
    for audio_chunk in tts_model.stream_tts_sync(response_text, KokoroTTSOptions(voice=voice_value, speed=1.0, lang=language_value=="Italian" and "it" or "en-us")):
        yield audio_chunk, AdditionalOutputs(new_convo, conversation_value)

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
    voice_state = gr.State("af_heart")
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
        outputs=(voice_state, voice_selector, language_state)
    )
    voice_selector.change(
        fn=update_voice,
        inputs=voice_selector,
        outputs= (voice_selector, voice_state)
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
        response    ),
        inputs=[audio, transformers_convo, conversation_state, language_state, voice_state], 
        outputs=[audio],
        )
        audio.on_additional_outputs(
            lambda s, a: (s,a),
            outputs=[transcript, conversation_state],
            queue=False,
            show_progress="hidden",
        )  
    
    demo.launch(inbrowser=True)   

