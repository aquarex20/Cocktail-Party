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
import html
import time
load_dotenv()

OLLAMA_MODEL = "gemma3:4b"

tts_model = get_tts_model()  # kokoro

kokoro=Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

voices_choices={"English": ["af_heart", "af_bella","af_nicole", "am_michael", "am_puck"], "Italian": ["if_sara", "im_nicola"]}
def sanitize_messages(msgs):
    safe = []
    for m in msgs or []:
        role = m.get("role", "assistant")
        content = m.get("content", "")
        if content is None:
            content = ""
        # force string
        content = str(content)
        safe.append({"role": role, "content": content})
    return safe
def render_chat(msgs):
    print(f"Rendering chat: {msgs}")
    out = ["<div style='display:flex;flex-direction:column;gap:8px;'>"]
    for m in msgs:
        role = m["role"]
        txt = html.escape(m["content"])
        if role == "user":
            out.append(
                "<div style='align-self:flex-end;max-width:75%;"
                "background:#2b2b2b;color:#fff;padding:10px 12px;"
                "border-radius:16px 16px 4px 16px;'>"
                f"{txt}</div>"
            )
        else:
            out.append(
                "<div style='align-self:flex-start;max-width:75%;"
                "background:#f2f2f2;color:#111;padding:10px 12px;"
                "border-radius:16px 16px 16px 4px;'>"
                f"{txt}</div>"
            )
    out.append("</div>")
    return "\n".join(out)


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

def sanitize_messages(msgs):
    safe = []
    for m in msgs or []:
        role = m.get("role", "assistant")
        content = m.get("content", "")
        if content is None:
            content = ""
        # force string
        content = str(content)
        safe.append({"role": role, "content": content})
    return safe

def _validate_chatbot_messages(msgs):
    if not isinstance(msgs, list):
        raise TypeError(f"chatbot payload must be list, got {type(msgs)}")
    for m in msgs[-5:]:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise TypeError(f"bad message: {m!r}")


def response(audio: tuple[int, np.ndarray], string_identifier: str, transformers_convo: list[dict],conversation_value: str, language_value: str, voice_value: str): # 
    sample_rate, audio_array = preprocess_audio(*audio)

    transcript =transcribe_on_pause((sample_rate, audio_array), language_value=="Italian" and "it" or "en")
    if transcript is None:
        print("No transcript")
        return
    new_convo=sanitize_messages(transformers_convo)+[{"role": "user", "content": transcript}]
    conversation_value += "User: " + transcript + "\n"

    # before yielding

    #yield AdditionalOutputs(("user", transcript))

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
    #yield AdditionalOutputs(("assistant", response_text))

    for audio_chunk in tts_model.stream_tts_sync(response_text, KokoroTTSOptions(voice=voice_value, speed=1.0, lang=language_value=="Italian" and "it" or "en-us")):
        yield audio_chunk, AdditionalOutputs(conversation_value, new_convo)
def render_bubbles(messages):
    """
    messages: list of dicts like {"role": "user"|"assistant", "content": "..."}
    returns: HTML string to put inside gr.Markdown
    """
    out = ["""
    <div style="
      height:420px;
      overflow-y:auto;
      overflow-x:hidden;
      border:1px solid #3333;
      border-radius:12px;
      padding:12px;
    ">
    <div style="display:flex;flex-direction:column;gap:10px;">
    """]

    for m in messages:
        role = m.get("role", "assistant")
        text = html.escape(str(m.get("content", "") or ""))

        if role == "user":
            out.append(
                "<div style='display:flex;justify-content:flex-end;'>"
                "<div style='max-width:75%;background:#2b2b2b;color:#fff;"
                "padding:10px 12px;border-radius:16px 16px 4px 16px;"
                "white-space:pre-wrap;word-wrap:break-word;'>"
                f"{text}</div></div>"
            )
        else:
            out.append(
                "<div style='display:flex;justify-content:flex-start;'>"
                "<div style='max-width:75%;background:#f2f2f2;color:#111;"
                "padding:10px 12px;border-radius:16px 16px 16px 4px;"
                "white-space:pre-wrap;word-wrap:break-word;'>"
                f"{text}</div></div>"
            )

    out.append("</div>")
    return "\n".join(out)

def ao_handler(*args):
    # Most common: args[0] is the payload you yielded in AdditionalOutputs(...)
    payload = args[0]

    # If you yielded AdditionalOutputs(x, y), payload is usually (x, y)
    if isinstance(payload, (list, tuple)) and len(payload) == 2:
        return payload[0], payload[1]

    # Fallback: don't break UI
    return gr.update(), gr.update()

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
    transformers_convo = gr.State(value=[])
    conversation_state = gr.State("")

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
            gr.Markdown("<div style='text-align:center'><h2>Chat History</h2></div>")

            chat_md = gr.HTML()
            chat_state = gr.State([])

            timer=gr.Timer(2.0)
            def update_html(transformers_convo):
                return render_bubbles(transformers_convo)
            timer.tick(update_html, inputs=[transformers_convo], outputs=[chat_md])
        audio.stream(fn=ReplyOnPause(
        response    ),
        inputs=[audio, transformers_convo, conversation_state, language_state, voice_state], 
        outputs=[audio],
        )
        audio.on_additional_outputs( lambda s,r: (s,r),
            outputs=[conversation_state, transformers_convo],
            queue=False
        )  
demo.launch(inbrowser=True, debug=True)   

