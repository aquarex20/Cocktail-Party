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
import threading
import base64
import secrets
import copy
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

DIARIZATION_SERVER_URL = os.getenv("DIARIZATION_SERVER_URL", "http://127.0.0.1:8001")
SESSION_ID = secrets.token_urlsafe(12)

def get_diarization_utterances(session_id: str) -> dict:
    response = httpx.get(f"{DIARIZATION_SERVER_URL}/diarize/utterances/{session_id}")
    #print(f"Diarization utterances: {response.json()}")
    return response.json()
def _speaker_runs_from_assigned(assigned: dict | None):
    """
    Returns list of {"speaker": str, "text": str} runs in order.
    """
    if not isinstance(assigned, dict):
        return []

    runs = []
    cur_spk = None
    cur_words = []

    for seg in assigned.get("segments") or []:
        words = seg.get("words") or []
        for w in words:
            spk = w.get("speaker") or seg.get("speaker") or "unassigned"
            token = (w.get("word") or "").strip()
            if not token:
                continue

            if cur_spk is None:
                cur_spk, cur_words = spk, [token]
            elif spk == cur_spk:
                cur_words.append(token)
            else:
                runs.append({"speaker": cur_spk, "text": " ".join(cur_words)})
                cur_spk, cur_words = spk, [token]

    if cur_spk is not None and cur_words:
        runs.append({"speaker": cur_spk, "text": " ".join(cur_words)})

    return runs


def annotate_user_messages_with_speaker_runs(transformers_convo: list[dict], utterances: dict) -> list[dict]:
    """
    "Monolith" approach:
    - Keep ONE user message per reply_id
    - Attach `speaker_runs=[{speaker,text},...]` when available
    - Never expands the list length (so timer ticks stay idempotent)
    """
    out = []
    for m in transformers_convo or []:
        if m.get("role") != "user":
            out.append(m)
            continue

        rid = m.get("reply_id")
        u = (utterances or {}).get(rid) or {}
        assigned = u.get("assigned_refined") or u.get("assigned")
        runs = _speaker_runs_from_assigned(assigned)

        mm = dict(m)
        if runs:
            mm["speaker_runs"] = runs
            if len(runs) == 1:
                mm["speaker_label"] = runs[0].get("speaker") or "unassigned"
            else:
                mm["speaker_label"] = "mixed"
        else:
            mm["speaker_runs"] = []
            mm["speaker_label"] = "unassigned"
        out.append(mm)

    return out

def enqueue_diarization(
    session_id: str,
    utterance_id: str,
    audio_array: np.ndarray,
    sample_rate: int,
    whisper_segments: list[dict] | None,
    language_code: str,
) -> None:
    """
    Fire-and-forget diarization: send audio to the FastAPI server without blocking UI/audio.
    """
    try:
        payload_b64 = base64.b64encode(audio_array.astype(np.float32, copy=False).tobytes()).decode("ascii")
    except Exception:
        return

    def _post():
        try:
            httpx.post(
                f"{DIARIZATION_SERVER_URL}/diarize/ingest_assign",
                json={
                    "name": session_id or "default",
                    "utterance_id": utterance_id,
                    "sample_rate": int(sample_rate),
                    "audio_f32_b64": payload_b64,
                    "whisper_segments": whisper_segments,
                    "language": language_code,
                },
                timeout=2,
            )
        except Exception:
            pass

    threading.Thread(target=_post, daemon=True).start()
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


def convo_to_transcript(transformers_convo: list[dict]) -> str:
    lines = []
    for m in transformers_convo or []:
        role = m.get("role", "assistant")

        if role == "assistant":
            lines.append(f"AI: {m.get('content','')}")
            continue

        if role == "user":
            runs = m.get("speaker_runs") or []
            if isinstance(runs, list) and runs:
                for r in runs:
                    spk = r.get("speaker") or "unassigned"
                    txt = r.get("text") or ""
                    if txt.strip():
                        lines.append(f"{spk}: {txt}")
            else:
                spk = m.get("speaker_label") or "unassigned"
                lines.append(f"{spk}: {m.get('content','')}")
            continue

        # fallback for any other roles
        lines.append(f"{role.upper()}: {m.get('content','')}")

    return "\n".join(lines).strip()
def response(audio: tuple[int, np.ndarray], string_identifier: str, transformers_convo: list[dict],conversation_value: str, language_value: str, voice_value: str): # 
    sample_rate, audio_array = preprocess_audio(*audio)
    reply_id = secrets.token_urlsafe(12)

    transcript, whisper_segments, lang_code = transcribe_on_pause(
        (sample_rate, audio_array),
        language_value == "Italian" and "it" or "en",
        return_segments=True,
    )
    if transcript is None:
        print("No transcript")
        return

    # Non-blocking server-side: diarize + align + assign word speakers.
    enqueue_diarization(SESSION_ID, reply_id, audio_array, sample_rate, whisper_segments, lang_code)

    new_convo=transformers_convo+[{"role": "user", "content": transcript, "reply_id": reply_id}]
    conversation_value += "User: " + transcript + "\n"

    # before yielding

    #yield AdditionalOutputs(("user", transcript))

    logger.debug(f"🎤 Transcript: {transcript}")
    response = chat(
        model="gemma3:4b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant at a cocktail party.\n"
                    "The conversation transcript you receive may contain speaker labels like SPK_00, SPK_01, etc., and AI.\n"
                    "Those labels are ONLY for context.\n"
                    "Respond naturally to the conversation in "
                    + language_value +
                    ".\n"
                    "IMPORTANT: Output ONLY the response text. Do NOT include any speaker labels or prefixes (no 'AI:', no 'User:', no 'SPK_00:')."
                ),
            },
            {"role": "user", "content": convo_to_transcript(new_convo) or "User: " + transcript + "\n"},
        ],
        options={"num_predict": 200},
    )
    response_text = clean_text_for_tts(response["message"]["content"])
    logger.debug(f"🤖 Response: {response_text}")
    new_convo= new_convo +[{"role": "assistant", "content": response_text}]
    conversation_value += "AI: " + response_text + "\n"
    #yield AdditionalOutputs(("assistant", response_text))

    for audio_chunk in tts_model.stream_tts_sync(response_text, KokoroTTSOptions(voice=voice_value, speed=1.0, lang=language_value=="Italian" and "it" or "en-us")):
        yield audio_chunk, AdditionalOutputs(conversation_value, new_convo)
def render_bubbles(messages):
    """
    messages: list of dicts like {"role": "user"|"assistant", "content": "..."}
    returns: HTML string to put inside gr.Markdown
    """
    def _user_bg_for_label(label: str | None) -> str:
        """
        Deterministic, high-separation colors for speaker labels.
        - SPK_00, SPK_01, ... map to distinct hues (no palette collisions)
        - unassigned -> neutral gray
        """
        import re

        if not label or label == "unassigned":
            return "#555555"

        m = re.match(r"^SPK_(\d+)$", str(label))
        if m:
            i = int(m.group(1))
            # golden angle for evenly spaced hues
            hue = (i * 137.508) % 360.0
            return f"hsl({hue:.1f}, 55%, 33%)"

        # fallback for any other label
        hue = (sum(ord(c) for c in str(label)) * 13) % 360
        return f"hsl({hue}, 50%, 34%)"

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
        speaker_label = m.get("speaker_label")
        header_label = "AI" if role != "user" else (speaker_label or "unassigned")
        header_label = html.escape(str(header_label))

        if role == "user":
            runs = m.get("speaker_runs") or []
            if isinstance(runs, list) and len(runs) > 0:
                out.append(
                    "<div style='display:flex;justify-content:flex-end;'>"
                    "<div style='display:flex;flex-direction:column;gap:6px;max-width:75%;align-items:flex-end;'>"
                )
                for r in runs:
                    spk = str(r.get("speaker") or "unassigned")
                    run_text = html.escape(str(r.get("text") or ""))
                    bg = _user_bg_for_label(spk)
                    spk_h = html.escape(spk)
                    out.append(
                        "<div style='width:100%;background:"
                        f"{bg}"
                        ";color:#fff;"
                        "padding:10px 12px;border-radius:16px 16px 4px 16px;"
                        "white-space:pre-wrap;word-wrap:break-word;'>"
                        f"<div style='font-size:11px;opacity:0.9;margin-bottom:6px;font-weight:600;'>{spk_h}</div>"
                        f"{run_text}</div>"
                    )
                out.append("</div></div>")
            else:
                bg = _user_bg_for_label(str(speaker_label) if speaker_label is not None else None)
                out.append(
                    "<div style='display:flex;justify-content:flex-end;'>"
                    "<div style='max-width:75%;background:"
                    f"{bg}"
                    ";color:#fff;"
                    "padding:10px 12px;border-radius:16px 16px 4px 16px;"
                    "white-space:pre-wrap;word-wrap:break-word;'>"
                    f"<div style='font-size:11px;opacity:0.9;margin-bottom:6px;font-weight:600;'>{header_label}</div>"
                    f"{text}</div></div>"
                )
        else:
            out.append(
                "<div style='display:flex;justify-content:flex-start;'>"
                "<div style='max-width:75%;background:#f2f2f2;color:#111;"
                "padding:10px 12px;border-radius:16px 16px 16px 4px;"
                "white-space:pre-wrap;word-wrap:break-word;'>"
                f"<div style='font-size:11px;opacity:0.8;margin-bottom:6px;font-weight:600;color:#555;'>{header_label}</div>"
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

            timer_diarization=gr.Timer(2.0)
            def update_convo_with_diarization(transformers_convo):
                diarization_utterances = get_diarization_utterances(SESSION_ID)
                new_convo = annotate_user_messages_with_speaker_runs(transformers_convo, diarization_utterances)
                return new_convo
            timer_diarization.tick(update_convo_with_diarization, inputs=[transformers_convo], outputs=[transformers_convo])
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

