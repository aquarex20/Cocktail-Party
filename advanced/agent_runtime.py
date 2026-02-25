"""
Agent Runtime - class-based agent for internal party mode.
Each agent: subscribes to another's output bus, publishes own TTS to its output bus.
"""

import time as time_module
import threading

import numpy as np
from loguru import logger

from audio_utils import play_audio
from conversation_mode import compute_conversation_context
from llm_client import stream_llm_response
from voice_models import stream_tts_sync, stt_transcribe


class AgentRuntime:
    """Single agent: STT → LLM → TTS, with bus publish/subscribe for party routing."""

    def __init__(
        self,
        agent_id: int,
        language: str = "en",
        tts_voice=None,
        monitor_output: bool = True,
        output_bus=None,
        stop_event: threading.Event = None,
        speaker_lock: threading.Lock = None,
    ):
        self.agent_id = agent_id
        self.language = language
        self.tts_voice = tts_voice
        self.speaker_lock = speaker_lock
        self.monitor_output = monitor_output
        self.output_bus = output_bus
        self.stop = stop_event or threading.Event()

        self.conversation = "\nTranscript:\n "
        self.summary = "\nSummary:\n"
        self.someone_talking = False  # unused in party mode; kept for compatibility
        self.ai_is_speaking = False
        self.ai_is_thinking = False  # True when has lock and streaming LLM (before TTS)

    def talk_generator(self):
        """Yield (sample_rate, audio) chunks from LLM+TTS. Uses conversation context."""
        if self.stop.is_set():
            return
        ctx = compute_conversation_context(self.conversation, self.summary)
        if ctx.wait_before_speaking_sec > 0:
            time_module.sleep(ctx.wait_before_speaking_sec)

        text_buffer = ""
        ai_reply = "\nAI:"

        self.ai_is_thinking = True
        for chunk in stream_llm_response(ctx.context, mode=ctx.mode, language=self.language):
            if self.stop.is_set() or self.someone_talking:
                self.ai_is_thinking = False
                return
            text_buffer += chunk
            ai_reply += chunk
            if any(p in text_buffer for p in [".", "!", "?"]) or (
                len(text_buffer) > 80 and text_buffer[-1] == ","
            ):
                speak_part = text_buffer
                text_buffer = ""
                self.ai_is_thinking = False
                for sr, audio_data in stream_tts_sync(
                    speak_part, self.language, voice=self.tts_voice
                ):
                    if self.stop.is_set():
                        return
                    yield (sr, audio_data)
                self.ai_is_thinking = True  # may get more LLM chunks

        self.ai_is_thinking = False
        text_buffer = text_buffer.strip()
        if text_buffer and not self.stop.is_set():
            for sr, audio_data in stream_tts_sync(
                text_buffer, self.language, voice=self.tts_voice
            ):
                if self.stop.is_set():
                    return
                yield (sr, audio_data)

        self.conversation += ai_reply

    def handle_utterance(self, audio_utter: np.ndarray, sample_rate: int = 16000):
        """Process one utterance: STT → LLM → TTS, publish to bus, optionally play.
        Uses speaker_lock so only one agent speaks at a time (no overlapping)."""
        if self.stop.is_set():
            return
        try:
            transcript = stt_transcribe((sample_rate, audio_utter), self.language)
        except Exception as e:
            logger.exception(f"[Agent{self.agent_id}] STT failed: {e}")
            return
        if not transcript or not transcript.strip():
            return

        self.conversation += f"\nUser: {transcript.strip()}"

        # Acquire speaker lock: only one agent produces audio at a time
        lock = self.speaker_lock
        if lock:
            lock.acquire()
        self.ai_is_speaking = True
        self.ai_is_thinking = False  # talk_generator will set it when in LLM phase
        try:
            for sr, audio_data in self.talk_generator():
                if self.stop.is_set():
                    break
                audio_data = np.asarray(audio_data, dtype=np.float32).flatten()
                if audio_data.size == 0:
                    continue
                if self.output_bus is not None:
                    self.output_bus.publish(sr, audio_data)
                if self.monitor_output:
                    play_audio(audio_data, sr)
        except Exception as e:
            logger.exception(f"[Agent{self.agent_id}] Talk failed: {e}")
        finally:
            self.ai_is_speaking = False
            self.ai_is_thinking = False
            if lock:
                lock.release()
