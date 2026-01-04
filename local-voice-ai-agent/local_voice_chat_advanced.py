import sys
import argparse
import time
from dataclasses import dataclass, field
from threading import Event
from typing import Generator, Optional

import numpy as np
from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model
from fastrtc.tracks import StreamHandler, EmitType
# VAD not needed - using simple RMS-based detection
from fastrtc.utils import create_message
from loguru import logger
from ollama import chat
import requests
import json

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
import sounddevice as sd

# Configuration for silence detection
SILENCE_THRESHOLD_SECONDS = 4.0  # Wait 4 seconds of silence before AI responds
RMS_THRESHOLD = 0.01  # RMS threshold for speech detection (0-1 range after normalization)

#comment this if you want to use your own microphone with your own party
sd.default.device = ("BlackHole 2ch", 1)  # (input, output)


@dataclass
class SilenceState:
    """State for silence-aware handler."""
    sampling_rate: int = 0
    started_talking: bool = False
    responding: bool = False
    silence_start_time: Optional[float] = None
    has_speech_content: bool = False


class SilenceAwareHandler(StreamHandler):
    """
    A StreamHandler that waits for extended silence before responding.

    Key behavior:
    - Buffers all audio while people are talking
    - Tracks silence duration using Silero VAD
    - Only triggers AI response after SILENCE_THRESHOLD_SECONDS of silence
    - This prevents AI from interrupting ongoing conversations
    """

    def __init__(
        self,
        response_fn,
        silence_threshold: float = SILENCE_THRESHOLD_SECONDS,
        rms_threshold: float = RMS_THRESHOLD,
        expected_layout: str = "mono",
        output_sample_rate: int = 24000,
        output_frame_size: int = 480,
        input_sample_rate: int = 48000,
    ):
        super().__init__(
            expected_layout,
            output_sample_rate,
            output_frame_size,
            input_sample_rate=input_sample_rate,
        )

        self.response_fn = response_fn
        self.silence_threshold = silence_threshold
        self.rms_threshold = rms_threshold

        # State
        self.state = SilenceState()
        self.event = Event()
        self.generator: Optional[Generator] = None

        # Audio buffer to accumulate speech
        self.audio_buffer: list = []

        logger.info(f"SilenceAwareHandler initialized with {silence_threshold}s silence threshold")

    def copy(self):
        return SilenceAwareHandler(
            self.response_fn,
            self.silence_threshold,
            self.rms_threshold,
            self.expected_layout,
            self.output_sample_rate,
            self.output_frame_size,
            self.input_sample_rate,
        )

    def receive(self, frame: tuple) -> None:
        """Process incoming audio frame."""
        sample_rate, audio = frame
        current_time = time.time()

        # Debug: log that we're receiving audio
        if not hasattr(self, '_logged_receive'):
            logger.debug(f"Receiving audio: sample_rate={sample_rate}, shape={audio.shape}")
            self._logged_receive = True

        # Don't process new audio while AI is responding
        if self.state.responding:
            return

        # Store sample rate
        if not self.state.sampling_rate:
            self.state.sampling_rate = sample_rate

        # Squeeze audio to 1D
        audio_array = np.squeeze(audio)

        # Simple RMS-based speech detection (works on any chunk size)
        # Normalize int16 audio to float (-1 to 1 range)
        audio_float = audio_array.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_float**2))

        # Detect speech based on RMS threshold
        is_speech = rms > self.rms_threshold

        # Debug: periodically log RMS results
        if not hasattr(self, '_log_count'):
            self._log_count = 0
        self._log_count += 1
        if self._log_count % 50 == 0:  # Log every 50 frames (~1 second)
            logger.debug(f"RMS: {rms:.4f}, is_speech={is_speech}")

        if is_speech:
            # Speech detected - buffer audio and reset silence timer
            self.audio_buffer.append(audio_array)
            self.state.started_talking = True
            self.state.has_speech_content = True
            self.state.silence_start_time = None  # Reset silence timer
            logger.debug("Speech detected, buffering audio...")
        else:
            # Silence detected
            if self.state.started_talking:
                # Start or continue silence timer
                if self.state.silence_start_time is None:
                    self.state.silence_start_time = current_time
                    logger.debug("Silence started...")
                else:
                    silence_duration = current_time - self.state.silence_start_time

                    # Check if we've reached the silence threshold
                    if silence_duration >= self.silence_threshold and self.state.has_speech_content:
                        logger.info(f"Silence threshold reached ({silence_duration:.1f}s), triggering response")
                        self.event.set()

    def emit(self) -> Optional[EmitType]:
        """Generate AI response when silence threshold is reached."""
        if not self.event.is_set():
            return None

        if not self.generator:
            self.send_message_sync(create_message("log", "silence_threshold_reached"))
            logger.info("Generating AI response after silence...")

            # Concatenate all buffered audio
            if self.audio_buffer:
                full_audio = np.concatenate(self.audio_buffer)
                audio_tuple = (self.state.sampling_rate, full_audio.reshape(1, -1))
            else:
                # No audio buffered, skip
                self.reset()
                return None

            # Create generator from response function
            self.generator = self.response_fn(audio_tuple)
            self.state.responding = True

        try:
            output = next(self.generator)
            return output
        except StopIteration:
            logger.info("AI response complete")
            self.reset()
            return None
        except Exception as e:
            logger.error(f"Error in emit: {e}")
            self.reset()
            raise

    def reset(self):
        """Reset state after response."""
        super().reset()
        self.generator = None
        self.event.clear()
        self.state = SilenceState()
        self.audio_buffer = []


stt_model = get_stt_model()  # moonshine/base
tts_model = get_tts_model()  # kokoro

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
    transcript = stt_model.stt(audio)
    logger.debug(f"🎤 Transcript: {transcript}")
    conversation+="\nUser:"+transcript
    logger.debug("🧠 Starting streamed LLM response...")
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
    # Use SilenceAwareHandler instead of ReplyOnPause
    # This waits for 4 seconds of silence before AI responds
    return Stream(SilenceAwareHandler(echo), modality="audio", mode="send-receive")


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
