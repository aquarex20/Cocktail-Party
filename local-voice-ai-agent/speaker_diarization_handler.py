"""
Multi-speaker aware stream handler that replaces ReplyOnPause.
Detects multiple speakers and only responds after silence threshold.
"""
import time
from dataclasses import dataclass, field
from threading import Event
from typing import Generator, Optional, Callable, Any

import numpy as np
from loguru import logger

from fastrtc.tracks import StreamHandler, EmitType
from fastrtc.utils import create_message

from config import DiarizationConfig, ConversationConfig
from diarization_engine import DiarizationEngine
from conversation_tracker import ConversationTracker


@dataclass
class MultiSpeakerState:
    """State for multi-speaker handler."""
    # Audio stream accumulated for transcription
    stream: Optional[np.ndarray] = None
    sampling_rate: int = 0

    # Speech detection state
    started_talking: bool = False
    responding: bool = False

    # Buffer for incoming audio
    buffer: Optional[np.ndarray] = None

    # Whether silence trigger has fired
    silence_triggered: bool = False

    # Track last process time
    last_process_time: float = 0.0

    # Track if we've responded in this silence period
    has_responded_this_silence: bool = False

    def new(self):
        """Create a fresh state, preserving started_talking."""
        return MultiSpeakerState()


# Type for response function
ResponseFn = Callable[[str], Generator[EmitType, None, None]]


class MultiSpeakerPauseHandler(StreamHandler):
    """
    A StreamHandler that:
    1. Detects when multiple speakers are talking (stays silent)
    2. Only responds after configurable silence duration (default 5s)
    3. Tracks full conversation with speaker attribution
    4. Generates contextual insights and follow-up questions
    """

    def __init__(
        self,
        response_fn: ResponseFn,
        stt_model: Any,
        diarization_config: Optional[DiarizationConfig] = None,
        conversation_config: Optional[ConversationConfig] = None,
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
        self.stt_model = stt_model

        # Initialize diarization
        self.diarization_config = diarization_config or DiarizationConfig()
        self.diarization_engine = DiarizationEngine(self.diarization_config)

        # Initialize conversation tracker
        self.conversation_config = conversation_config or ConversationConfig()
        self.conversation_tracker = ConversationTracker(self.conversation_config)

        # State management
        self.state = MultiSpeakerState()
        self.event = Event()
        self.generator: Optional[Generator[EmitType, None, None]] = None

        # Audio buffer for transcription
        self.transcription_buffer: list = []
        self.buffer_start_time: Optional[float] = None

        # Track last AI response text for adding to conversation
        self.last_ai_response: str = ""

        logger.info(
            f"MultiSpeakerPauseHandler initialized with "
            f"silence_threshold={self.diarization_config.silence_threshold_seconds}s"
        )

    def copy(self):
        """Create a copy of this handler."""
        return MultiSpeakerPauseHandler(
            self.response_fn,
            self.stt_model,
            self.diarization_config,
            self.conversation_config,
            self.expected_layout,
            self.output_sample_rate,
            self.output_frame_size,
            self.input_sample_rate,
        )

    def receive(self, frame: tuple) -> None:
        """Process incoming audio frame."""
        sample_rate, audio = frame
        current_time = time.time()

        # Don't process while AI is speaking
        if self.state.responding:
            return

        # Squeeze to 1D if needed
        audio_array = np.squeeze(audio)

        # Store sampling rate
        if not self.state.sampling_rate:
            self.state.sampling_rate = sample_rate

        # Process through diarization engine
        is_speech, speaker_count, speaker_ids = self.diarization_engine.process_audio_chunk(
            audio_array.astype(np.float32),
            sample_rate,
            current_time
        )

        if is_speech:
            # Add to transcription buffer
            if self.buffer_start_time is None:
                self.buffer_start_time = current_time
            self.transcription_buffer.append(audio_array)
            self.state.started_talking = True

            # Reset the silence response flag when speech detected
            self.state.has_responded_this_silence = False

            # Log multi-speaker detection
            if speaker_count > 1:
                self.send_message_sync(create_message("log", "multi_speaker_detected"))
                logger.debug(f"Multi-speaker activity: {speaker_ids}")
        else:
            # Check for silence trigger
            if (self.diarization_engine.should_ai_respond(current_time)
                and not self.state.has_responded_this_silence):

                # Transcribe buffered audio first
                if self.state.started_talking and self.transcription_buffer:
                    self._process_and_transcribe()

                # Only trigger if we have conversation context
                if self.conversation_tracker.get_turn_count() > 0:
                    logger.info("Silence threshold reached, triggering AI response")
                    self.event.set()
                    self.state.silence_triggered = True
                    self.state.has_responded_this_silence = True

    def _process_and_transcribe(self) -> None:
        """Transcribe buffered audio and add to conversation."""
        if not self.transcription_buffer:
            return

        try:
            # Concatenate all buffered audio
            full_audio = np.concatenate(self.transcription_buffer)

            # Transcribe using STT model
            # The stt_model.stt expects (sample_rate, audio) tuple
            transcript = self.stt_model.stt((self.state.sampling_rate, full_audio))

            if transcript and transcript.strip():
                # Get the current speaker (last identified)
                speakers = self.diarization_engine.state.current_speakers
                speaker_id = speakers[0] if speakers else "SPEAKER_00"

                # Add to conversation tracker
                self.conversation_tracker.add_turn(speaker_id, transcript)
                logger.info(f"Transcribed [{speaker_id}]: {transcript[:50]}...")

        except Exception as e:
            logger.error(f"Transcription error: {e}")

        finally:
            # Clear buffer
            self.transcription_buffer = []
            self.buffer_start_time = None

    def emit(self) -> Optional[EmitType]:
        """Generate AI response when triggered."""
        if not self.event.is_set():
            return None

        if not self.generator:
            self.send_message_sync(create_message("log", "ai_response_triggered"))
            logger.info("Generating AI response...")

            # Get conversation context
            context = self.conversation_tracker.get_formatted_context(max_turns=20)

            # Log context for debugging
            topics = self.conversation_tracker.get_recent_topics()
            logger.debug(f"Context topics: {topics}")

            # Generate response
            self.generator = self.response_fn(context)
            self.state.responding = True
            self.last_ai_response = ""

        try:
            output = next(self.generator)

            # Track the response text if it's a tuple with text
            if isinstance(output, tuple) and len(output) >= 2:
                # Output is typically (sample_rate, audio_array)
                pass

            return output

        except StopIteration:
            # Response complete
            logger.info("AI response complete")

            # Add AI response to conversation (if we tracked it)
            if self.last_ai_response:
                self.conversation_tracker.add_turn("AI", self.last_ai_response, is_ai=True)

            # Reset the diarization silence timer
            self.diarization_engine.reset_silence_timer()

            self.reset()
            return None

        except Exception as e:
            logger.error(f"Error in emit: {e}")
            self.reset()
            raise

    def reset(self) -> None:
        """Reset state after response."""
        super().reset()
        self.generator = None
        self.event.clear()
        self.state = MultiSpeakerState()
        self.transcription_buffer = []
        self.buffer_start_time = None
        self.last_ai_response = ""

    def get_conversation_context(self) -> str:
        """Get the current conversation context."""
        return self.conversation_tracker.get_formatted_context()

    def get_speaker_count(self) -> int:
        """Get number of detected speakers."""
        return self.diarization_engine.get_speaker_count()

    def clear_conversation(self) -> None:
        """Clear conversation history."""
        self.conversation_tracker.clear()
        self.diarization_engine.reset()
