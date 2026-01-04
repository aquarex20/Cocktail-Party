"""
Real-time speaker diarization engine using pyannote-audio.
Processes audio in streaming chunks and maintains speaker embeddings.
"""
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import time

import torch
import librosa
from loguru import logger

from config import DiarizationConfig


@dataclass
class SpeakerSegment:
    """Represents a segment of speech from a specific speaker."""
    speaker_id: str
    start_time: float
    end_time: float
    embedding: Optional[np.ndarray] = None
    confidence: float = 1.0


@dataclass
class DiarizationState:
    """Maintains the current state of the diarization engine."""
    # Known speaker embeddings: speaker_id -> embedding vector
    known_speakers: Dict[str, np.ndarray] = field(default_factory=dict)

    # Currently detected speakers in the active window
    current_speakers: List[str] = field(default_factory=list)

    # Timestamp of last detected speech
    last_speech_time: float = 0.0

    # When silence started (None if currently speaking)
    silence_start_time: Optional[float] = None

    # Whether multiple speakers were detected in recent window
    is_multi_speaker: bool = False

    # Audio buffer for processing (stores ~10s at 16kHz)
    audio_buffer: deque = field(default_factory=lambda: deque(maxlen=160000))

    # Track if we've had speech before (to avoid triggering on startup)
    has_had_speech: bool = False


class DiarizationEngine:
    """
    Real-time speaker diarization using pyannote-audio.

    Key features:
    - Streaming audio processing
    - Speaker embedding extraction
    - Multi-speaker detection
    - Silence tracking with configurable threshold
    """

    def __init__(self, config: DiarizationConfig):
        self.config = config
        self.state = DiarizationState()
        self._init_models()

    def _init_models(self):
        """Initialize pyannote models with HuggingFace token."""
        try:
            from pyannote.audio import Model

            logger.info("Loading pyannote models...")

            # Determine device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")

            # VAD/Segmentation model for speech detection
            self.vad_model = Model.from_pretrained(
                "pyannote/segmentation-3.0",
                use_auth_token=self.config.hf_token
            )
            self.vad_model.to(self.device)
            self.vad_model.eval()

            # Speaker embedding model
            self.embedding_model = Model.from_pretrained(
                "pyannote/embedding",
                use_auth_token=self.config.hf_token
            )
            self.embedding_model.to(self.device)
            self.embedding_model.eval()

            logger.info("Pyannote models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load pyannote models: {e}")
            logger.warning("Make sure HF_TOKEN is set and you've accepted model terms")
            logger.warning("Visit: https://huggingface.co/pyannote/segmentation-3.0")
            raise

    def process_audio_chunk(
        self,
        audio: np.ndarray,
        sample_rate: int,
        current_time: float
    ) -> Tuple[bool, int, List[str]]:
        """
        Process an audio chunk and return diarization info.

        Args:
            audio: Audio samples as numpy array
            sample_rate: Sample rate of the audio
            current_time: Current timestamp in seconds

        Returns:
            is_speech: bool - whether speech is detected
            speaker_count: int - number of speakers in current window
            speaker_ids: List[str] - identified speakers
        """
        # Resample to processing sample rate if needed
        if sample_rate != self.config.processing_sample_rate:
            audio = librosa.resample(
                audio.astype(np.float32),
                orig_sr=sample_rate,
                target_sr=self.config.processing_sample_rate
            )

        # Add to buffer
        self.state.audio_buffer.extend(audio.tolist())

        # Check for speech activity
        is_speech = self._detect_speech(audio)

        if is_speech:
            self.state.last_speech_time = current_time
            self.state.silence_start_time = None
            self.state.has_had_speech = True

            # Extract speaker embedding and identify
            speaker_ids = self._identify_speakers(audio)
            self.state.current_speakers = speaker_ids
            self.state.is_multi_speaker = len(speaker_ids) > 1

            if self.state.is_multi_speaker:
                logger.debug(f"Multi-speaker detected: {speaker_ids}")

            return True, len(speaker_ids), speaker_ids
        else:
            # Track silence
            if self.state.silence_start_time is None and self.state.has_had_speech:
                self.state.silence_start_time = current_time
                logger.debug(f"Silence started at {current_time:.2f}s")

            return False, 0, []

    def _detect_speech(self, audio: np.ndarray) -> bool:
        """Use pyannote VAD to detect speech."""
        try:
            with torch.no_grad():
                # Ensure audio is the right shape: (batch, samples)
                if audio.ndim == 1:
                    audio = audio.reshape(1, -1)

                audio_tensor = torch.from_numpy(audio).float().to(self.device)

                # Get speech probabilities from segmentation model
                output = self.vad_model(audio_tensor)

                # Output shape is (batch, frames, classes) where classes include speech
                # Take mean probability across frames
                speech_prob = output.mean().item()

                return speech_prob > self.config.vad_threshold

        except Exception as e:
            logger.warning(f"VAD detection error: {e}")
            # Fallback to simple RMS-based detection
            rms = np.sqrt(np.mean(audio**2))
            return rms > 0.02

    def _identify_speakers(self, audio: np.ndarray) -> List[str]:
        """Extract speaker embedding and match to known speakers."""
        try:
            with torch.no_grad():
                # Ensure audio is the right shape
                if audio.ndim == 1:
                    audio = audio.reshape(1, -1)

                audio_tensor = torch.from_numpy(audio).float().to(self.device)

                # Get speaker embedding
                embedding = self.embedding_model(audio_tensor)
                embedding = embedding.cpu().numpy().flatten()

            # Compare with known speakers
            best_match = None
            best_similarity = -1

            for speaker_id, known_embedding in self.state.known_speakers.items():
                # Cosine similarity
                similarity = np.dot(embedding, known_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(known_embedding) + 1e-8
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = speaker_id

            # Check if similarity is above threshold
            if best_match and best_similarity >= self.config.speaker_similarity_threshold:
                logger.debug(f"Matched speaker {best_match} (similarity: {best_similarity:.3f})")
                return [best_match]

            # New speaker detected
            new_id = f"SPEAKER_{len(self.state.known_speakers):02d}"
            self.state.known_speakers[new_id] = embedding
            logger.info(f"New speaker detected: {new_id}")
            return [new_id]

        except Exception as e:
            logger.warning(f"Speaker identification error: {e}")
            return ["SPEAKER_00"]

    def get_silence_duration(self, current_time: float) -> float:
        """Get how long silence has been detected."""
        if self.state.silence_start_time is None:
            return 0.0
        return current_time - self.state.silence_start_time

    def should_ai_respond(self, current_time: float) -> bool:
        """
        Determine if AI should respond based on:
        1. Silence duration >= configured threshold (default 5 seconds)
        2. No multi-speaker activity recently
        3. There has been speech before (not startup silence)
        """
        if not self.state.has_had_speech:
            return False

        silence_duration = self.get_silence_duration(current_time)

        should_respond = (
            silence_duration >= self.config.silence_threshold_seconds
            and not self.state.is_multi_speaker
        )

        if should_respond:
            logger.info(
                f"AI should respond: silence={silence_duration:.1f}s, "
                f"multi_speaker={self.state.is_multi_speaker}"
            )

        return should_respond

    def get_known_speakers(self) -> List[str]:
        """Get list of known speaker IDs."""
        return list(self.state.known_speakers.keys())

    def get_speaker_count(self) -> int:
        """Get total number of known speakers."""
        return len(self.state.known_speakers)

    def reset(self):
        """Reset state for new conversation."""
        logger.info("Resetting diarization state")
        self.state = DiarizationState()

    def reset_silence_timer(self):
        """Reset just the silence timer (after AI responds)."""
        self.state.silence_start_time = None
        self.state.is_multi_speaker = False
