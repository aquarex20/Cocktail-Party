"""Main diarization interface combining embedding extraction and clustering."""

import logging
from typing import Tuple, Optional
import numpy as np

from .embedding_extractor import SpeakerEmbeddingExtractor
from .online_clusterer import OnlineSpeakerClusterer
from .utils import validate_audio

logger = logging.getLogger(__name__)


class SpeakerDiarizer:
    """
    High-level interface for speaker diarization.

    Combines embedding extraction and online clustering into a single
    easy-to-use class for post-utterance speaker identification.

    Example usage:
        diarizer = SpeakerDiarizer()

        # Process utterances as they come in
        speaker1 = diarizer.identify_speaker(audio1, sample_rate=48000)
        speaker2 = diarizer.identify_speaker(audio2, sample_rate=48000)

        # speaker1 might be "SPEAKER_0"
        # speaker2 might be "SPEAKER_0" (same person) or "SPEAKER_1" (new person)

        # Reset for new conversation
        diarizer.reset_conversation()

    Attributes:
        FALLBACK_SPEAKER: Default speaker ID returned on errors
    """

    FALLBACK_SPEAKER = "UNKNOWN_SPEAKER"
    MIN_RELIABLE_DURATION = 1.5  # Minimum seconds for reliable speaker ID

    def __init__(
        self,
        similarity_threshold: float = 0.78,
        max_speakers: int = 10,
        device: Optional[str] = None,
        model_cache_dir: Optional[str] = None,
        min_audio_duration: float = 0.5  # Minimum to even attempt diarization
    ):
        """
        Initialize the diarizer.

        Args:
            similarity_threshold: Threshold for same-speaker detection (0.75-0.85)
                                  Higher = stricter, more unique speakers detected
            max_speakers: Maximum speakers to track simultaneously
            device: Compute device ("cpu", "cuda", "mps", or None for auto)
            model_cache_dir: Directory to cache the SpeechBrain model
                             Defaults to ~/.cache/speechbrain/spkrec-ecapa-voxceleb
            min_audio_duration: Minimum audio duration in seconds (default 0.1s)

        Note:
            First initialization downloads the model (~100MB) if not cached.
            Subsequent initializations use the cached model.
        """
        self.min_audio_duration = min_audio_duration
        self._initialized = False

        try:
            self._extractor = SpeakerEmbeddingExtractor(
                device=device,
                save_dir=model_cache_dir
            )
            self._clusterer = OnlineSpeakerClusterer(
                similarity_threshold=similarity_threshold,
                max_speakers=max_speakers
            )
            self._initialized = True
            logger.info("SpeakerDiarizer initialized successfully")
            print("[DIARIZATION] Initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SpeakerDiarizer: {e}")
            print(f"[DIARIZATION] FAILED TO INITIALIZE: {e}")
            import traceback
            traceback.print_exc()
            self._extractor = None
            self._clusterer = OnlineSpeakerClusterer(
                similarity_threshold=similarity_threshold,
                max_speakers=max_speakers
            )

    def identify_speaker(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> str:
        """
        Identify the speaker of an audio utterance.

        Args:
            audio: Audio data (int16 or float32 numpy array)
            sample_rate: Sample rate of the audio in Hz

        Returns:
            Speaker ID string (e.g., "SPEAKER_0", "SPEAKER_1")
            Returns FALLBACK_SPEAKER on errors

        Note:
            The same speaker should get the same ID across multiple calls,
            as long as reset_conversation() is not called.
        """
        if not self._initialized:
            logger.warning("Diarizer not initialized, returning fallback speaker")
            return self.FALLBACK_SPEAKER

        try:
            # Validate input
            if not validate_audio(audio, sample_rate, self.min_audio_duration):
                logger.warning(
                    f"Invalid audio: len={len(audio) if audio is not None else 0}, "
                    f"sr={sample_rate}, min_dur={self.min_audio_duration}s"
                )
                return self._get_last_speaker_or_fallback()

            # Check if audio is long enough for reliable identification
            audio_duration = len(audio) / sample_rate
            is_short_audio = audio_duration < self.MIN_RELIABLE_DURATION

            # Extract embedding
            embedding = self._extractor.extract_embedding(audio, sample_rate)

            # For short audio, use a lower threshold to avoid creating spurious new speakers
            if is_short_audio and self._clusterer.get_speaker_count() > 0:
                # Temporarily lower threshold for short audio
                original_threshold = self._clusterer.similarity_threshold
                self._clusterer.similarity_threshold = original_threshold * 0.7  # 30% more lenient
                print(f"[DIARIZATION] Short audio ({audio_duration:.1f}s), using lower threshold: {self._clusterer.similarity_threshold:.2f}")

                speaker_id = self._clusterer.identify_speaker(embedding)

                # Restore original threshold
                self._clusterer.similarity_threshold = original_threshold
            else:
                speaker_id = self._clusterer.identify_speaker(embedding)

            self._last_speaker = speaker_id
            return speaker_id

        except Exception as e:
            logger.warning(f"Speaker identification failed: {e}")
            return self._get_last_speaker_or_fallback()

    def _get_last_speaker_or_fallback(self) -> str:
        """Return the last identified speaker or fallback."""
        if hasattr(self, '_last_speaker') and self._last_speaker:
            return self._last_speaker
        return self.FALLBACK_SPEAKER

    def identify_speaker_with_confidence(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> Tuple[str, float, bool]:
        """
        Identify speaker with confidence score and new-speaker indication.

        Args:
            audio: Audio data (int16 or float32 numpy array)
            sample_rate: Sample rate of the audio in Hz

        Returns:
            Tuple of:
                - speaker_id: Speaker ID string
                - confidence: Similarity score (0-1 for existing speakers, 1.0 for new)
                - is_new_speaker: True if this is a newly created speaker profile

        Note:
            Confidence represents cosine similarity with the matched speaker's
            centroid. For new speakers, confidence is 1.0 (perfect match with self).
        """
        if not self._initialized:
            return self._get_last_speaker_or_fallback(), 0.0, False

        try:
            if not validate_audio(audio, sample_rate, self.min_audio_duration):
                return self._get_last_speaker_or_fallback(), 0.0, False

            # Check if audio is long enough for reliable identification
            audio_duration = len(audio) / sample_rate
            is_short_audio = audio_duration < self.MIN_RELIABLE_DURATION

            embedding = self._extractor.extract_embedding(audio, sample_rate)

            # For short audio, use a lower threshold
            if is_short_audio and self._clusterer.get_speaker_count() > 0:
                original_threshold = self._clusterer.similarity_threshold
                self._clusterer.similarity_threshold = original_threshold * 0.7
                print(f"[DIARIZATION] Short audio ({audio_duration:.1f}s), using lower threshold: {self._clusterer.similarity_threshold:.2f}")

                speaker_id, confidence, is_new = self._clusterer.identify_speaker_with_confidence(embedding)

                self._clusterer.similarity_threshold = original_threshold
            else:
                speaker_id, confidence, is_new = self._clusterer.identify_speaker_with_confidence(embedding)

            self._last_speaker = speaker_id
            return speaker_id, confidence, is_new

        except Exception as e:
            logger.warning(f"Speaker identification failed: {e}")
            return self._get_last_speaker_or_fallback(), 0.0, False

    def reset_conversation(self) -> None:
        """
        Reset speaker tracking for a new conversation.

        Clears all speaker profiles. Call this when starting a new
        conversation where previous speaker identities should not persist.
        """
        if self._clusterer:
            self._clusterer.reset()
        logger.info("Conversation reset - all speaker profiles cleared")

    @property
    def speaker_count(self) -> int:
        """Number of unique speakers identified so far."""
        if self._clusterer:
            return self._clusterer.get_speaker_count()
        return 0

    def get_speaker_stats(self) -> dict:
        """
        Get statistics about identified speakers.

        Returns:
            Dictionary mapping speaker_id to utterance_count
        """
        if self._clusterer:
            return self._clusterer.get_speaker_stats()
        return {}

    def is_ready(self) -> bool:
        """
        Check if the diarizer is initialized and ready.

        Returns:
            True if ready to process audio, False otherwise
        """
        return self._initialized and self._extractor is not None

    def set_similarity_threshold(self, threshold: float) -> None:
        """
        Update the similarity threshold for speaker matching.

        Args:
            threshold: New threshold value (0.0 to 1.0)
                       Higher = stricter matching, more speakers
                       Lower = looser matching, fewer speakers
        """
        if self._clusterer:
            self._clusterer.similarity_threshold = threshold
            logger.info(f"Similarity threshold updated to {threshold}")

    def export_state(self) -> dict:
        """
        Export the current state for persistence.

        Returns:
            Dictionary containing serializable state that can be
            restored with import_state()
        """
        if self._clusterer:
            return self._clusterer.export_state()
        return {}

    def import_state(self, state: dict) -> None:
        """
        Import a previously saved state.

        Args:
            state: Dictionary from export_state()

        Note:
            This allows persisting speaker profiles across sessions.
        """
        if self._clusterer:
            self._clusterer.import_state(state)
