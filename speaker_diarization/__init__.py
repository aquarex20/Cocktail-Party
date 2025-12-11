"""
Speaker Diarization Module for Cocktail-Party Applications.

This module provides speaker diarization capabilities using SpeechBrain's
ECAPA-TDNN model for embedding extraction and online clustering for
real-time speaker identification.

Main Classes:
    SpeakerDiarizer: High-level interface for speaker identification
    SpeakerEmbeddingExtractor: Low-level embedding extraction
    OnlineSpeakerClusterer: Online clustering for speaker tracking

Example:
    from speaker_diarization import SpeakerDiarizer

    # Initialize (downloads model on first run)
    diarizer = SpeakerDiarizer(
        similarity_threshold=0.78,
        max_speakers=10,
        device="cpu"
    )

    # Identify speakers from audio
    speaker_id = diarizer.identify_speaker(audio_array, sample_rate=48000)
    print(f"Speaker: {speaker_id}")  # e.g., "SPEAKER_0"

    # Get detailed info
    speaker_id, confidence, is_new = diarizer.identify_speaker_with_confidence(
        audio_array, sample_rate=48000
    )

    # Reset for new conversation
    diarizer.reset_conversation()
"""

# Suppress warnings and patch torchaudio before importing speechbrain
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import torchaudio
    if not hasattr(torchaudio, 'list_audio_backends'):
        torchaudio.list_audio_backends = lambda: ['soundfile', 'sox']
except ImportError:
    pass

from .diarizer import SpeakerDiarizer
from .embedding_extractor import SpeakerEmbeddingExtractor
from .online_clusterer import OnlineSpeakerClusterer, SpeakerProfile
from .utils import (
    resample_audio,
    normalize_audio,
    int16_to_float32,
    float32_to_int16,
    ensure_mono,
    validate_audio,
)

__all__ = [
    # Main interface
    "SpeakerDiarizer",
    # Low-level components
    "SpeakerEmbeddingExtractor",
    "OnlineSpeakerClusterer",
    "SpeakerProfile",
    # Utilities
    "resample_audio",
    "normalize_audio",
    "int16_to_float32",
    "float32_to_int16",
    "ensure_mono",
    "validate_audio",
]

__version__ = "0.1.0"
