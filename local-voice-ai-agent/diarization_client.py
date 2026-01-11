"""
Diarization Client Module
-------------------------
Provides speaker diarization using audio embeddings.
Non-blocking, failure-safe, with throttling to prevent blocking the main loop.

Uses a tiered approach:
1. Try resemblyzer (lightweight, accurate speaker verification)
2. Try speechbrain deep learning model (best accuracy but has torchaudio issues)
3. Fall back to MFCC-based features (works without ML dependencies)
"""

import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from loguru import logger
from typing import Optional

# ============================================================================
# GLOBAL STATE
# ============================================================================

_embedding_model = None
_embedding_mode: str = "none"  # "resemblyzer", "speechbrain", "mfcc", or "none"
_speaker_embeddings: dict[str, np.ndarray] = {}  # label -> embedding vector
_speaker_counter: int = 0  # For assigning A, B, C, ... labels
_last_label: Optional[str] = None  # Fallback/sticky label
_last_call_time: float = 0.0  # For throttling
_initialized: bool = False
_executor: Optional[ThreadPoolExecutor] = None

# Per-speaker dialogue storage (for future personality features)
speaker_dialogue: dict[str, list[str]] = {}

# Configuration
THROTTLE_INTERVAL_MS = 500  # Minimum time between diarization calls
TIMEOUT_MS = 500  # Max time to wait for diarization result
SIMILARITY_THRESHOLD = 0.75  # Cosine similarity threshold for speechbrain
RESEMBLYZER_SIMILARITY_THRESHOLD = 0.78  # Threshold for resemblyzer (well-tuned)
MFCC_SIMILARITY_THRESHOLD = 0.80  # Threshold for MFCC (less accurate)
SAMPLE_RATE = 16000


# ============================================================================
# MFCC-BASED FALLBACK EMBEDDING
# ============================================================================

def _compute_mfcc_embedding(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Optional[np.ndarray]:
    """
    Compute a simple MFCC-based speaker embedding.
    This is a fallback when deep learning models aren't available.
    Less accurate but works without external dependencies.
    """
    try:
        from scipy.fft import dct
        from scipy.signal import get_window

        audio = np.asarray(audio, dtype=np.float32).flatten()

        # Skip if too short
        if len(audio) < sample_rate * 0.5:
            return None

        # Pre-emphasis
        pre_emphasis = 0.97
        audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

        # Frame parameters
        frame_size = int(0.025 * sample_rate)  # 25ms
        frame_stride = int(0.01 * sample_rate)  # 10ms
        n_fft = 512
        n_mfcc = 13
        n_mels = 26

        # Framing
        num_frames = 1 + (len(audio) - frame_size) // frame_stride
        if num_frames < 1:
            return None

        frames = np.zeros((num_frames, frame_size))
        for i in range(num_frames):
            start = i * frame_stride
            frames[i] = audio[start:start + frame_size]

        # Windowing
        window = get_window('hamming', frame_size)
        frames *= window

        # FFT
        mag_spec = np.abs(np.fft.rfft(frames, n_fft))
        pow_spec = (1.0 / n_fft) * (mag_spec ** 2)

        # Mel filterbank
        low_freq_mel = 0
        high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

        fbank = np.zeros((n_mels, n_fft // 2 + 1))
        for m in range(1, n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            for k in range(f_m_minus, f_m):
                if f_m != f_m_minus:
                    fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus != f_m:
                    fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

        # Apply filterbank
        filter_banks = np.dot(pow_spec, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)

        # DCT to get MFCCs
        mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]

        # Compute statistics over frames (mean and std)
        mfcc_mean = np.mean(mfcc, axis=0)
        mfcc_std = np.std(mfcc, axis=0)

        # Also add some spectral features
        spectral_centroid = np.mean(np.sum(mag_spec * np.arange(mag_spec.shape[1]), axis=1) / (np.sum(mag_spec, axis=1) + 1e-8))
        spectral_rolloff = np.mean(np.percentile(mag_spec, 85, axis=1))

        # Combine into embedding
        embedding = np.concatenate([mfcc_mean, mfcc_std, [spectral_centroid, spectral_rolloff]])

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    except Exception as e:
        logger.debug(f"MFCC embedding error: {e}")
        return None


# ============================================================================
# INITIALIZATION
# ============================================================================

def init_diarization() -> bool:
    """
    Initialize the speaker embedding model.
    Non-blocking, fails gracefully if model can't be loaded.
    Falls back to simpler methods if deep learning models fail.

    Returns:
        bool: True if initialization succeeded (including fallback), False otherwise
    """
    global _embedding_model, _embedding_mode, _initialized, _executor

    if _initialized:
        return _embedding_mode != "none"

    _initialized = True
    _executor = ThreadPoolExecutor(max_workers=1)

    # Try resemblyzer first (lightweight, designed for speaker verification)
    try:
        from resemblyzer import VoiceEncoder

        logger.info("Loading speaker embedding model (resemblyzer)...")
        _embedding_model = VoiceEncoder()
        _embedding_mode = "resemblyzer"
        logger.info("Speaker embedding model loaded successfully (resemblyzer)")
        return True

    except Exception as e:
        logger.debug(f"Resemblyzer failed: {e}")

    # Try speechbrain (best accuracy but may have torchaudio issues)
    try:
        from speechbrain.inference.speaker import EncoderClassifier

        logger.info("Loading speaker embedding model (speechbrain/spkrec-ecapa-voxceleb)...")
        _embedding_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/speechbrain_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"}
        )
        _embedding_mode = "speechbrain"
        logger.info("Speaker embedding model loaded successfully (speechbrain)")
        return True

    except Exception as e:
        logger.debug(f"Speechbrain failed: {e}")

    # Try MFCC fallback
    try:
        # Test that scipy is available for MFCC
        from scipy.fft import dct
        from scipy.signal import get_window

        # Test with dummy audio
        test_audio = np.random.randn(SAMPLE_RATE).astype(np.float32)
        test_embedding = _compute_mfcc_embedding(test_audio)

        if test_embedding is not None:
            _embedding_mode = "mfcc"
            logger.info("Using MFCC-based speaker embeddings (fallback mode)")
            return True

    except Exception as e:
        logger.debug(f"MFCC fallback failed: {e}")

    # All methods failed
    _embedding_mode = "none"
    logger.warning("Diarization disabled. Transcripts will use 'User:' without labels.")
    return False


def shutdown_diarization():
    """Cleanup resources."""
    global _executor
    if _executor:
        _executor.shutdown(wait=False)
        _executor = None


# ============================================================================
# SPEAKER LABEL FUNCTIONS
# ============================================================================

def _compute_embedding(audio_chunk: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute speaker embedding for an audio chunk.
    Uses the appropriate method based on _embedding_mode.
    Internal function, should be called within executor.
    """
    if _embedding_mode == "none":
        return None

    try:
        # Ensure audio is float32 and 1D
        audio = np.asarray(audio_chunk, dtype=np.float32).flatten()

        # Skip if audio is too short (< 0.5 seconds)
        if len(audio) < SAMPLE_RATE * 0.5:
            return None

        if _embedding_mode == "resemblyzer":
            # Resemblyzer expects audio at 16kHz, normalized
            from resemblyzer import preprocess_wav

            # Preprocess (normalize, trim silence)
            try:
                processed = preprocess_wav(audio, source_sr=SAMPLE_RATE)
                if len(processed) < SAMPLE_RATE * 0.3:  # Too short after processing
                    return None
            except Exception:
                # If preprocessing fails, use raw audio normalized
                processed = audio / (np.max(np.abs(audio)) + 1e-8)

            # Get embedding (256-dimensional vector)
            embedding = _embedding_model.embed_utterance(processed)
            return embedding

        elif _embedding_mode == "speechbrain":
            # SpeechBrain expects torch tensor
            import torch
            audio_tensor = torch.tensor(audio).unsqueeze(0)

            # Get embedding
            embedding = _embedding_model.encode_batch(audio_tensor)
            return embedding.squeeze().cpu().numpy()

        elif _embedding_mode == "mfcc":
            # Use MFCC-based embedding
            return _compute_mfcc_embedding(audio, SAMPLE_RATE)

        return None

    except Exception as e:
        logger.debug(f"Error computing speaker embedding: {e}")
        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def _int_to_label(n: int) -> str:
    """Convert integer to letter label: 0->A, 1->B, ..., 25->Z, 26->AA, etc."""
    result = ""
    while True:
        result = chr(ord('A') + n % 26) + result
        n = n // 26 - 1
        if n < 0:
            break
    return result


def _find_or_create_speaker(embedding: np.ndarray) -> str:
    """
    Find matching speaker or create new one.
    Returns speaker label (A, B, C, ...).
    """
    global _speaker_counter, _speaker_embeddings

    # Use appropriate threshold based on embedding mode
    if _embedding_mode == "resemblyzer":
        threshold = RESEMBLYZER_SIMILARITY_THRESHOLD
    elif _embedding_mode == "mfcc":
        threshold = MFCC_SIMILARITY_THRESHOLD
    else:
        threshold = SIMILARITY_THRESHOLD

    # Compare with known speakers
    best_match = None
    best_similarity = 0.0

    for label, known_embedding in _speaker_embeddings.items():
        similarity = _cosine_similarity(embedding, known_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = label

    # If similarity above threshold, return existing speaker
    if best_match and best_similarity >= threshold:
        # Update embedding with exponential moving average for robustness
        _speaker_embeddings[best_match] = (
            0.8 * _speaker_embeddings[best_match] + 0.2 * embedding
        )
        return best_match

    # Create new speaker
    new_label = _int_to_label(_speaker_counter)
    _speaker_counter += 1
    _speaker_embeddings[new_label] = embedding
    logger.debug(f"New speaker detected: {new_label}")
    return new_label


def get_speaker_label(audio_chunk: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Optional[str]:
    """
    Get speaker label for an audio chunk.

    Non-blocking with timeout. Returns cached label if throttled.
    Returns None on failure (never raises).

    Args:
        audio_chunk: Audio data as numpy array (can be tuple from fastrtc)
        sample_rate: Sample rate of audio (default 16000)

    Returns:
        Speaker label (A, B, C, ...) or None if unavailable
    """
    global _last_label, _last_call_time

    # Handle fastrtc tuple format (sample_rate, audio_data)
    if isinstance(audio_chunk, tuple):
        sample_rate, audio_chunk = audio_chunk

    # Check if initialized and enabled
    if not _initialized or _embedding_mode == "none":
        return None

    # Throttle check
    current_time = time.time() * 1000  # Convert to ms
    if current_time - _last_call_time < THROTTLE_INTERVAL_MS:
        return _last_label

    _last_call_time = current_time

    try:
        # Submit to executor with timeout
        if _executor is None:
            return _last_label

        future = _executor.submit(_compute_embedding, audio_chunk)
        embedding = future.result(timeout=TIMEOUT_MS / 1000)

        if embedding is None:
            return _last_label

        # Find or create speaker
        label = _find_or_create_speaker(embedding)
        _last_label = label
        return label

    except FuturesTimeoutError:
        logger.debug("Diarization timed out, using last label")
        return _last_label
    except Exception as e:
        logger.debug(f"Diarization error: {e}")
        return _last_label


def record_speaker_text(label: str, text: str):
    """
    Record text for a speaker (side storage for future personality features).

    Args:
        label: Speaker label (A, B, C, ...)
        text: Transcribed text from this speaker
    """
    if label is None:
        return

    if label not in speaker_dialogue:
        speaker_dialogue[label] = []

    speaker_dialogue[label].append(text)


def get_speaker_history(label: str) -> list[str]:
    """Get all recorded text for a speaker."""
    return speaker_dialogue.get(label, [])


def get_all_speakers() -> list[str]:
    """Get list of all detected speaker labels."""
    return list(_speaker_embeddings.keys())


def reset_speakers():
    """Reset all speaker data (useful for new sessions)."""
    global _speaker_embeddings, _speaker_counter, _last_label, speaker_dialogue
    _speaker_embeddings = {}
    _speaker_counter = 0
    _last_label = None
    speaker_dialogue = {}
    logger.info("Speaker data reset")
