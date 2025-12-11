"""Utility functions for audio preprocessing in speaker diarization."""

import numpy as np
from scipy.signal import resample_poly
from typing import Union


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = 16000
) -> np.ndarray:
    """
    Resample audio to target sample rate using polyphase filtering.

    Args:
        audio: Input audio array (any dtype, will be converted to float32)
        orig_sr: Original sample rate in Hz
        target_sr: Target sample rate in Hz (default 16kHz for SpeechBrain)

    Returns:
        Resampled audio as float32 numpy array
    """
    if orig_sr == target_sr:
        return audio.astype(np.float32) if audio.dtype != np.float32 else audio

    # Convert to float32 for processing
    audio_float = audio.astype(np.float32) if audio.dtype != np.float32 else audio

    # Use scipy's polyphase resampler for quality
    # Find GCD to simplify the ratio
    from math import gcd
    g = gcd(target_sr, orig_sr)
    up = target_sr // g
    down = orig_sr // g

    resampled = resample_poly(audio_float, up, down)
    return resampled.astype(np.float32)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to float32 in range [-1, 1].

    Handles both int16 and float32 inputs.

    Args:
        audio: Input audio array (int16 or float32)

    Returns:
        Normalized float32 audio in [-1, 1] range
    """
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.float32:
        # Check if already normalized
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            return audio / max_val
        return audio
    else:
        # Try to convert and normalize
        audio_float = audio.astype(np.float32)
        max_val = np.abs(audio_float).max()
        if max_val > 1.0:
            return audio_float / max_val
        return audio_float


def int16_to_float32(audio: np.ndarray) -> np.ndarray:
    """
    Convert int16 audio to float32 in [-1, 1] range.

    Args:
        audio: Input audio as int16 numpy array

    Returns:
        Audio as float32 in [-1, 1] range
    """
    return audio.astype(np.float32) / 32768.0


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    """
    Convert float32 audio ([-1, 1] range) to int16.

    Args:
        audio: Input audio as float32 numpy array in [-1, 1] range

    Returns:
        Audio as int16 numpy array
    """
    # Clip to prevent overflow
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """
    Ensure audio is mono (1D array).

    If stereo, averages channels. If already mono, returns as-is.

    Args:
        audio: Input audio array (1D or 2D)

    Returns:
        Mono audio as 1D numpy array
    """
    if audio.ndim == 1:
        return audio
    elif audio.ndim == 2:
        # Average channels if stereo
        if audio.shape[0] == 2:  # (channels, samples)
            return np.mean(audio, axis=0).astype(audio.dtype)
        elif audio.shape[1] == 2:  # (samples, channels)
            return np.mean(audio, axis=1).astype(audio.dtype)
        else:
            # Take first channel
            return audio[0] if audio.shape[0] < audio.shape[1] else audio[:, 0]
    else:
        raise ValueError(f"Unexpected audio shape: {audio.shape}")


def validate_audio(
    audio: np.ndarray,
    sample_rate: int,
    min_duration_sec: float = 0.1
) -> bool:
    """
    Validate audio meets minimum requirements for speaker embedding.

    Args:
        audio: Audio numpy array
        sample_rate: Sample rate in Hz
        min_duration_sec: Minimum duration in seconds (default 0.1s)

    Returns:
        True if audio is valid, False otherwise
    """
    if audio is None or audio.size == 0:
        return False

    if sample_rate <= 0:
        return False

    min_samples = int(sample_rate * min_duration_sec)
    if len(audio) < min_samples:
        return False

    return True
