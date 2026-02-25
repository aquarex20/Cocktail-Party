"""
Audio resampling for STT input. TTS may produce 22.05k/24k; STT expects 16k.
Uses scipy when available for better quality, else simple linear interpolation.
"""

import numpy as np


def linear_resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample audio from sr_in to sr_out. Returns float32 mono."""
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32).flatten()
    x = np.asarray(x, dtype=np.float32).flatten()
    n_in = len(x)
    n_out = int(round(n_in * sr_out / sr_in))
    if n_in < 2 or n_out < 2:
        return np.zeros((0,), dtype=np.float32)
    try:
        from scipy import signal
        return signal.resample(x, n_out).astype(np.float32)
    except ImportError:
        t_in = np.linspace(0.0, 1.0, n_in, endpoint=False)
        t_out = np.linspace(0.0, 1.0, n_out, endpoint=False)
        return np.interp(t_out, t_in, x).astype(np.float32)
