"""
Audio device listing, device selection, and playback for the local voice chat app.
"""

import numpy as np
from scipy import signal
import sounddevice as sd
from loguru import logger

from voice_config import TARGET_SAMPLE_RATE


def get_device_lists():
    """Return (input_devices, output_devices) as lists of (index, name)."""
    devices = sd.query_devices()
    inputs = []
    outputs = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            inputs.append((i, d["name"]))
        if d["max_output_channels"] > 0:
            outputs.append((i, d["name"]))
    return inputs, outputs


def set_audio_devices(input_device=None, output_device=None):
    """Set sounddevice default device. Accepts name (str) or index (int). Only applies when both are set."""
    if input_device is not None and output_device is not None:
        sd.default.device = (input_device, output_device)
        logger.info(f"Audio devices set: input={input_device!r}, output={output_device!r}")


def play_audio(audio_data, sample_rate):
    """Play audio with resampling to target sample rate."""
    audio_data = np.asarray(audio_data, dtype=np.float32).flatten()
    if audio_data.size == 0:
        return
    if sample_rate != TARGET_SAMPLE_RATE:
        audio_data = signal.resample(audio_data, int(len(audio_data) * TARGET_SAMPLE_RATE / sample_rate))
    sd.play(audio_data, samplerate=TARGET_SAMPLE_RATE, blocking=True)
