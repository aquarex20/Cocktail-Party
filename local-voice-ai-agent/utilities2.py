import re
import numpy as np
import librosa
import onnxruntime as ort

def clean_text_for_tts(text: str) -> str:
    # keep letters, numbers, spaces, and . , ; ! ?
    return re.sub(r'[^A-Za-z0-9 .,;!?]', '', text)


def preprocess_audio(sr, x):
    # (1, N) → (N,)
    if x.ndim == 2:
        x = x.squeeze(0)

    # int16 → float32 normalized
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0

    # resample 48k → 16k
    if sr != 16000:
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
        sr = 16000

    return sr, x

def split_for_tts(
    text: str,
    max_chars: int = 180,
    min_chars: int = 40,
):
    """
    Splits text into speakable chunks.
    - Splits on ., ; , 
    - Also splits when exceeding max_chars
    - Avoids tiny fragments (< min_chars)
    """

    if not text or not text.strip():
        return []

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text.strip())

    # Split while keeping punctuation
    parts = re.split(r"([.,';?!])", text)
    chunks = []
    current = ""

    for part in parts:
        if not part:
            continue

        current += part

        # If we hit punctuation or length limit
        if (
            part in {".", ",", ";"}
            or len(current) >= max_chars
        ):
            if len(current.strip()) >= min_chars:
                chunks.append(current.strip())
                current = ""

    # Remainder
    if current.strip():
        if chunks and len(current.strip()) < min_chars:
            # attach small tail to previous chunk
            chunks[-1] += " " + current.strip()
        else:
            chunks.append(current.strip())

    return chunks
