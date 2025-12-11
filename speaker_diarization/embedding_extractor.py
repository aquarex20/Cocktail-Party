"""Speaker embedding extraction using SpeechBrain ECAPA-TDNN model."""

import os
import logging
from typing import Optional, Dict
import numpy as np

# Patch torchaudio before importing speechbrain to avoid compatibility issues
try:
    import torchaudio
    # Fix for newer torchaudio versions that removed list_audio_backends
    if not hasattr(torchaudio, 'list_audio_backends'):
        torchaudio.list_audio_backends = lambda: ['soundfile', 'sox']
except ImportError:
    pass

try:
    import torch
    import torchaudio.transforms as T
    from speechbrain.inference.speaker import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False

from .utils import normalize_audio, ensure_mono

logger = logging.getLogger(__name__)


class SpeakerEmbeddingExtractor:
    """
    Extracts speaker embeddings from audio using SpeechBrain's ECAPA-TDNN model.

    The model expects 16kHz mono audio. This class handles conversion from
    other sample rates and formats automatically.

    Attributes:
        TARGET_SAMPLE_RATE: Expected sample rate for the model (16000 Hz)
        EMBEDDING_DIM: Dimensionality of output embeddings (192)
    """

    TARGET_SAMPLE_RATE = 16000
    EMBEDDING_DIM = 192

    def __init__(
        self,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        save_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the embedding extractor.

        Args:
            model_source: HuggingFace model identifier or local path
            save_dir: Directory to cache the downloaded model.
                      Defaults to ~/.cache/speechbrain/spkrec-ecapa-voxceleb
            device: Compute device ("cpu", "cuda", "mps", or None for auto-detect)

        Raises:
            ImportError: If speechbrain, torch, or torchaudio are not installed
            RuntimeError: If model loading fails
        """
        if not SPEECHBRAIN_AVAILABLE:
            raise ImportError(
                "SpeechBrain dependencies not installed. "
                "Install with: pip install speechbrain torch torchaudio"
            )

        # Set up device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Set up cache directory
        if save_dir is None:
            save_dir = os.path.expanduser("~/.cache/speechbrain/spkrec-ecapa-voxceleb")
        else:
            save_dir = os.path.expanduser(save_dir)

        logger.info(f"Loading SpeechBrain model on device: {self.device}")

        try:
            self.classifier = EncoderClassifier.from_hparams(
                source=model_source,
                savedir=save_dir,
                run_opts={"device": self.device}
            )
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to load SpeechBrain model: {e}")
            self._initialized = False
            raise RuntimeError(f"Failed to load speaker embedding model: {e}")

        # Cache for resamplers at different sample rates
        self._resamplers: Dict[int, T.Resample] = {}

    def extract_embedding(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Extract speaker embedding from audio.

        Args:
            audio: Audio data as numpy array (int16 or float32, mono or stereo)
            sample_rate: Sample rate of the input audio in Hz

        Returns:
            numpy array of shape (192,) - the speaker embedding

        Raises:
            ValueError: If audio is empty or invalid
            RuntimeError: If model is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Embedding extractor not initialized")

        if audio is None or len(audio) == 0:
            raise ValueError("Empty audio input")

        # Preprocess audio to tensor
        audio_tensor = self._preprocess_audio(audio, sample_rate)

        # For short audio, pad to at least 1 second for better embeddings
        min_samples = self.TARGET_SAMPLE_RATE  # 1 second at 16kHz
        if audio_tensor.shape[1] < min_samples:
            padding = min_samples - audio_tensor.shape[1]
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))

        # Extract embedding using SpeechBrain
        with torch.no_grad():
            embedding = self.classifier.encode_batch(audio_tensor)

        # Return as numpy array (192,)
        return embedding.squeeze().cpu().numpy()

    def _preprocess_audio(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> torch.Tensor:
        """
        Preprocess audio to 16kHz mono float32 tensor.

        Args:
            audio: Raw audio array (any format)
            sample_rate: Input sample rate in Hz

        Returns:
            Preprocessed torch tensor ready for the model, shape (1, samples)
        """
        # Ensure mono
        audio = ensure_mono(audio)

        # Normalize to float32 [-1, 1]
        audio = normalize_audio(audio)

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()

        # Ensure 2D: (batch=1, samples)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Resample if needed
        if sample_rate != self.TARGET_SAMPLE_RATE:
            if sample_rate not in self._resamplers:
                self._resamplers[sample_rate] = T.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.TARGET_SAMPLE_RATE
                ).to(self.device)
            audio_tensor = audio_tensor.to(self.device)
            audio_tensor = self._resamplers[sample_rate](audio_tensor)
        else:
            audio_tensor = audio_tensor.to(self.device)

        return audio_tensor

    def is_ready(self) -> bool:
        """Check if the extractor is initialized and ready."""
        return self._initialized

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings."""
        return self.EMBEDDING_DIM
