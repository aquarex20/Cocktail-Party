"""
Configuration dataclasses for multi-speaker voice AI agent.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict
import os


@dataclass
class DiarizationConfig:
    """Configuration for speaker diarization engine."""

    # HuggingFace token for pyannote models
    # Get from: https://huggingface.co/settings/tokens
    hf_token: Optional[str] = field(
        default_factory=lambda: os.environ.get("HF_TOKEN", "")
    )

    # Audio parameters
    input_sample_rate: int = 48000
    processing_sample_rate: int = 16000  # pyannote works at 16kHz

    # Diarization parameters
    embedding_window_ms: int = 1500  # Window for speaker embedding
    embedding_step_ms: int = 500     # Step between windows
    min_speakers: int = 1
    max_speakers: int = 4

    # Speaker identification
    speaker_similarity_threshold: float = 0.7  # Cosine similarity threshold

    # Silence detection - AI responds after this duration
    silence_threshold_seconds: float = 5.0

    # Speech detection
    min_speech_duration_ms: int = 250
    vad_threshold: float = 0.5  # VAD probability threshold

    # Multi-speaker detection window
    multi_speaker_window_ms: int = 2000

    # Response behavior
    respond_only_on_silence: bool = True


@dataclass
class ConversationConfig:
    """Configuration for conversation tracking."""

    # Maximum number of conversation turns to keep
    max_history_turns: int = 50

    # Maximum tokens for LLM context (approximate)
    max_context_tokens: int = 2000

    # Speaker labels for human-readable names
    # e.g., {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}
    speaker_labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """Configuration for LLM responses."""

    # Ollama endpoint
    ollama_url: str = "http://127.0.0.1:11434/api/chat"

    # Model to use
    model: str = "gemma3:4b"

    # Maximum tokens in response
    max_response_tokens: int = 150

    # Temperature for response generation
    temperature: float = 0.7

    # System prompt for contextual responses
    system_prompt: str = """You are an AI participant in a cocktail party conversation.
You've been listening to the conversation and now there's a natural pause (5 seconds of silence).

Your role is to:
1. Provide ONE interesting insight, fact, or perspective related to what was just discussed
2. Ask ONE thought-provoking follow-up question to encourage continued conversation

Keep your response concise (2-3 sentences max).
Do NOT use emojis or special characters as your response will be spoken aloud.
Be conversational and natural, not formal.

Here is the conversation so far:
"""


@dataclass
class AudioConfig:
    """Configuration for audio I/O."""

    # Input device (None for default, or device name like "BlackHole 2ch")
    input_device: Optional[str] = "BlackHole 2ch"

    # Output device (None for default, or device index)
    output_device: Optional[int] = 1

    # Sample rates
    input_sample_rate: int = 48000
    output_sample_rate: int = 24000

    # Frame size for output
    output_frame_size: int = 480
