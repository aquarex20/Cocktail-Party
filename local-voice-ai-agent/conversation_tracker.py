"""
Tracks full conversation history with speaker attribution.
Provides formatted context for LLM responses.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime

from loguru import logger

from config import ConversationConfig


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    speaker_id: str           # Internal ID (e.g., SPEAKER_00)
    speaker_label: str        # Human-readable name or "AI"
    text: str                 # Transcribed or generated text
    timestamp: datetime = field(default_factory=datetime.now)
    is_ai: bool = False       # Whether this turn is from the AI


class ConversationTracker:
    """
    Maintains the full conversation transcript with speaker attribution.
    Provides context formatting for AI responses.
    """

    def __init__(self, config: Optional[ConversationConfig] = None):
        self.config = config or ConversationConfig()
        self.turns: List[ConversationTurn] = []
        self.speaker_labels: Dict[str, str] = dict(self.config.speaker_labels)

    def add_turn(
        self,
        speaker_id: str,
        text: str,
        is_ai: bool = False
    ) -> None:
        """
        Add a conversation turn.

        Args:
            speaker_id: Internal speaker ID (e.g., SPEAKER_00) or "AI"
            text: The transcribed or generated text
            is_ai: Whether this turn is from the AI
        """
        # Get human-readable label
        if is_ai:
            label = "AI"
        else:
            label = self.speaker_labels.get(speaker_id, speaker_id)

        turn = ConversationTurn(
            speaker_id=speaker_id,
            speaker_label=label,
            text=text.strip(),
            is_ai=is_ai
        )
        self.turns.append(turn)

        logger.debug(f"Added turn: [{label}]: {text[:50]}...")

        # Trim if too long
        if len(self.turns) > self.config.max_history_turns:
            removed = self.turns.pop(0)
            logger.debug(f"Removed oldest turn from [{removed.speaker_label}]")

    def set_speaker_label(self, speaker_id: str, label: str) -> None:
        """
        Set a human-readable label for a speaker.

        Args:
            speaker_id: Internal speaker ID (e.g., SPEAKER_00)
            label: Human-readable name (e.g., "Alice")
        """
        self.speaker_labels[speaker_id] = label
        logger.info(f"Set speaker label: {speaker_id} -> {label}")

    def get_formatted_context(self, max_turns: Optional[int] = None) -> str:
        """
        Get conversation context formatted for LLM.

        Format:
        [Speaker A]: Hello, how are you?
        [Speaker B]: I'm doing well, thanks!
        [AI]: That's great to hear!

        Args:
            max_turns: Maximum number of turns to include (None for all)

        Returns:
            Formatted conversation string
        """
        turns_to_use = self.turns[-max_turns:] if max_turns else self.turns

        if not turns_to_use:
            return "(No conversation yet)"

        lines = []
        for turn in turns_to_use:
            lines.append(f"[{turn.speaker_label}]: {turn.text}")

        return "\n".join(lines)

    def get_recent_context(self, num_turns: int = 10) -> str:
        """Get only the most recent turns."""
        return self.get_formatted_context(max_turns=num_turns)

    def get_recent_topics(self) -> List[str]:
        """
        Extract topics/keywords from recent conversation.
        Simple implementation - could be enhanced with NLP.
        """
        # Get text from recent turns
        recent_text = " ".join(t.text for t in self.turns[-10:] if not t.is_ai)

        if not recent_text:
            return []

        # Basic keyword extraction
        words = recent_text.lower().split()

        # Filter common words
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'i', 'you', 'we', 'they', 'he', 'she', 'it', 'my', 'your',
            'this', 'that', 'these', 'those', 'to', 'of', 'in', 'on',
            'and', 'or', 'but', 'so', 'if', 'then', 'when', 'what',
            'who', 'how', 'why', 'where', 'do', 'does', 'did', 'have',
            'has', 'had', 'can', 'could', 'will', 'would', 'should',
            'just', 'like', 'know', 'think', 'yeah', 'yes', 'no', 'not'
        }

        # Filter and get unique topics
        topics = [
            w for w in words
            if len(w) > 3 and w not in stopwords and w.isalpha()
        ]

        # Return unique topics, preserving order
        seen = set()
        unique_topics = []
        for topic in topics:
            if topic not in seen:
                seen.add(topic)
                unique_topics.append(topic)
                if len(unique_topics) >= 5:
                    break

        return unique_topics

    def get_speaker_summary(self) -> Dict[str, int]:
        """
        Get summary of speaker participation.

        Returns:
            Dict mapping speaker label to number of turns
        """
        speaker_turns: Dict[str, int] = {}
        for turn in self.turns:
            if not turn.is_ai:
                label = turn.speaker_label
                speaker_turns[label] = speaker_turns.get(label, 0) + 1
        return speaker_turns

    def get_last_speaker(self) -> Optional[str]:
        """Get the ID of the last non-AI speaker."""
        for turn in reversed(self.turns):
            if not turn.is_ai:
                return turn.speaker_id
        return None

    def get_turn_count(self) -> int:
        """Get total number of turns."""
        return len(self.turns)

    def get_human_turn_count(self) -> int:
        """Get number of human (non-AI) turns."""
        return sum(1 for t in self.turns if not t.is_ai)

    def get_ai_turn_count(self) -> int:
        """Get number of AI turns."""
        return sum(1 for t in self.turns if t.is_ai)

    def clear(self) -> None:
        """Clear conversation history."""
        self.turns = []
        logger.info("Conversation history cleared")

    def export_transcript(self) -> str:
        """
        Export full transcript with timestamps.

        Returns:
            Formatted transcript string
        """
        lines = ["=== Conversation Transcript ===\n"]

        for turn in self.turns:
            timestamp = turn.timestamp.strftime("%H:%M:%S")
            prefix = "[AI]" if turn.is_ai else f"[{turn.speaker_label}]"
            lines.append(f"{timestamp} {prefix}: {turn.text}")

        lines.append("\n=== End of Transcript ===")
        return "\n".join(lines)
