"""
Conversation Mode Module
------------------------
Shared enum and logic for conversation context selection.
Used by local_voice_chat_advanced (to pick mode + context) and llm_client (to choose prompt).
"""

from dataclasses import dataclass
from enum import Enum

from utilities import extract_last_replies, back_and_forth


class ConversationMode(Enum):
    """How the AI should respond based on recent conversation flow."""

    ALONE = "alone"  # AI spoke last 2 turns with no User reply → re-engage
    BACK_AND_FORTH = "back_and_forth"  # Recent alternating User/AI turns → respond to user
    COCKTAIL_PARTY = "cocktail_party"  # Longer/mixed history → use summary + full transcript


@dataclass
class ConversationContext:
    """Result of mode selection: which mode applies, what context to send, and whether to wait."""

    mode: ConversationMode
    context: str
    wait_before_speaking_sec: float  # 0 = no wait


def compute_conversation_context(conversation: str, summary: str) -> ConversationContext:
    """
    Decide which mode applies and build the context string for the LLM.

    Case assignment (order matters):
    1. ALONE: last 2 replies are AI (no User response) → last 2 replies, prompt to re-engage
    2. BACK_AND_FORTH: last 6 replies alternate User/AI → last 6 replies
    3. COCKTAIL_PARTY: else → summary + full conversation
    """
    replies_1 = extract_last_replies(conversation, 1)
    replies_2 = extract_last_replies(conversation, 2)
    ai_just_spoke = bool(replies_1) and all(r.startswith("AI:") for r in replies_1)
    alone = bool(replies_2) and all(r.startswith("AI:") for r in replies_2)
    is_back_and_forth = back_and_forth(conversation, 6)

    if alone:
        context = "\n".join(replies_2)
        wait = 2.0 if ai_just_spoke else 0.0
        return ConversationContext(ConversationMode.ALONE, context, wait)
    if is_back_and_forth:
        context = "\n".join(extract_last_replies(conversation, 6))
        return ConversationContext(ConversationMode.BACK_AND_FORTH, context, 0.0)
    return ConversationContext(
        ConversationMode.COCKTAIL_PARTY, summary + conversation, 0.0
    )
