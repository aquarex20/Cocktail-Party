"""
Utilities Module
-----------------
Contains all utility functions for the local voice AI agent.
"""

# ============================================================================
# FUNCTIONS
# ============================================================================

#Extract everything after "Transcript:" from the conversation
def extract_transcript(text: str) -> str:
    marker = "Transcript:"
    idx = text.find(marker)
    if idx == -1:
        return ""

    line_end = text.find("\n", idx)
    if line_end == -1:
        return ""

    return text[line_end + 1:]

import re

import re

def extract_last_replies(text: str, n: int = 4) -> list[str]:
    # Matches lines that start with optional whitespace + optional quote + (User|AI) + optional (X) + :
    # Handles both "User: hello" and "User (A): hello" formats
    pattern = r'^[ \t"]*(User|AI)(?:\s*\([^)]+\))?:[ \t"]*(.*)$'
    matches = re.findall(pattern, text, flags=re.MULTILINE)

    replies = []
    for speaker, content in matches:
        # Keep content as-is (except trimming trailing spaces)
        content = content.rstrip()
        # Format as "User: content" or "AI: content" (normalized without label)
        replies.append(f"{speaker}: {content}" if content else f"{speaker}:")

    return replies[-n:]
def back_and_forth(transcript: str, n:int=4) -> str:
    last = extract_last_replies(transcript, n)
    if len(last) < n:
        return False

    prev = None
    for r in last:
        # Extract base speaker, handling both "User: text" and "User (A): text" formats
        speaker_full = r.split(":", 1)[0]
        speaker = speaker_full.split("(")[0].strip()
        if speaker not in ("AI", "User"):
            return False
        if prev is not None and speaker == prev:
            return False
        prev = speaker

    return True

example_conversation = """
Transcript:

User: Hello, how are you?
AI: I'm doing great, thank you! How about you?
User: I'm good, thanks. What's new?
AI: Not much, just working on a new project.
User: That sounds interesting. What is it?
AI: It's a new chatbot that I'm building.
User: Cool! How does it work?
AI: It uses a combination of natural language processing and machine learning to understand user intent and respond appropriately.
User: That sounds like a lot of work.
AI: Yeah, it is. But it's also a lot of fun.
"""

