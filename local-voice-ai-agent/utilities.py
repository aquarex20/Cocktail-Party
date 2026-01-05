"""
Utilities Module
-----------------
Contains all utility functions for the local voice AI agent.
"""

# ============================================================================
# FUNCTIONS
# ============================================================================

#Extract everything after "Transcript:" from the conversation
def extract_transcript(conversation: str) -> str:
    marker = "Transcript:\n"
    idx = conversation.find(marker)
    if idx == -1:
        return ""
    return conversation[idx + len(marker):]

import re

#Extract the last n replies from the transcript
def extract_last_replies(transcript: str, n: int = 4) -> list[str]:
    pattern = r'(?:^|\n)(User:|AI:)(.*?)(?=\n(?:User:|AI:)|$)'
    matches = re.findall(pattern, transcript, flags=re.DOTALL)

    replies = [speaker + content.strip() for speaker, content in matches]
    return replies[-n:]

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

print(extract_last_replies(example_conversation, 3))
