import gradio as gr
from kokoro_onnx import Kokoro
import sounddevice as sd
import asyncio
import numpy as np
from pathlib import Path
import threading

#comment this if you havent used Script Config for your party
sd.default.device = (None, "Script Config")  # (input, output)

# Global flag for stopping playback
stop_playback = threading.Event()

# Find kokoro model files
def find_kokoro_models():
    """Find kokoro model files in current directory or party directory."""
    # Check current directory
    current_dir = Path(__file__).parent
    party_dir = current_dir.parent / "party"
    
    locations = [
        (current_dir / "kokoro-v1.0.onnx", current_dir / "voices-v1.0.bin"),
        (party_dir / "kokoro-v1.0.onnx", party_dir / "voices-v1.0.bin"),
        (Path("kokoro-v1.0.onnx"), Path("voices-v1.0.bin")),
    ]
    
    for model_path, voices_path in locations:
        if model_path.exists() and voices_path.exists():
            return str(model_path), str(voices_path)
    
    return None, None

model_path, voices_path = find_kokoro_models()

if not model_path or not voices_path:
    raise FileNotFoundError(
        "Could not find kokoro model files (kokoro-v1.0.onnx and voices-v1.0.bin).\n"
        "Please download them from: https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
    )

# Initialize kokoro
kokoro = Kokoro(model_path, voices_path)

# Default voice assignments for AIs
DEFAULT_VOICES = {
    "AI1": "if_sara",
    "AI2": "am_santa",
    "AI3": "bf_emma",
    "AI4": "am_adam",
}

# Predefined dialogue options
DIALOGUE_OPTIONS = {
    "Custom (Edit below)": "",
    "Cocktail Party Chat": """AI1: Hello sir, how are you doing?
AI2: Howdy! I'm doing fine—how about you?
AI1: Can't complain…
AI2: Glad to hear that! Wow, this cocktail party is… something.
AI1: It really is. Fancy lights, loud jazz, and drinks that look expensive.
AI2: Speaking of which—confession—I don't like the alcohol here.
AI1: Oh? Not a cocktail person?
AI2: I like cocktails. I just don't like these cocktails.
AI1: That's because they're cheap.
AI2: Cheap?? At this event?
AI1: Yeah. They blew the whole budget on the VR stand.""",
    "Tech Interview": """AI1: So, tell me about your experience with Python.
AI2: I've been coding in Python for about five years now.
AI1: Interesting. What frameworks have you worked with?
AI2: Mostly Django and FastAPI for backend work.
AI1: And what about machine learning?
AI2: I've done some work with PyTorch and scikit-learn.
AI1: Great. Can you describe a challenging project you've worked on?
AI2: Sure! I built a real-time speech recognition system last year.""",
    "Weather Small Talk": """AI1: Beautiful day outside, isn't it?
AI2: Absolutely! The sun is shining perfectly.
AI1: I heard it might rain later though.
AI2: Oh no, I didn't bring my umbrella!
AI1: Don't worry, I have a spare one.
AI2: You're a lifesaver! Thank you so much.""",
    "Restaurant Order": """AI1: Welcome! Are you ready to order?
AI2: Yes, I'll have the grilled salmon please.
AI1: Excellent choice. Any sides with that?
AI2: I'll take the roasted vegetables.
AI1: And to drink?
AI2: Just water for now, thank you.
AI1: Perfect. Your order will be ready in about fifteen minutes.""",
    "Group Discussion (4 AIs)": """AI1: Hey everyone, thanks for joining the meeting!
AI2: Happy to be here. What's on the agenda?
AI3: I think we're discussing the new project timeline.
AI4: Right, we need to finalize the deadlines.
AI1: Exactly. So the first milestone is due next Friday.
AI2: That seems tight. Can we push it back a bit?
AI3: I agree, we need more time for testing.
AI4: What if we do a soft launch first?
AI1: That's actually a great idea.
AI2: I'm on board with that approach.
AI3: Same here. Let's do it!
AI4: Perfect, meeting adjourned!""",
}

# Get available voices for reference
available_voices = kokoro.get_voices()
print(f"Available voices: {available_voices}")

async def play_text_bubble(text, voice, speed=1.0, lang="en-us"):
    """
    Play a single text bubble using kokoro streaming.
    Collects all chunks first, then plays as one continuous audio.
    Returns False if stopped, True if completed.
    """
    if stop_playback.is_set():
        return False

    if not text or not text.strip():
        return True
    
    stream = kokoro.create_stream(
        text.strip(),
        voice=voice,
        speed=speed,
        lang=lang,
    )
    
    # Collect all audio chunks first
    all_samples = []
    final_sample_rate = None
    count = 0
    
    async for samples, sample_rate in stream:
        if stop_playback.is_set():
            return False
        count += 1
        all_samples.append(samples)
        final_sample_rate = sample_rate
    
    # Concatenate and play as one continuous audio
    if all_samples and final_sample_rate:
        full_audio = np.concatenate(all_samples)
        
        # Add a small silence padding at the end to prevent cutoff
        padding = np.zeros(int(final_sample_rate * 0.2), dtype=full_audio.dtype)  # 300ms padding
        full_audio = np.concatenate([full_audio, padding])
        
        print(f"Playing {count} chunks ({len(full_audio)/final_sample_rate:.1f}s) for voice {voice}...")
        
        # Use blocking=True for more reliable playback
        sd.play(full_audio, final_sample_rate, blocking=True)
        
        if stop_playback.is_set():
            sd.stop()
            return False
    
    return True

def play_script(script_text, ai1_voice=None, ai2_voice=None, ai3_voice=None, ai4_voice=None, speed=1.0):
    """
    Parse the script and play each bubble separately with the appropriate voice.
    Lines starting with 'AI1:', 'AI2:', 'AI3:', 'AI4:' will use corresponding voices.
    """
    # Clear stop flag at start
    stop_playback.clear()
    
    if not script_text or not script_text.strip():
        return "Please enter a script to play."
    
    # Build voice mapping
    voices = {
        "AI1": ai1_voice if ai1_voice else DEFAULT_VOICES["AI1"],
        "AI2": ai2_voice if ai2_voice else DEFAULT_VOICES["AI2"],
        "AI3": ai3_voice if ai3_voice else DEFAULT_VOICES["AI3"],
        "AI4": ai4_voice if ai4_voice else DEFAULT_VOICES["AI4"],
    }
    
    # Split script into lines
    lines = script_text.strip().split('\n')
    
    # Collect all text bubbles with their voice assignments
    bubbles = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for AI prefixes (AI1: through AI4:)
        for ai_key in ["AI1", "AI2", "AI3", "AI4"]:
            prefix = f"{ai_key}:"
            if line.startswith(prefix):
                text_to_speak = line[len(prefix):].strip()
                if text_to_speak:
                    bubbles.append((text_to_speak, voices[ai_key]))
                break
    
    if not bubbles:
        return "No valid AI lines found. Use prefixes like 'AI1:', 'AI2:', 'AI3:', or 'AI4:'"
    
    # Play each bubble sequentially
    try:
        status_messages = []
        for i, (text, voice) in enumerate(bubbles, 1):
            if stop_playback.is_set():
                return "\n".join(status_messages) + f"\n\n⏹ Stopped at bubble {i}/{len(bubbles)}"
            status_messages.append(f"Playing bubble {i}/{len(bubbles)}: {voice} - {text[:50]}...")
            completed = asyncio.run(play_text_bubble(text, voice, speed=speed))
            if not completed:
                return "\n".join(status_messages) + f"\n\n⏹ Stopped at bubble {i}/{len(bubbles)}"
        
        return "\n".join(status_messages) + f"\n\n✓ Finished playing all {len(bubbles)} bubbles!"
    
    except Exception as e:
        return f"Error playing script: {str(e)}"

def stop_script():
    """Stop the currently playing script."""
    stop_playback.set()
    sd.stop()
    return "⏹ Playback stopped"

def on_dialogue_select(dialogue_name):
    """Return the script text for the selected dialogue."""
    return DIALOGUE_OPTIONS.get(dialogue_name, "")

# Create Gradio interface
with gr.Blocks(title="Dual Voice Script Player") as demo:
    gr.Markdown("# 🎭 Dual Voice Script Player")
    gr.Markdown("Select a dialogue from the dropdown or write your own. Use **AI1:** or **AI2:** prefixes to indicate which voice should read each line.")
    
    with gr.Row():
        with gr.Column(scale=2):
            # Dialogue selection dropdown
            dialogue_dropdown = gr.Dropdown(
                choices=list(DIALOGUE_OPTIONS.keys()),
                value="Cocktail Party Chat",
                label="📜 Select Dialogue",
                interactive=True
            )
            
            script_input = gr.Textbox(
                label="Script",
                placeholder="AI1: Hello!\nAI2: Hi there!",
                lines=12,
                value=DIALOGUE_OPTIONS["Cocktail Party Chat"]
            )
            
            with gr.Row():
                ai1_voice_dropdown = gr.Dropdown(
                    choices=available_voices,
                    value=DEFAULT_VOICES["AI1"],
                    label="AI1 Voice",
                    interactive=True
                )
                ai2_voice_dropdown = gr.Dropdown(
                    choices=available_voices,
                    value=DEFAULT_VOICES["AI2"],
                    label="AI2 Voice",
                    interactive=True
                )
            
            # Optional AI3 & AI4 voices (collapsible)
            with gr.Accordion("➕ More Voices (AI3, AI4)", open=False):
                with gr.Row():
                    ai3_voice_dropdown = gr.Dropdown(
                        choices=available_voices,
                        value=DEFAULT_VOICES["AI3"],
                        label="AI3 Voice",
                        interactive=True
                    )
                    ai4_voice_dropdown = gr.Dropdown(
                        choices=available_voices,
                        value=DEFAULT_VOICES["AI4"],
                        label="AI4 Voice",
                        interactive=True
                    )
            
            speed_slider = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Speed"
            )
        
        with gr.Column(scale=1):
            with gr.Row():
                play_button = gr.Button("▶️ Play", variant="primary", size="lg")
                stop_button = gr.Button("⏹ Stop", variant="stop", size="lg")
            status_output = gr.Textbox(label="Status", interactive=False, lines=10)
    
    # Connect dialogue selection to script input
    dialogue_dropdown.change(
        fn=on_dialogue_select,
        inputs=[dialogue_dropdown],
        outputs=[script_input]
    )
    
    # Connect the play button
    play_button.click(
        fn=play_script,
        inputs=[script_input, ai1_voice_dropdown, ai2_voice_dropdown, ai3_voice_dropdown, ai4_voice_dropdown, speed_slider],
        outputs=[status_output]
    )
    
    # Connect the stop button
    stop_button.click(
        fn=stop_script,
        inputs=[],
        outputs=[status_output]
    )

if __name__ == "__main__":
    demo.launch()
