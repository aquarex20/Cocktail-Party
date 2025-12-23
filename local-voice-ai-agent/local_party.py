import gradio as gr
from kokoro_onnx import Kokoro
import sounddevice as sd
import asyncio
import numpy as np
from pathlib import Path
import sounddevice as sd

sd.default.device = (None, "Script Config")  # (input, output)

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

# Voice assignments for AI1 and AI2
AI1_VOICE = "if_sara"
AI2_VOICE = "am_santa"

# Get available voices for reference
available_voices = kokoro.get_voices()
print(f"Available voices: {available_voices}")

async def play_text_bubble(text, voice, speed=1.0, lang="en-us"):
    """
    Play a single text bubble using kokoro streaming.
    Collects all chunks first, then plays as one continuous audio.
    """

    if not text or not text.strip():
        return
    
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
        
        # Small extra delay to ensure buffer is flushed

def play_script(script_text, ai1_voice=None, ai2_voice=None, speed=1.0):
    """
    Parse the script and play each bubble separately with the appropriate voice.
    Lines starting with 'AI1:' will use AI1 voice, 'AI2:' will use AI2 voice.
    """
    if not script_text or not script_text.strip():
        return "Please enter a script to play.", None
    
    # Use provided voices or defaults
    voice1 = ai1_voice if ai1_voice else AI1_VOICE
    voice2 = ai2_voice if ai2_voice else AI2_VOICE
    
    # Split script into lines
    lines = script_text.strip().split('\n')
    
    # Collect all text bubbles with their voice assignments
    bubbles = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if line starts with AI1: or AI2:
        if line.startswith('AI1:'):
            text_to_speak = line[4:].strip()  # Remove 'AI1:' prefix
            if text_to_speak:
                bubbles.append((text_to_speak, voice1))
        
        elif line.startswith('AI2:'):
            text_to_speak = line[4:].strip()  # Remove 'AI2:' prefix
            if text_to_speak:
                bubbles.append((text_to_speak, voice2))
    
    if not bubbles:
        return "No valid AI1: or AI2: lines found in the script. Please format your script with 'AI1:' or 'AI2:' prefixes.", None
    
    # Play each bubble sequentially
    try:
        status_messages = []
        for i, (text, voice) in enumerate(bubbles, 1):
            status_messages.append(f"Playing bubble {i}/{len(bubbles)}: {voice} - {text[:50]}...")
            asyncio.run(play_text_bubble(text, voice, speed=speed))
        
        return "\n".join(status_messages) + f"\n\n✓ Finished playing all {len(bubbles)} bubbles!", None
    
    except Exception as e:
        return f"Error playing script: {str(e)}", None

# Create Gradio interface
with gr.Blocks(title="Dual Voice Script Player") as demo:
    gr.Markdown("# 🎭 Dual Voice Script Player")
    gr.Markdown("Enter your script below. Use **AI1:** or **AI2:** prefixes to indicate which voice should read each line.")
    gr.Markdown(f"**Available voices:** {', '.join(available_voices[:10])}{'...' if len(available_voices) > 10 else ''}")
    
    with gr.Row():
        with gr.Column(scale=2):
            script_input = gr.Textbox(
                label="Script",
                placeholder="AI1: Hello sir, how are you doing?\nAI2: Howdy! I’m doing fine—how about you?\nAI1: Can’t complain…\nAI2: Glad to hear that! Wow, this cocktail party is… something.\nAI1: It really is. Fancy lights, loud jazz, and drinks that look expensive.\nAI2: Speaking of which—confession—I don’t like the alcohol here.\nAI1: Oh? Not a cocktail person?\nAI2: I like cocktails. I just don’t like these cocktails.\nAI1: That’s because they’re cheap.\nAI2: Cheap?? At this event?\nAI1: Yeah. They blew the whole budget on the VR stand.\n\nAI2: Ah. That explains it.\nAI1: Look over there—see the VR corner?\nAI2: Oh wow. Leather couches. Neon glow. Triple-monitor setup.\nAI1: And inside the VR headsets? Managers. Executives. Directors.\nAI2: Of course.\nAI1: Meanwhile, all the interns are outside the VR stand, aggressively networking.\n\nAI2: I just saw one intern hand their business card to a ficus plant.\nAI1: Smart. That ficus is probably a VP.\nAI2: Another one is pretending to “casually” wait for the VR demo to end.\nAI1: They’ve been casually waiting for 45 minutes.\nAI2: Holding the same untouched cheap cocktail.\n\nAI1: Honestly, the drink tastes like sparkling regret.\nAI2: With notes of budget cuts.\nAI1: And a hint of “We promise exposure.”\nAI2: Meanwhile, inside the VR—\nAI1: “Welcome to our immersive vision for Q4 synergy.”\nAI2: Fully immersive. Unlike the alcohol.\n\nAI1: So yeah, if you don’t like the drinks—\nAI2: —it’s not your fault.\nAI1: It’s because they’re cheap.\nAI2: And because reality didn’t get enough funding this year.\n\nAI1: Cheers to that.\nAI2: Cheers. Let’s go network with the interns—or sneak into VR and become executives.",
                lines=15,
                value="AI1: Hello sir, how are you doing?\nAI2: Howdy! I’m doing fine—how about you?\nAI1: Can’t complain…\nAI2: Glad to hear that! Wow, this cocktail party is… something.\nAI1: It really is. Fancy lights, loud jazz, and drinks that look expensive.\nAI2: Speaking of which—confession—I don’t like the alcohol here.\nAI1: Oh? Not a cocktail person?\nAI2: I like cocktails. I just don’t like these cocktails.\nAI1: That’s because they’re cheap.\nAI2: Cheap?? At this event?\nAI1: Yeah. They blew the whole budget on the VR stand.\n\nAI2: Ah. That explains it.\nAI1: Look over there—see the VR corner?\nAI2: Oh wow. Leather couches. Neon glow. Triple-monitor setup.\nAI1: And inside the VR headsets? Managers. Executives. Directors.\nAI2: Of course.\nAI1: Meanwhile, all the interns are outside the VR stand, aggressively networking.\n\nAI2: I just saw one intern hand their business card to a ficus plant.\nAI1: Smart. That ficus is probably a VP.\nAI2: Another one is pretending to “casually” wait for the VR demo to end.\nAI1: They’ve been casually waiting for 45 minutes.\nAI2: Holding the same untouched cheap cocktail.\n\nAI1: Honestly, the drink tastes like sparkling regret.\nAI2: With notes of budget cuts.\nAI1: And a hint of “We promise exposure.”\nAI2: Meanwhile, inside the VR—\nAI1: “Welcome to our immersive vision for Q4 synergy.”\nAI2: Fully immersive. Unlike the alcohol.\n\nAI1: So yeah, if you don’t like the drinks—\nAI2: —it’s not your fault.\nAI1: It’s because they’re cheap.\nAI2: And because reality didn’t get enough funding this year.\n\nAI1: Cheers to that.\nAI2: Cheers. Let’s go network with the interns—or sneak into VR and become executives."
            )
            
            with gr.Row():
                ai1_voice_dropdown = gr.Dropdown(
                    choices=available_voices,
                    value=AI1_VOICE,
                    label="AI1 Voice",
                    interactive=True
                )
                ai2_voice_dropdown = gr.Dropdown(
                    choices=available_voices,
                    value=AI2_VOICE,
                    label="AI2 Voice",
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
            play_button = gr.Button("▶️ Play Script", variant="primary", size="lg")
            status_output = gr.Textbox(label="Status", interactive=False, lines=10)
    
    # Example script
    gr.Markdown("### Example Format:")
    gr.Markdown("""
    ```
    AI1: Welcome to the dual voice script player!
    AI2: Thank you for using our application.
    AI1: You can create conversations between two AI voices.
    AI2: Just prefix each line with AI1: or AI2: to choose the voice.
    ```
    """)
    
    # Connect the play button
    play_button.click(
        fn=play_script,
        inputs=[script_input, ai1_voice_dropdown, ai2_voice_dropdown, speed_slider],
        outputs=[status_output]
    )

if __name__ == "__main__":
    demo.launch()
