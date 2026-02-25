"""
Configuration, CLI, and config help for the local voice chat advanced app.
"""

import argparse
import os
import subprocess
import sys

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
TARGET_SAMPLE_RATE = 48000  # Mac speakers prefer 48kHz
LOCAL_PARTY_PORT = 7861
SYSTEM_DEFAULT_LABEL = "— System default —"

# -----------------------------------------------------------------------------
# Config / troubleshooting text (used by UI and CLI)
# -----------------------------------------------------------------------------
CONFIG_HELP_MD = """
**Party + AI Chat setup**

- **local_party.py** (Script Player) outputs to **Script Config** (multi-output: BlackHole 2ch + your headphones/speakers).
- **This app** uses **BlackHole 2ch** as *input* and your **headphones/speakers** as *output*.

Use the **same output device** for both so you hear the script and the AI in one place.
"""

TROUBLESHOOTING_MD = """
**If you can't hear properly:**  
1. Set your system sound output to your headphones/speakers.  
2. In Audio MIDI Setup, ensure **Script Config** includes that same device.  
3. Set this app's **Output device** above to that same device.
"""


def parse_device(value):
    """Parse device arg: int string -> int, else keep as str (device name)."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value.strip()


def print_config_explanation():
    """Print how local_party and advanced chat fit together."""
    print("""
================================================================================
  AUDIO CONFIGURATION (Party + AI Chat)
================================================================================

  • local_party.py (Script Player)
    Output: "Script Config" (multi-output device that sends audio to:
            - BlackHole 2ch  (virtual cable)
            - Your headphones/speakers (the same device you listen on)

  • local_voice_chat_advanced.py (this app)
    Input:  BlackHole 2ch (receives what the party sends + your mic if routed)
    Output: Your headphones/speakers (so you hear the AI and the party together)

  Both apps should use the SAME output device (your headphones/speakers) so you
  hear the script and the AI in one place.

  If you can't hear properly:
  → Check your Mac (or system) sound output is set to your headphones/speakers.
  → In Audio MIDI Setup, ensure "Script Config" includes that same device.
  → Set this app's output (--output-device) to that same device.

================================================================================
""")


def build_parser():
    """Build and return the argparse parser for the app."""
    parser = argparse.ArgumentParser(
        description="Local Voice Chat Advanced",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Audio device configuration (optional):
  Use --input-device and --output-device to set where to capture and play audio.
  For the party setup: input = "BlackHole 2ch", output = your headphones/speakers.
  Use --testing to list devices, get configuration help, and optionally run local_party.py.
        """,
    )
    parser.add_argument("--phone", action="store_true", help="Launch with FastRTC phone interface (get a temp phone number)")
    parser.add_argument("--input-device", type=str, default=None, metavar="NAME_OR_INDEX", help="Audio input device (e.g. 'BlackHole 2ch' or index). Required for party setup.")
    parser.add_argument("--output-device", type=str, default=None, metavar="NAME_OR_INDEX", help="Audio output device (e.g. your headphones name or index). Same as Script Config output.")
    parser.add_argument("--port", type=int, default=None, metavar="PORT", help="Gradio server port (default: auto when not set). Use e.g. --port 7860 for a fixed port.")
    parser.add_argument("--testing", action="store_true", help="Testing mode: show config explanation, list devices, optionally run local_party.py, then start session.")
    parser.add_argument("--language", type=str, choices=["en", "it"], default="en", help="Language: 'en' (English) or 'it' (Italian). Set before starting; Italian uses Italian prompts and TTS.")
    return parser


def run_testing_mode(args, get_device_lists, set_audio_devices):
    """
    Interactive testing: explain config, list devices, optionally run local_party, then gate session start.
    get_device_lists and set_audio_devices are callables from audio_utils.
    Returns (input_device, output_device) if user entered them; otherwise (None, None).
    """
    print_config_explanation()
    inputs, outputs = get_device_lists()
    print("Available INPUT devices (use name or index for --input-device):")
    for i, name in inputs:
        print(f"  [{i}] {name}")
    print("\nAvailable OUTPUT devices (use name or index for --output-device):")
    for i, name in outputs:
        print(f"  [{i}] {name}")

    test_input_dev = None
    test_output_dev = None
    try:
        inp = input("\nInput device (name or index, or Enter to use --input-device / default): ").strip()
        test_input_dev = parse_device(inp) if inp else None
        out = input("Output device (name or index, or Enter to use --output-device / default): ").strip()
        test_output_dev = parse_device(out) if out else None
    except EOFError:
        pass

    local_party_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_party.py")
    if os.path.isfile(local_party_path):
        try:
            reply = input("\nRun local_party.py in the background to test audio? [y/N]: ").strip().lower()
            if reply in ("y", "yes"):
                subprocess.Popen(
                    [sys.executable, local_party_path],
                    cwd=os.path.dirname(local_party_path),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print("Started local_party.py. Use it to play a script; you should hear it on your output device.")
        except EOFError:
            pass
    else:
        print(f"\n(local_party.py not found at {local_party_path}; skip running it for testing.)")

    print("\nIf you have sound issues, check:")
    print("  • System output is your headphones/speakers.")
    print("  • Script Config and this app use the SAME output device.")
    print()
    try:
        input("Press Enter to start the voice chat session (or Ctrl+C to exit)...")
    except EOFError:
        pass
    return (test_input_dev, test_output_dev)
