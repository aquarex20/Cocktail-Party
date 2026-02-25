"""
AI Party Setup Module
---------------------
Automates setup for multiple AI agents talking to each other via Blackhole virtual audio.
Handles: Blackhole build/install, multi-output device instructions, launching multiple instances.
"""

import os
import subprocess
import sys
from pathlib import Path

# Default ports for each agent instance (7860, 7861, 7862, ...)
PARTY_BASE_PORT = 7860

# Blackhole target configs for N agents (each agent needs: input from other, output to other + headphones)
# Agent 1: input=blackhole2chprime (hears agent 2), output=AI-Output1 (blackhole2ch + headphones)
# Agent 2: input=blackhole2ch (hears agent 1), output=AI-Output2 (blackhole2chprime + headphones)
# For 2 agents: blackhole2ch, blackhole2chprime
# For 3 agents: blackhole2ch, blackhole2chprime, blackhole2ch3, etc.
BLACKHOLE_TARGETS = {
    2: [
        {"name": "BlackHole2ch", "bundle_id": "audio.existential.BlackHole2ch"},
        {"name": "BlackHole2chPrime", "bundle_id": "audio.existential.BlackHole2chPrime"},
    ],
    3: [
        {"name": "BlackHole2ch", "bundle_id": "audio.existential.BlackHole2ch"},
        {"name": "BlackHole2chPrime", "bundle_id": "audio.existential.BlackHole2chPrime"},
        {"name": "BlackHole2ch3", "bundle_id": "audio.existential.BlackHole2ch3"},
    ],
    4: [
        {"name": "BlackHole2ch", "bundle_id": "audio.existential.BlackHole2ch"},
        {"name": "BlackHole2chPrime", "bundle_id": "audio.existential.BlackHole2chPrime"},
        {"name": "BlackHole2ch3", "bundle_id": "audio.existential.BlackHole2ch3"},
        {"name": "BlackHole2ch4", "bundle_id": "audio.existential.BlackHole2ch4"},
    ],
}


def get_party_config(num_agents: int, headphones_name: str) -> list[dict]:
    """
    Return config for each agent: input device, output device (multi-output name), port.
    AI-Output1 = blackhole2ch + headphones (agent 1 outputs here; agent 2 hears via blackhole2ch)
    AI-Output2 = blackhole2chprime + headphones (agent 2 outputs here; agent 1 hears via blackhole2chprime)
    """
    if num_agents not in BLACKHOLE_TARGETS:
        raise ValueError(f"Supported agent counts: 2, 3, 4. Got {num_agents}")

    targets = BLACKHOLE_TARGETS[num_agents]
    configs = []
    for i in range(num_agents):
        # Agent i listens to target[(i+1) % n] (the "next" agent's output blackhole)
        input_bh = targets[(i + 1) % num_agents]["name"]
        # Agent i outputs to AI-Output{i+1} = target[i] + headphones
        output_name = f"AI-Output{i + 1}"
        port = PARTY_BASE_PORT + i
        configs.append({
            "agent_id": i + 1,
            "input_device": input_bh,
            "output_device": output_name,
            "port": port,
        })
    return configs


def get_multi_output_instructions(num_agents: int, headphones_name: str) -> str:
    """Generate instructions for creating multi-output devices in Audio MIDI Setup."""
    configs = get_party_config(num_agents, headphones_name)
    lines = [
        "## Create Multi-Output Devices in Audio MIDI Setup",
        "",
        "1. Open **Audio MIDI Setup** (Applications → Utilities → Audio MIDI Setup)",
        "2. Click the **+** button at bottom-left → **Create Multi-Output Device**",
        "3. For each agent, create a device as follows:",
        "",
    ]
    targets = BLACKHOLE_TARGETS[num_agents]
    for i in range(num_agents):
        output_name = f"AI-Output{i + 1}"
        bh_name = targets[i]["name"]
        lines.append(f"### {output_name}")
        lines.append(f"- Name: **{output_name}**")
        lines.append(f"- Enable: **{bh_name}** and **{headphones_name}**")
        lines.append(f"- Ensure **{headphones_name}** has drift correction enabled (except if it's the clock source)")
        lines.append("")
    lines.append("4. Restart CoreAudio if devices don't appear: `sudo killall -9 coreaudiod`")
    return "\n".join(lines)


def run_blackhole_setup_script(num_agents: int) -> tuple[bool, str]:
    """
    Run the Blackhole setup script (clone, build, install).
    Returns (success, message).
    """
    script_dir = Path(__file__).parent
    scripts_dir = script_dir / "scripts"
    setup_script = scripts_dir / "blackhole_setup.sh"

    if not setup_script.exists():
        return False, f"Setup script not found: {setup_script}"

    try:
        result = subprocess.run(
            [str(setup_script), str(num_agents)],
            cwd=str(script_dir),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            return False, f"Setup failed:\n{result.stderr or result.stdout}"
        return True, result.stdout or "Blackhole drivers built and installed successfully."
    except subprocess.TimeoutExpired:
        return False, "Setup timed out (build can take several minutes)."
    except Exception as e:
        return False, str(e)


def launch_party_instances(num_agents: int, headphones_name: str) -> tuple[bool, str]:
    """
    Launch N instances of local_voice_chat_advanced.py with correct device config.
    Returns (success, message with links).
    """
    script_dir = Path(__file__).parent
    main_script = script_dir / "local_voice_chat_advanced.py"

    if not main_script.exists():
        return False, f"Main script not found: {main_script}"

    configs = get_party_config(num_agents, headphones_name)
    links = []

    for cfg in configs:
        cmd = [
            sys.executable,
            str(main_script),
            "--input-device", cfg["input_device"],
            "--output-device", cfg["output_device"],
            "--port", str(cfg["port"]),
        ]
        try:
            subprocess.Popen(
                cmd,
                cwd=str(script_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            links.append(f"Agent {cfg['agent_id']}: http://127.0.0.1:{cfg['port']}")
        except Exception as e:
            return False, f"Failed to launch agent {cfg['agent_id']}: {e}"

    return True, "Launched:\n" + "\n".join(f"- {l}" for l in links)
