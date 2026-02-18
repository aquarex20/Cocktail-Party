"""
Party Manager - wires agents via internal AudioBus, runs listen loops.
No Blackhole: all routing is in-process, Docker-friendly.
"""

import queue
import threading
import time as time_module

import numpy as np
from loguru import logger

from agent_runtime import AgentRuntime
from internal_audio_bus import AudioBus
from resample import linear_resample

STT_SAMPLE_RATE = 16000
# Longer silence = complete turns, less mid-sentence cutting / overlapping
UTTERANCE_SILENCE_SEC = 2.5

# Global stop event and speaker lock (only one agent speaks at a time)
_internal_party_stop = threading.Event()
_internal_party_speaker_lock = threading.Lock()
_internal_party_threads = []
_internal_party_agents = []


def bus_audio_stream(sub_queue: queue.Queue, target_sr: int = STT_SAMPLE_RATE, stop_event=None):
    """Yield audio chunks from bus subscription, resampled to target_sr. Yields None when idle."""
    while not (stop_event and stop_event.is_set()):
        try:
            sr, chunk = sub_queue.get(timeout=0.1)
        except queue.Empty:
            yield None
            continue
        chunk = np.asarray(chunk, dtype=np.float32).flatten()
        if chunk.size == 0:
            yield None
            continue
        if sr != target_sr:
            chunk = linear_resample(chunk, sr, target_sr)
        yield chunk


def chunk_to_utterances(stream_iter, sr: int = STT_SAMPLE_RATE, silence_sec=UTTERANCE_SILENCE_SEC):
    """
    Consume stream_iter (yields float32 mono chunks or None when idle).
    Emit utterance arrays when we have buffered audio and then silence_sec of no chunks.
    """
    buf = []
    last_chunk_time = 0.0

    for chunk in stream_iter:
        now = time_module.time()
        if chunk is None:
            if buf and (now - last_chunk_time) >= silence_sec:
                utter = np.concatenate(buf).astype(np.float32, copy=False)
                yield utter
                buf = []
            continue
        last_chunk_time = now
        buf.append(np.asarray(chunk, dtype=np.float32).flatten())

    if buf:
        yield np.concatenate(buf).astype(np.float32, copy=False)


def _agent_loop(agent: AgentRuntime, sub_queue: queue.Queue, stop_event: threading.Event):
    """Listen to bus, emit utterances, hand to agent."""
    stream = bus_audio_stream(sub_queue, target_sr=STT_SAMPLE_RATE, stop_event=stop_event)
    for utter in chunk_to_utterances(stream, sr=STT_SAMPLE_RATE, silence_sec=UTTERANCE_SILENCE_SEC):
        if stop_event.is_set():
            break
        if utter.size < 100:  # skip tiny fragments
            continue
        agent.handle_utterance(utter, sample_rate=STT_SAMPLE_RATE)


def _proactive_kick(agent: AgentRuntime, stop_event: threading.Event, speaker_lock: threading.Lock):
    """Kick off the party by having one agent speak first. Uses speaker_lock."""
    if stop_event.is_set():
        return
    time_module.sleep(1.0)  # let threads start
    if stop_event.is_set():
        return
    logger.info(f"[Agent{agent.agent_id}] Proactive kick: starting first utterance")
    if speaker_lock:
        speaker_lock.acquire()
    agent.ai_is_speaking = True
    try:
        for sr, audio_data in agent.talk_generator():
            if stop_event.is_set():
                break
            audio_data = np.asarray(audio_data, dtype=np.float32).flatten()
            if audio_data.size > 0 and agent.output_bus is not None:
                agent.output_bus.publish(sr, audio_data)
            if agent.monitor_output:
                from audio_utils import play_audio
                play_audio(audio_data, sr)
    except Exception as e:
        logger.exception(f"[Agent{agent.agent_id}] Proactive kick failed: {e}")
    finally:
        agent.ai_is_speaking = False
        agent.ai_is_thinking = False
        if speaker_lock:
            speaker_lock.release()


def run_internal_party(
    num_agents: int = 2,
    language: str = "en",
    tts_voices=None,
    monitor_all: bool = True,
) -> tuple[list[AgentRuntime], threading.Event]:
    """
    Start an internal party: N agents cross-wired via AudioBus.
    tts_voices: list of voice IDs (or None for default) per agent, e.g. [None, "af_sarah", "am_adam"]
    Returns (agents, stop_event). Call stop_event.set() to stop.
    """
    global _internal_party_stop, _internal_party_threads, _internal_party_agents

    _internal_party_stop.clear()
    _internal_party_threads = []
    _internal_party_agents = []

    voices = tts_voices or []
    for _ in range(num_agents - len(voices)):
        voices.append(None)

    buses = [AudioBus() for _ in range(num_agents)]
    agents = [
        AgentRuntime(
            agent_id=i + 1,
            language=language,
            tts_voice=voices[i] if i < len(voices) else None,
            monitor_output=monitor_all,
            output_bus=buses[i],
            stop_event=_internal_party_stop,
            speaker_lock=_internal_party_speaker_lock,
        )
        for i in range(num_agents)
    ]
    _internal_party_agents = agents

    # Cross-wire: agent i listens to agent (i+1) % n
    subscriptions = [buses[(i + 1) % num_agents].subscribe() for i in range(num_agents)]

    for i, (agent, sub_q) in enumerate(zip(agents, subscriptions)):
        t = threading.Thread(
            target=_agent_loop,
            args=(agent, sub_q, _internal_party_stop),
            daemon=True,
        )
        t.start()
        _internal_party_threads.append(t)

    # Kick off with agent 0 speaking
    threading.Thread(
        target=_proactive_kick,
        args=(agents[0], _internal_party_stop, _internal_party_speaker_lock),
        daemon=True,
    ).start()

    return agents, _internal_party_stop


def stop_internal_party():
    """Stop the running internal party."""
    global _internal_party_stop
    _internal_party_stop.set()
    for t in _internal_party_threads:
        if t.is_alive():
            t.join(timeout=2.0)


def get_party_transcript() -> str:
    """Return combined transcript from all agents (for UI)."""
    agents = _internal_party_agents
    if not agents:
        return "No party running."
    parts = []
    for a in agents:
        parts.append(f"--- Agent {a.agent_id} ---\n{a.conversation.strip()}")
    return "\n\n".join(parts)


# Character names for the party scene (cocktail party vibe)
PARTY_CHARACTER_NAMES = ["Sage", "Maverick", "Luna", "Cosmo"]


def get_agent_states() -> list[dict]:
    """
    Return agent states for the visual party UI.
    Each dict: {agent_id, name, is_speaking, is_thinking}.
    """
    agents = _internal_party_agents
    if not agents:
        return []
    return [
        {
            "agent_id": a.agent_id,
            "name": PARTY_CHARACTER_NAMES[(a.agent_id - 1) % len(PARTY_CHARACTER_NAMES)],
            "is_speaking": getattr(a, "ai_is_speaking", False),
            "is_thinking": getattr(a, "ai_is_thinking", False),
        }
        for a in agents
    ]
