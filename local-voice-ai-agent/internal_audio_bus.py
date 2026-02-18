"""
Internal Audio Bus - in-process publish/subscribe for audio chunks.
Replaces Blackhole for agent-to-agent routing: no virtual devices, Docker-friendly.
"""

import queue
import threading


class AudioBus:
    """Publish/subscribe bus for (sample_rate, float32 mono np.array) audio chunks."""

    def __init__(self, max_chunks=200):
        self._subs = []
        self._lock = threading.Lock()
        self._max_chunks = max_chunks

    def subscribe(self):
        q = queue.Queue(maxsize=self._max_chunks)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            self._subs = [s for s in self._subs if s is not q]

    def publish(self, sample_rate, audio):
        """Publish (sample_rate, audio) to all subscribers. audio: np.array float32 mono."""
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait((sample_rate, audio))
            except queue.Full:
                pass  # drop if subscriber is too slow
