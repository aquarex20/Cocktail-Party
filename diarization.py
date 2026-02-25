import numpy as np
import pandas as pd
try:
    from whisperx.diarize import DiarizationPipeline
except Exception:
    DiarizationPipeline = None


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

class IncrementalDiarizationSession:
    def __init__(self, diarizer, window_s=30.0, overlap_s=2.0, sim_threshold=0.80):
        self.diarizer = diarizer
        self.window_s = float(window_s)
        self.overlap_s = float(overlap_s)
        self.sim_threshold = float(sim_threshold)

        self.sr = None
        self.buffer = np.zeros((0,), dtype=np.float32)
        self.processed_samples = 0  # how many samples we have "committed" globally

        self.global_segments = []  # (start, end, speaker_global)
        self.global_speakers = {}  # speaker_global -> embedding np.ndarray
        self._speaker_count = 0

    def _new_global_speaker(self, emb: np.ndarray) -> str:
        gid = f"SPK_{self._speaker_count:02d}"
        self._speaker_count += 1
        self.global_speakers[gid] = emb
        return gid

    def _match_to_global(self, chunk_embs: dict[str, list[float]]) -> dict[str, str]:
        mapping = {}
        for chunk_label, emb_list in chunk_embs.items():
            emb = np.asarray(emb_list, dtype=np.float32)

            best_gid, best_sim = None, -1.0
            for gid, gemb in self.global_speakers.items():
                s = cosine_sim(emb, gemb)
                if s > best_sim:
                    best_sim, best_gid = s, gid

            if best_gid is None or best_sim < self.sim_threshold:
                mapping[chunk_label] = self._new_global_speaker(emb)
            else:
                # update prototype (EMA)
                self.global_speakers[best_gid] = 0.9 * self.global_speakers[best_gid] + 0.1 * emb
                mapping[chunk_label] = best_gid

        return mapping

    def ingest(self, audio_array: np.ndarray, sample_rate: int) -> pd.DataFrame:
        if self.sr is None:
            self.sr = int(sample_rate)
        elif self.sr != int(sample_rate):
            raise ValueError(f"Sample rate changed: {self.sr} -> {sample_rate}")

        # mono safety
        if audio_array.ndim != 1:
            raise ValueError(f"Expected mono 1D audio, got shape {audio_array.shape}")

        # float32
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32, copy=False)

        # append new audio
        self.buffer = np.concatenate([self.buffer, audio_array], axis=0)

        window_n = int(self.window_s * self.sr)
        overlap_n = int(self.overlap_s * self.sr)
        advance_n = max(1, window_n - overlap_n)

        new_rows = []

        while len(self.buffer) >= window_n:
            chunk = self.buffer[:window_n]
            chunk_t0_abs = self.processed_samples / self.sr  # absolute seconds

            diarize_df, embs = self.diarizer(chunk, return_embeddings=True)

            # map speakers to stable global IDs
            if embs is not None and len(embs) > 0:
                mapping = self._match_to_global(embs)
                diarize_df["speaker"] = diarize_df["speaker"].map(lambda x: mapping.get(x, x))

            # make times absolute
            diarize_df["start"] = diarize_df["start"] + chunk_t0_abs
            diarize_df["end"]   = diarize_df["end"] + chunk_t0_abs

            # store
            for _, r in diarize_df.iterrows():
                row = {"start": float(r["start"]), "end": float(r["end"]), "speaker": str(r["speaker"])}
                self.global_segments.append((row["start"], row["end"], row["speaker"]))
                new_rows.append(row)

            # commit and keep overlap context
            self.processed_samples += advance_n
            self.buffer = self.buffer[advance_n:]

        return pd.DataFrame(new_rows, columns=["start", "end", "speaker"])

    def ingest_chunk(self, audio_array: np.ndarray, sample_rate: int) -> tuple[pd.DataFrame, float]:
        """
        Diarize a single chunk immediately (no 30s windowing).
        Returns (diarize_df_local, chunk_t0_abs_seconds).
        """
        if self.sr is None:
            self.sr = int(sample_rate)
        elif self.sr != int(sample_rate):
            raise ValueError(f"Sample rate changed: {self.sr} -> {sample_rate}")

        if audio_array.ndim != 1:
            raise ValueError(f"Expected mono 1D audio, got shape {audio_array.shape}")

        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32, copy=False)

        chunk_t0_abs = self.processed_samples / self.sr

        diarize_df, embs = self.diarizer(audio_array, return_embeddings=True)

        # map speakers to stable global IDs
        if embs is not None and len(embs) > 0:
            mapping = self._match_to_global(embs)
            diarize_df["speaker"] = diarize_df["speaker"].map(lambda x: mapping.get(x, x))

        # store absolute segments
        diarize_abs = diarize_df.copy()
        diarize_abs["start"] = diarize_abs["start"] + chunk_t0_abs
        diarize_abs["end"] = diarize_abs["end"] + chunk_t0_abs
        for _, r in diarize_abs.iterrows():
            self.global_segments.append((float(r["start"]), float(r["end"]), str(r["speaker"])))

        # commit chunk length
        self.processed_samples += int(audio_array.shape[0])

        return diarize_df, float(chunk_t0_abs)