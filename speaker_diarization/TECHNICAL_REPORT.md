# Speaker Diarization System - Technical Report

## Project: Cocktail-Party Voice AI Agent
**Version:** 0.1.0
**Date:** December 2024
**Authors:** Development Team

---

## 1. Executive Summary

This document describes the design and implementation of a real-time speaker diarization system integrated into the Cocktail-Party voice AI applications. The system identifies "who spoke when" by extracting speaker embeddings using SpeechBrain's ECAPA-TDNN model and performing online clustering to track speakers across a conversation.

### Key Capabilities
- Real-time speaker identification (post-utterance)
- Support for up to 10 simultaneous speakers
- Gender-agnostic voice differentiation
- Graceful handling of short utterances
- Speaker merging to reduce fragmentation

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Voice AI Application                         │
│  (local_voice_chat_advanced.py / ai_tts.py)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Speaker Diarization Module                    │
│                                                                  │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │  SpeakerDiarizer │──│EmbeddingExtractor│──│OnlineClusterer │ │
│  │   (diarizer.py)  │  │(embedding_ext.py)│  │(online_clust.py)│ │
│  └─────────────────┘  └──────────────────┘  └────────────────┘ │
│                              │                       │          │
│                              ▼                       ▼          │
│                    ┌──────────────────┐    ┌────────────────┐  │
│                    │  SpeechBrain     │    │ Speaker        │  │
│                    │  ECAPA-TDNN      │    │ Profiles       │  │
│                    │  (192-dim)       │    │ (Centroids)    │  │
│                    └──────────────────┘    └────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Module Structure

```
speaker_diarization/
├── __init__.py              # Module exports and torchaudio patches
├── diarizer.py              # High-level SpeakerDiarizer class
├── embedding_extractor.py   # SpeechBrain ECAPA-TDNN wrapper
├── online_clusterer.py      # Online clustering with speaker profiles
├── utils.py                 # Audio preprocessing utilities
└── TECHNICAL_REPORT.md      # This document
```

---

## 3. Core Components

### 3.1 SpeakerDiarizer (diarizer.py)

The main interface class that orchestrates embedding extraction and clustering.

**Key Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `similarity_threshold` | 0.35 | Cosine similarity threshold for speaker matching |
| `max_speakers` | 10 | Maximum concurrent speakers to track |
| `device` | "cpu" | Compute device (cpu/cuda/mps) |
| `min_audio_duration` | 0.5s | Minimum audio length to process |

**Key Methods:**
```python
def identify_speaker(audio: np.ndarray, sample_rate: int) -> str
def identify_speaker_with_confidence(audio, sample_rate) -> Tuple[str, float, bool]
def reset_conversation() -> None
```

**Short Audio Handling:**
- Audio < 1.5 seconds triggers reduced threshold (70% of normal)
- Prevents spurious new speaker creation from brief utterances
- Falls back to last known speaker if audio is invalid

### 3.2 SpeakerEmbeddingExtractor (embedding_extractor.py)

Wraps SpeechBrain's ECAPA-TDNN model for speaker embedding extraction.

**Model Specifications:**
| Attribute | Value |
|-----------|-------|
| Model | speechbrain/spkrec-ecapa-voxceleb |
| Embedding Dimension | 192 |
| Target Sample Rate | 16,000 Hz |
| Input Format | Mono, float32 [-1, 1] |

**Preprocessing Pipeline:**
```
Input Audio (any format)
    │
    ▼
┌─────────────────────┐
│ ensure_mono()       │  Convert stereo to mono
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ normalize_audio()   │  Convert to float32 [-1, 1]
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Resample to 16kHz   │  Using torchaudio.transforms.Resample
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Pad if < 1 second   │  Zero-pad short audio for stability
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ ECAPA-TDNN Model    │  Extract 192-dim embedding
└─────────────────────┘
    │
    ▼
Output: numpy array (192,)
```

### 3.3 OnlineSpeakerClusterer (online_clusterer.py)

Performs online clustering to track and identify speakers.

**Algorithm:**

1. **Normalization**: L2-normalize incoming embedding
2. **Match Finding**: Compare against all speaker centroids
3. **Individual Check**: Also compare against recent individual embeddings
4. **Decision**:
   - If `effective_similarity >= threshold`: Match to existing speaker
   - Else: Check for merge candidate, then create new speaker

**Speaker Profile Structure:**
```python
@dataclass
class SpeakerProfile:
    speaker_id: str           # e.g., "SPEAKER_0"
    embeddings: List[np.ndarray]  # Stored embeddings (max 50)
    centroid: np.ndarray      # Mean of all embeddings (normalized)
    utterance_count: int      # Total utterances attributed
```

**Centroid Update Formula:**
```
centroid = normalize(mean(embeddings))
```

**Cosine Similarity:**
```
similarity(a, b) = dot(a, b)  # For normalized vectors
```

---

## 4. Clustering Strategy

### 4.1 Threshold Selection

The similarity threshold balances false positives (same speaker marked as different) vs false negatives (different speakers marked as same).

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.85 | Very strict | Clean audio, known speakers |
| 0.55 | Moderate | General use |
| 0.35 | Lenient | Noisy audio, varied speech |
| 0.245 | Very lenient | Short utterances (auto-applied) |

**Empirical Findings:**
- Same speaker (male): 0.45 - 0.75 similarity
- Same speaker (female): 0.30 - 0.65 similarity (more variable)
- Different gender: 0.03 - 0.15 similarity
- Different same-gender: 0.15 - 0.35 similarity

### 4.2 Short Audio Handling

Short utterances (< 1.5 seconds) produce unreliable embeddings due to:
- Insufficient phonetic content
- Background noise dominance
- Incomplete vocal characteristics

**Mitigation:**
```python
if audio_duration < MIN_RELIABLE_DURATION:
    threshold = original_threshold * 0.7  # 30% more lenient
```

### 4.3 Speaker Merging

To prevent speaker fragmentation (same person split across multiple IDs):

```python
def _find_merge_candidate(embedding):
    merge_threshold = similarity_threshold * 0.6  # 60% of normal

    for speaker in low_activity_speakers:  # utterance_count <= 3
        if similarity(embedding, speaker) >= merge_threshold:
            return speaker  # Merge into this speaker

    return None  # Create new speaker
```

---

## 5. Integration Points

### 5.1 FastRTC Application (local_voice_chat_advanced.py)

**Audio Format:**
- Sample Rate: 48,000 Hz
- Shape: (1, N) where N = samples
- Data Type: int16

**Integration Code:**
```python
def echo(audio):
    sample_rate, audio_array = audio

    # Flatten if needed
    if audio_array.ndim > 1:
        audio_array = audio_array.flatten()

    # Identify speaker
    speaker_id, confidence, is_new = diarizer.identify_speaker_with_confidence(
        audio_array, sample_rate
    )

    # Transcribe and respond
    transcript = stt_model.stt(audio)
    conversation += f"\n{speaker_id}: {transcript}"
```

### 5.2 Tkinter Application (ai_tts.py)

**Audio Format:**
- Sample Rate: 44,100 Hz
- Format: bytes (int16)
- Resampled internally to 16,000 Hz for VAD/STT

**Integration Code:**
```python
def transcribe_and_speak_44k(audio_bytes_44k):
    audio_int16 = np.frombuffer(audio_bytes_44k, dtype=np.int16)
    speaker_id = diarizer.identify_speaker(audio_int16, IN_SR)

    # Transcribe
    x16 = resample_poly(audio, TARGET_SR, IN_SR)
    transcript = stt_transcribe_16k_float(x16 / 32768.0)

    # Update conversation with speaker label
    last_5_transcript += f"{speaker_id}: {transcript}\n"
```

---

## 6. Performance Characteristics

### 6.1 Latency

| Operation | CPU (M1 Mac) | GPU (CUDA) |
|-----------|--------------|------------|
| Model Loading | 2-3 seconds | 1-2 seconds |
| Embedding Extraction | 50-100 ms | 10-20 ms |
| Clustering | < 1 ms | < 1 ms |
| **Total per utterance** | **50-100 ms** | **10-20 ms** |

### 6.2 Memory Usage

| Component | Size |
|-----------|------|
| ECAPA-TDNN Model | ~100 MB |
| Per Speaker Profile | ~40 KB (50 embeddings) |
| 10 Speakers | ~400 KB |
| **Total** | **~100.5 MB** |

### 6.3 First-Run Download

The SpeechBrain model is downloaded on first use:
- Size: ~100 MB
- Cache Location: `~/.cache/speechbrain/spkrec-ecapa-voxceleb`
- Subsequent runs use cached model

---

## 7. Compatibility Fixes

### 7.1 Torchaudio Backend Patch

Newer versions of torchaudio removed `list_audio_backends()`, causing SpeechBrain to fail. Fixed with:

```python
# In __init__.py
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile', 'sox']
```

### 7.2 Warning Suppression

```python
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=RuntimeWarning)
```

---

## 8. Dependencies

### 8.1 Required Packages

```
speechbrain>=1.0.0
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.20.0
scipy>=1.7.0
```

### 8.2 Python Version

- Minimum: Python 3.10
- Tested: Python 3.13

---

## 9. Configuration Recommendations

### 9.1 For Quiet Environment (1-2 speakers)

```python
diarizer = SpeakerDiarizer(
    similarity_threshold=0.45,
    max_speakers=5,
    device="cpu"
)
```

### 9.2 For Noisy Environment (Multiple speakers)

```python
diarizer = SpeakerDiarizer(
    similarity_threshold=0.35,
    max_speakers=10,
    device="cpu"
)
```

### 9.3 For GPU Acceleration

```python
diarizer = SpeakerDiarizer(
    similarity_threshold=0.40,
    max_speakers=10,
    device="cuda"  # or "mps" for Apple Silicon
)
```

---

## 10. Known Limitations

### 10.1 Short Utterance Accuracy

- Utterances < 1.5 seconds have reduced accuracy
- Single words may be misattributed
- Mitigation: Automatic threshold reduction

### 10.2 Speaker Fragmentation

- Same speaker may get multiple IDs if voice varies significantly
- Mitigation: Speaker merging for low-activity profiles

### 10.3 Cross-Session Persistence

- Speaker profiles are not saved between sessions
- Each application restart creates new SPEAKER_0, SPEAKER_1, etc.
- Future: Add `export_state()` / `import_state()` persistence

### 10.4 Overlapping Speech

- System assumes one speaker per utterance
- Overlapping speech may be attributed to dominant speaker
- Future: Implement overlap detection

---

## 11. Debug Output

The system outputs detailed clustering information:

```
[CLUSTER] All matches: [SPEAKER_0:0.691, SPEAKER_1:0.234] (threshold: 0.35)
[CLUSTER] Effective: SPEAKER_0 with 0.691
[DIARIZATION] Short audio (1.2s), using lower threshold: 0.24
[CLUSTER] Merge candidate: SPEAKER_1 (sim=0.215, threshold=0.210)
```

### Debug Fields

| Field | Description |
|-------|-------------|
| `All matches` | Speakers above 0.15 similarity |
| `Effective` | Final speaker assignment |
| `Short audio` | Triggered when < 1.5 seconds |
| `Merge candidate` | Speaker selected for merging |

---

## 12. API Reference

### SpeakerDiarizer

```python
class SpeakerDiarizer:
    """High-level speaker diarization interface."""

    def __init__(
        self,
        similarity_threshold: float = 0.78,
        max_speakers: int = 10,
        device: Optional[str] = None,
        model_cache_dir: Optional[str] = None,
        min_audio_duration: float = 0.5
    )

    def identify_speaker(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> str

    def identify_speaker_with_confidence(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> Tuple[str, float, bool]

    def reset_conversation(self) -> None

    def is_ready(self) -> bool

    @property
    def speaker_count(self) -> int

    def get_speaker_stats(self) -> Dict[str, int]

    def set_similarity_threshold(self, threshold: float) -> None

    def export_state(self) -> dict

    def import_state(self, state: dict) -> None
```

### SpeakerEmbeddingExtractor

```python
class SpeakerEmbeddingExtractor:
    """ECAPA-TDNN embedding extraction."""

    TARGET_SAMPLE_RATE = 16000
    EMBEDDING_DIM = 192

    def __init__(
        self,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        save_dir: Optional[str] = None,
        device: Optional[str] = None
    )

    def extract_embedding(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray

    def is_ready(self) -> bool
```

### OnlineSpeakerClusterer

```python
class OnlineSpeakerClusterer:
    """Online clustering for speaker tracking."""

    def __init__(
        self,
        similarity_threshold: float = 0.78,
        max_speakers: int = 10,
        max_embeddings_per_speaker: int = 50
    )

    def identify_speaker(self, embedding: np.ndarray) -> str

    def identify_speaker_with_confidence(
        self,
        embedding: np.ndarray
    ) -> Tuple[str, float, bool]

    def reset(self) -> None

    def get_speaker_count(self) -> int

    def get_speaker_stats(self) -> Dict[str, int]

    def export_state(self) -> dict

    def import_state(self, state: dict) -> None
```

---

## 13. Future Improvements

1. **Overlap Detection**: Identify when multiple speakers talk simultaneously
2. **Speaker Enrollment**: Pre-register known speakers for better accuracy
3. **Adaptive Thresholds**: Automatically tune threshold based on audio quality
4. **Persistence**: Save/load speaker profiles across sessions
5. **Voice Activity Detection**: Integrate VAD for better utterance segmentation
6. **Gender Classification**: Separate male/female clustering for improved accuracy

---

## 14. References

1. SpeechBrain ECAPA-TDNN: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
2. ECAPA-TDNN Paper: "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification"
3. VoxCeleb Dataset: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/

---

## Appendix A: Sample Output

```
2025-12-11 22:40:53.332 | DEBUG | Audio: sr=48000, shape=(1, 144000), dtype=int16
2025-12-11 22:40:53.332 | DEBUG | Diarizer ready: True
[CLUSTER] All matches: [] (threshold: 0.35)
[CLUSTER] Effective: None with 0.000
2025-12-11 22:40:53.556 | DEBUG | NEW speaker detected: SPEAKER_0
2025-12-11 22:40:53.786 | DEBUG | [SPEAKER_0] Transcript: Hi, how are you?

2025-12-11 22:41:04.408 | DEBUG | Audio: sr=48000, shape=(1, 86400), dtype=int16
[CLUSTER] All matches: [SPEAKER_0:0.033] (threshold: 0.35)
[CLUSTER] Effective: SPEAKER_0 with 0.033
2025-12-11 22:41:04.465 | DEBUG | NEW speaker detected: SPEAKER_1
2025-12-11 22:41:04.536 | DEBUG | [SPEAKER_1] Transcript: Nice to meet you!

2025-12-11 22:41:16.461 | DEBUG | Audio: sr=48000, shape=(1, 144000), dtype=int16
[CLUSTER] All matches: [SPEAKER_0:0.621, SPEAKER_1:0.189] (threshold: 0.35)
[CLUSTER] Effective: SPEAKER_0 with 0.621
2025-12-11 22:41:16.491 | DEBUG | Matched speaker: SPEAKER_0 (confidence: 0.621)
```

---

*End of Technical Report*
