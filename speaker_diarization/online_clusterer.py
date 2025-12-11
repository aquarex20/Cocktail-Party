"""Online speaker clustering for real-time diarization."""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Sequence
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpeakerProfile:
    """
    Represents a known speaker with accumulated embeddings.

    Attributes:
        speaker_id: Unique identifier (e.g., "SPEAKER_0")
        embeddings: List of embedding vectors for this speaker
        centroid: Mean embedding vector (updated on each new embedding)
        utterance_count: Number of utterances attributed to this speaker
    """
    speaker_id: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    utterance_count: int = 0

    def update_centroid(self) -> None:
        """Recalculate centroid from all embeddings."""
        if self.embeddings:
            self.centroid = np.mean(self.embeddings, axis=0)
            # Normalize centroid for cosine similarity
            norm = np.linalg.norm(self.centroid)
            if norm > 0:
                self.centroid = self.centroid / norm

    def add_embedding(
        self,
        embedding: np.ndarray,
        max_embeddings: int = 50
    ) -> None:
        """
        Add embedding and update centroid.

        Uses a sliding window to keep memory bounded.

        Args:
            embedding: New embedding vector (192,)
            max_embeddings: Maximum embeddings to retain per speaker
        """
        # Normalize embedding before storing
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        self.embeddings.append(embedding)
        self.utterance_count += 1

        # Prune old embeddings if exceeding limit
        if len(self.embeddings) > max_embeddings:
            # Remove oldest embeddings
            self.embeddings = self.embeddings[-max_embeddings:]

        self.update_centroid()


class OnlineSpeakerClusterer:
    """
    Online clustering for speaker identification.

    Maintains speaker profiles and assigns new utterances to existing
    speakers or creates new speaker profiles based on embedding similarity.

    The algorithm:
    1. For each new embedding, compute cosine similarity with all known speaker centroids
    2. If max similarity >= threshold: assign to that speaker, update their centroid
    3. If max similarity < threshold: create new speaker profile

    Attributes:
        similarity_threshold: Minimum cosine similarity for same-speaker match
        max_speakers: Maximum number of distinct speakers to track
        max_embeddings_per_speaker: Max embeddings kept per speaker profile
    """

    def __init__(
        self,
        similarity_threshold: float = 0.78,
        max_speakers: int = 10,
        max_embeddings_per_speaker: int = 50
    ):
        """
        Initialize the online clusterer.

        Args:
            similarity_threshold: Cosine similarity threshold for same-speaker
                                  detection. Range [0, 1], recommended 0.75-0.85.
                                  Higher = stricter matching, more speakers.
            max_speakers: Maximum number of speakers to track simultaneously
            max_embeddings_per_speaker: Max embeddings to keep per speaker
        """
        self.similarity_threshold = similarity_threshold
        self.max_speakers = max_speakers
        self.max_embeddings = max_embeddings_per_speaker

        self.speakers: Dict[str, SpeakerProfile] = {}
        self._next_speaker_id = 0

    def identify_speaker(self, embedding: np.ndarray) -> str:
        """
        Identify the speaker for a given embedding.

        Args:
            embedding: Speaker embedding vector (192,)

        Returns:
            Speaker ID string (e.g., "SPEAKER_0", "SPEAKER_1")
        """
        # Normalize embedding for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Find best match among existing speakers
        best_speaker, best_similarity = self._find_best_match(embedding)

        # Also check similarity against individual embeddings, not just centroid
        best_individual_sim = 0.0
        if best_speaker and best_speaker in self.speakers:
            for stored_emb in self.speakers[best_speaker].embeddings[-5:]:  # Check last 5
                sim = self._cosine_similarity(embedding, stored_emb)
                best_individual_sim = max(best_individual_sim, sim)

        # Use the higher of centroid or individual similarity
        effective_similarity = max(best_similarity, best_individual_sim)

        # Debug: show similarity scores
        print(f"[CLUSTER] Best match: {best_speaker} centroid={best_similarity:.3f} individual={best_individual_sim:.3f} effective={effective_similarity:.3f} (threshold: {self.similarity_threshold})")

        if best_speaker and effective_similarity >= self.similarity_threshold:
            # Update existing speaker's profile
            self.speakers[best_speaker].add_embedding(embedding, self.max_embeddings)
            logger.debug(
                f"Matched {best_speaker} with similarity {effective_similarity:.3f}"
            )
            return best_speaker
        else:
            # Create new speaker
            new_id = self._create_new_speaker(embedding)
            logger.debug(
                f"Created {new_id} (best match was {effective_similarity:.3f})"
            )
            return new_id

    def identify_speaker_with_confidence(
        self,
        embedding: np.ndarray
    ) -> Tuple[str, float, bool]:
        """
        Identify speaker with confidence score and new-speaker flag.

        Args:
            embedding: Speaker embedding vector (192,)

        Returns:
            Tuple of (speaker_id, confidence_score, is_new_speaker)
        """
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Find ALL matches above a minimum threshold, not just the best
        all_matches = self._find_all_matches(embedding, min_threshold=0.15)

        # Debug output
        if all_matches:
            matches_str = ", ".join([f"{sp}:{sim:.3f}" for sp, sim in all_matches[:3]])
            print(f"[CLUSTER] All matches: [{matches_str}] (threshold: {self.similarity_threshold})")
        else:
            print(f"[CLUSTER] No matches found (threshold: {self.similarity_threshold})")

        # Get the best match
        if all_matches:
            best_speaker, best_similarity = all_matches[0]
        else:
            best_speaker, best_similarity = None, 0.0

        # Also check similarity against individual embeddings of top candidates
        best_individual_sim = 0.0
        best_individual_speaker = None
        for speaker_id, _ in all_matches[:3]:  # Check top 3 candidates
            if speaker_id in self.speakers:
                for stored_emb in self.speakers[speaker_id].embeddings[-5:]:
                    sim = self._cosine_similarity(embedding, stored_emb)
                    if sim > best_individual_sim:
                        best_individual_sim = sim
                        best_individual_speaker = speaker_id

        # Use the better of centroid or individual match
        if best_individual_sim > best_similarity:
            effective_similarity = best_individual_sim
            effective_speaker = best_individual_speaker
        else:
            effective_similarity = best_similarity
            effective_speaker = best_speaker

        print(f"[CLUSTER] Effective: {effective_speaker} with {effective_similarity:.3f}")

        if effective_speaker and effective_similarity >= self.similarity_threshold:
            self.speakers[effective_speaker].add_embedding(embedding, self.max_embeddings)
            return effective_speaker, effective_similarity, False
        else:
            new_id = self._create_new_speaker(embedding)
            return new_id, 1.0, True

    def _find_all_matches(
        self,
        embedding: np.ndarray,
        min_threshold: float = 0.1
    ) -> List[Tuple[str, float]]:
        """
        Find all speakers above minimum threshold, sorted by similarity.

        Returns:
            List of (speaker_id, similarity) tuples, sorted descending
        """
        if not self.speakers:
            return []

        matches = []
        for speaker_id, profile in self.speakers.items():
            if profile.centroid is None:
                continue
            similarity = self._cosine_similarity(embedding, profile.centroid)
            if similarity >= min_threshold:
                matches.append((speaker_id, similarity))

        # Sort by similarity descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def _cosine_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two normalized embeddings.

        Args:
            embedding1: First embedding (assumed normalized)
            embedding2: Second embedding (assumed normalized)

        Returns:
            Cosine similarity in range [-1, 1]
        """
        return float(np.dot(embedding1, embedding2))

    def _find_best_match(
        self,
        embedding: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        Find the best matching speaker for an embedding.

        Args:
            embedding: Normalized speaker embedding

        Returns:
            Tuple of (speaker_id or None if no speakers, similarity score)
        """
        if not self.speakers:
            return None, 0.0

        best_speaker = None
        best_similarity = -1.0

        for speaker_id, profile in self.speakers.items():
            if profile.centroid is None:
                continue

            similarity = self._cosine_similarity(embedding, profile.centroid)
            if similarity > best_similarity:
                best_similarity = similarity
                best_speaker = speaker_id

        return best_speaker, best_similarity

    def _create_new_speaker(self, embedding: np.ndarray) -> str:
        """
        Create a new speaker profile and return its ID.

        If max speakers reached, removes the least active speaker.

        Args:
            embedding: Normalized embedding for the new speaker

        Returns:
            New speaker ID string
        """
        # Before creating new speaker, check if we should merge with an existing non-primary speaker
        # This helps when the same person gets fragmented across multiple IDs
        if len(self.speakers) >= 2:
            merge_candidate = self._find_merge_candidate(embedding)
            if merge_candidate:
                print(f"[CLUSTER] Merging into existing speaker: {merge_candidate}")
                self.speakers[merge_candidate].add_embedding(embedding, self.max_embeddings)
                return merge_candidate

        # Check if we need to remove a speaker
        if len(self.speakers) >= self.max_speakers:
            # Find least active speaker (fewest utterances)
            least_active_id = min(
                self.speakers.keys(),
                key=lambda k: self.speakers[k].utterance_count
            )
            logger.info(
                f"Removing least active speaker {least_active_id} "
                f"({self.speakers[least_active_id].utterance_count} utterances)"
            )
            del self.speakers[least_active_id]

        # Create new speaker
        speaker_id = f"SPEAKER_{self._next_speaker_id}"
        self._next_speaker_id += 1

        profile = SpeakerProfile(speaker_id=speaker_id)
        profile.add_embedding(embedding, self.max_embeddings)
        self.speakers[speaker_id] = profile

        logger.info(f"Created new speaker: {speaker_id}")
        return speaker_id

    def _find_merge_candidate(self, embedding: np.ndarray) -> Optional[str]:
        """
        Find if this embedding should merge with a low-activity speaker.

        If a speaker has few utterances and the new embedding is somewhat similar,
        merge instead of creating yet another speaker (reduces fragmentation).

        Returns:
            Speaker ID to merge into, or None
        """
        # Only merge into speakers with low activity (likely fragmented)
        merge_threshold = self.similarity_threshold * 0.6  # Much lower threshold for merging

        for speaker_id, profile in self.speakers.items():
            # Only consider merging into speakers with few utterances (likely fragments)
            if profile.utterance_count > 3:
                continue

            if profile.centroid is None:
                continue

            similarity = self._cosine_similarity(embedding, profile.centroid)

            # Also check against individual embeddings
            for stored_emb in profile.embeddings:
                ind_sim = self._cosine_similarity(embedding, stored_emb)
                similarity = max(similarity, ind_sim)

            if similarity >= merge_threshold:
                print(f"[CLUSTER] Merge candidate: {speaker_id} (sim={similarity:.3f}, threshold={merge_threshold:.3f})")
                return speaker_id

        return None

    def reset(self) -> None:
        """Clear all speaker profiles and start fresh."""
        self.speakers.clear()
        self._next_speaker_id = 0
        logger.info("Speaker clusterer reset")

    def get_speaker_count(self) -> int:
        """Return the number of tracked speakers."""
        return len(self.speakers)

    def get_speaker_stats(self) -> Dict[str, int]:
        """
        Get utterance counts for all speakers.

        Returns:
            Dictionary mapping speaker_id to utterance_count
        """
        return {
            speaker_id: profile.utterance_count
            for speaker_id, profile in self.speakers.items()
        }

    def export_state(self) -> Dict[str, Any]:
        """
        Export clusterer state for persistence.

        Returns:
            Dictionary containing serializable state
        """
        return {
            "similarity_threshold": self.similarity_threshold,
            "max_speakers": self.max_speakers,
            "max_embeddings": self.max_embeddings,
            "next_speaker_id": self._next_speaker_id,
            "speakers": {
                speaker_id: {
                    "speaker_id": profile.speaker_id,
                    "embeddings": [e.tolist() for e in profile.embeddings],
                    "centroid": profile.centroid.tolist() if profile.centroid is not None else None,
                    "utterance_count": profile.utterance_count,
                }
                for speaker_id, profile in self.speakers.items()
            }
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        """
        Import previously saved state.

        Args:
            state: Dictionary from export_state()
        """
        self.similarity_threshold = state.get("similarity_threshold", self.similarity_threshold)
        self.max_speakers = state.get("max_speakers", self.max_speakers)
        self.max_embeddings = state.get("max_embeddings", self.max_embeddings)
        self._next_speaker_id = state.get("next_speaker_id", 0)

        self.speakers.clear()
        for speaker_id, profile_data in state.get("speakers", {}).items():
            profile = SpeakerProfile(
                speaker_id=profile_data["speaker_id"],
                embeddings=[np.array(e) for e in profile_data["embeddings"]],
                centroid=np.array(profile_data["centroid"]) if profile_data["centroid"] else None,
                utterance_count=profile_data["utterance_count"],
            )
            self.speakers[speaker_id] = profile

        logger.info(f"Imported state with {len(self.speakers)} speakers")
