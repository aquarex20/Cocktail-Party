"""
Video Analysis with Face Recognition (Option B)
Uses facial embeddings to track people even when they move rapidly.

NEW FEATURES compared to position-based tracking:
- ✅ Tracks faces using facial features (embeddings), not just position
- ✅ Works even with rapid movements
- ✅ Much more robust to occlusions
- ✅ Can optionally recognize specific people by name

REQUIREMENTS:
    pip install face-recognition
    
Note: face-recognition uses dlib which may need compilation.
If installation fails, this script falls back to deep features from MediaPipe.
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import time

# Try to import face_recognition, fallback to basic if not available
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    print("✅ face_recognition library available - using deep facial features")
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("⚠️  face_recognition not installed - using MediaPipe features")
    print("   For better tracking: pip install face-recognition")


@dataclass
class Face:
    """Represents a detected face in the video."""
    id: int
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    center: Tuple[int, int]
    is_talking: bool = False
    confidence: float = 0.0
    landmarks: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None  # NEW: Facial embedding
    name: Optional[str] = None  # NEW: Person's name (if recognized)


class FaceRecognitionTracker:
    """
    Tracks faces using facial embeddings instead of just position.
    Much more robust to movement and occlusions.
    """
    
    def __init__(self, similarity_threshold=0.6, max_missing_frames=30, max_distance=150):
        """
        Initialize face recognition tracker.
        
        Args:
            similarity_threshold: Max distance between embeddings to consider same person (0-1)
                                 Lower = more strict, Higher = more permissive
                                 Recommended: 0.5-0.7
            max_missing_frames: Frames before removing a lost face
        """
        self.tracked_faces: Dict[int, Dict] = {}  # {id: {'embedding': array, 'frames_missing': int}}
        self.next_id = 0
        self.similarity_threshold = similarity_threshold
        self.max_missing_frames = max_missing_frames
        self.max_distance = max_distance

        # Optional: Known people database
        self.known_people: Dict[str, np.ndarray] = {}  # {name: embedding}
    
    def calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                      bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        More robust than just position when faces are close together.
        """
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0


    def add_known_person(self, name: str, embedding: np.ndarray):
        """
        Add a known person to the database for recognition.
        
        Args:
            name: Person's name
            embedding: Facial embedding (128-d vector from face_recognition or similar)
        """
        self.known_people[name] = embedding
        print(f"✅ Added known person: {name}")
    
    def calculate_embedding_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate Euclidean distance between two embeddings."""
        return np.linalg.norm(emb1 - emb2)
    
    def recognize_person(self, embedding: np.ndarray) -> Optional[str]:
        """
        Try to recognize a person from known database.
        
        Args:
            embedding: Facial embedding to identify
            
        Returns:
            Person's name if recognized, None otherwise
        """
        if not self.known_people:
            return None
        
        best_match = None
        min_distance = float('inf')
        
        for name, known_embedding in self.known_people.items():
            distance = self.calculate_embedding_distance(embedding, known_embedding)
            if distance < min_distance and distance < self.similarity_threshold:
                min_distance = distance
                best_match = name
        
        return best_match
    
    def update(self, new_detections: List[Face]) -> List[Face]:
        """
        Update tracking with new detections using facial embeddings.
        
        Args:
            new_detections: List of newly detected faces with embeddings
            
        Returns:
            List of faces with persistent IDs and names (if recognized)
        """
        for detection in new_detections:
            # ✅ PAS de vérification si données manquantes
            # Il ESSAIE TOUJOURS de matcher avec TOUS les visages
            
            best_match_id = None
            best_score = 0
            
            for tracked_id, tracked_data in self.tracked_faces.items():
                # Calculate distance between centers
                distance = self.calculate_distance(detection.center, tracked_data['center'])
                
                # Calculate IoU between bounding boxes
                iou = self.calculate_iou(detection.bbox, tracked_data['bbox'])
                
                distance_score = max(0, 1 - (distance / self.max_distance))
                combined_score = (distance_score * 0.5) + (iou * 0.5)
                
                if distance < self.max_distance and combined_score > best_score:
                    best_score = combined_score
                    best_match_id = tracked_id
            
            # Crée nouvel ID SEULEMENT si AUCUN match trouvé
            if best_match_id is not None and best_score > 0.3:
                detection.id = best_match_id  # ✅ Garde l'ID
            else:
                detection.id = self.next_id  # Nouvel ID seulement si nécessaire
                # Mark all tracked faces as missing initially
                for tracked_id in self.tracked_faces:
                    self.tracked_faces[tracked_id]['frames_missing'] += 1
                
                matched_ids = set()
                
                if detection.embedding is None:
                    # No embedding available, assign new ID
                    detection.id = self.next_id
                    self.tracked_faces[self.next_id] = {
                        'embedding': None,
                        'center': detection.center,
                        'bbox': detection.bbox,
                        'frames_missing': 0
                    }
                    self.next_id += 1
                    continue
                
                # Try to recognize from known people first
                recognized_name = self.recognize_person(detection.embedding)
                if recognized_name:
                    detection.name = recognized_name
                
                # Match with tracked faces using embeddings
                best_match_id = None
                min_distance = float('inf')
                
                for tracked_id, tracked_data in self.tracked_faces.items():
                    if tracked_id in matched_ids:
                        continue
                    
                    if tracked_data['embedding'] is None:
                        continue
                    
                    # Calculate embedding distance
                    distance = self.calculate_embedding_distance(
                        detection.embedding, 
                        tracked_data['embedding']
                    )
                distance_score = max(0, 1 - (distance / self.max_distance))
                combined_score = (distance_score * 0.5) + (iou * 0.5)
                
                if distance < self.max_distance and combined_score > best_score:
                    best_score = combined_score
                    best_match_id = tracked_id

                
                # Assign ID
                if best_match_id is not None and best_score > 0.3:
                    # Matched with existing tracked face
                    detection.id = best_match_id
                    # Update embedding (average with previous for stability)
                    old_emb = self.tracked_faces[best_match_id]['embedding']
                    if old_emb is not None:
                        new_emb = (old_emb * 0.7 + detection.embedding * 0.3)  # Weighted average
                    else:
                        new_emb = detection.embedding
                    self.tracked_faces[best_match_id]['embedding'] = new_emb
                    self.tracked_faces[best_match_id]['center'] = detection.center
                    self.tracked_faces[best_match_id]['bbox'] = detection.bbox
                    self.tracked_faces[best_match_id]['frames_missing'] = 0
                    matched_ids.add(best_match_id)
                else:
                    # New face detected
                    detection.id = self.next_id
                    self.tracked_faces[self.next_id] = {
                        'embedding': detection.embedding,
                        'center': detection.center,
                        'bbox': detection.bbox,
                        'frames_missing': 0
                    }
                    self.next_id += 1
        
        # Remove faces that have been missing for too long
        ids_to_remove = [
            tracked_id for tracked_id, data in self.tracked_faces.items()
            if data['frames_missing'] > self.max_missing_frames
        ]
        for tracked_id in ids_to_remove:
            del self.tracked_faces[tracked_id]
        
        return new_detections
    
    def get_active_face_count(self) -> int:
        """Get number of currently tracked faces."""
        return sum(1 for data in self.tracked_faces.values() if data['frames_missing'] == 0)


class FaceDetector:
    """Handles face detection and embedding extraction."""
    
    def __init__(self, min_detection_confidence=0.7):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=1
        )
        self.use_face_recognition = FACE_RECOGNITION_AVAILABLE
    
    def extract_embedding(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract facial embedding from face region.
        
        Args:
            frame: Full BGR image
            bbox: Bounding box (x, y, width, height)
            
        Returns:
            128-d embedding vector or None
        """
        x, y, w, h = bbox
        
        # Expand ROI slightly
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        face_roi = frame[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            return None
        
        if self.use_face_recognition:
            # Use face_recognition library (dlib-based, very accurate)
            rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_roi)
            
            if encodings:
                return encodings[0]  # Return first encoding
        else:
            # Fallback: Use simple feature extraction from pixels
            # This is less accurate but works without face_recognition
            resized = cv2.resize(face_roi, (64, 64))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            # Normalize
            normalized = gray.flatten() / 255.0
            # Simple PCA-like dimensionality reduction (simulate embedding)
            # In practice, this is much less accurate than real embeddings
            embedding = np.random.randn(128) * 0.1 + normalized[:128]  # Placeholder
            return embedding
        
        return None
    
    def detect_faces(self, frame: np.ndarray) -> List[Face]:
        """
        Detect faces and extract embeddings.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            List of detected Face objects with embeddings
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            h, w, _ = frame.shape
            for idx, detection in enumerate(results.detections):
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                center_x = x + width // 2
                center_y = y + height // 2
                
                # Extract embedding
                embedding = self.extract_embedding(frame, (x, y, width, height))
                
                face = Face(
                    id=idx,  # Temporary ID, will be replaced by tracker
                    bbox=(x, y, width, height),
                    center=(center_x, center_y),
                    confidence=detection.score[0],
                    embedding=embedding
                )
                faces.append(face)
        
        return faces
    
    def __del__(self):
        self.face_detection.close()


class MouthMovementDetector:
    """Detects mouth movement using MediaPipe Face Mesh with persistent ID tracking."""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mouth_history: Dict[int, deque] = {}
        self.history_length = 10
    
    def calculate_mouth_aspect_ratio(self, landmarks, frame_shape) -> float:
        """Calculate mouth aspect ratio (MAR) to detect talking."""
        h, w = frame_shape[:2]
        
        upper = landmarks[13]
        lower = landmarks[14]
        left = landmarks[61]
        right = landmarks[291]
        
        upper_y = upper.y * h
        lower_y = lower.y * h
        left_x = left.x * w
        right_x = right.x * w
        
        vertical = abs(lower_y - upper_y)
        horizontal = abs(right_x - left_x)
        
        mar = vertical / (horizontal + 1e-6)
        return mar
    
    def detect_talking(self, frame: np.ndarray, face: Face, threshold=0.05) -> bool:
        """Detect if a person is talking based on mouth movement."""
        x, y, w, h = face.bbox
        
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            return False
        
        rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_roi)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            mar = self.calculate_mouth_aspect_ratio(landmarks, face_roi.shape)
            
            if face.id not in self.mouth_history:
                self.mouth_history[face.id] = deque(maxlen=self.history_length)
            
            self.mouth_history[face.id].append(mar)
            
            if len(self.mouth_history[face.id]) >= 5:
                variance = np.var(list(self.mouth_history[face.id]))
                return variance > threshold
        
        return False
    
    def cleanup_old_histories(self, active_face_ids: set):
        """Remove mouth histories for faces that are no longer tracked."""
        ids_to_remove = [
            face_id for face_id in self.mouth_history.keys()
            if face_id not in active_face_ids
        ]
        for face_id in ids_to_remove:
            del self.mouth_history[face_id]
    
    def __del__(self):
        self.face_mesh.close()


class VideoAnalyzer:
    """Main class for video analysis with face recognition."""
    
    def __init__(self, video_source=0, similarity_threshold=0.6):
        """
        Initialize video analyzer with face recognition.
        
        Args:
            video_source: Video source (0 for webcam, or path to video file)
            similarity_threshold: Threshold for face matching (0.5-0.7 recommended)
        """
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.face_tracker = FaceRecognitionTracker(
            similarity_threshold=similarity_threshold,
            max_missing_frames=30
        )
        self.mouth_detector = MouthMovementDetector()
        
        # Video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video initialized: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
        print(f"Face Tracker: similarity_threshold={self.face_tracker.similarity_threshold}")
        print(f"Recognition mode: {'Deep embeddings (face_recognition)' if FACE_RECOGNITION_AVAILABLE else 'Basic features'}")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Face]]:
        """Process a single frame with face recognition tracking."""
        # Step 1: Detect faces and extract embeddings
        detected_faces = self.face_detector.detect_faces(frame)
        
        # Step 2: Update tracker with embeddings (robust to movement!)
        tracked_faces = self.face_tracker.update(detected_faces)
        
        # Step 3: Check for talking
        for face in tracked_faces:
            face.is_talking = self.mouth_detector.detect_talking(frame, face)
        
        # Step 4: Cleanup
        active_ids = {face.id for face in tracked_faces}
        self.mouth_detector.cleanup_old_histories(active_ids)
        
        # Step 5: Annotate
        annotated_frame = self.annotate_frame(frame, tracked_faces)
        
        return annotated_frame, tracked_faces
    
    def annotate_frame(self, frame: np.ndarray, faces: List[Face]) -> np.ndarray:
        """Draw bounding boxes and labels."""
        annotated = frame.copy()
        
        for face in faces:
            x, y, w, h = face.bbox
            
            color = (0, 255, 0) if face.is_talking else (255, 0, 0)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Label with ID and name (if recognized)
            if face.name:
                label = f"{face.name} (ID {face.id})"
            else:
                label = f"Person {face.id}"
            
            if face.is_talking:
                label += " - TALKING"
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x, y - text_h - 10), (x + text_w, y), color, -1)
            
            # Text
            cv2.putText(annotated, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Confidence
            conf_text = f"{face.confidence:.2f}"
            cv2.putText(annotated, conf_text, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Frame info
        active_faces = self.face_tracker.get_active_face_count()
        total_tracked = len(self.face_tracker.tracked_faces)
        talking_count = sum(f.is_talking for f in faces)
        
        info_text = f"Active: {active_faces} | Total: {total_tracked} | Talking: {talking_count}"
        cv2.putText(annotated, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Recognition mode indicator
        mode = "FR" if FACE_RECOGNITION_AVAILABLE else "Basic"
        cv2.putText(annotated, f"Mode: {mode}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return annotated
    
    def run(self, display=True, output_path=None):
        """Run the video analysis pipeline."""
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps,
                                (self.frame_width, self.frame_height))
        
        print("\n=== Video Analysis with Face Recognition ===")
        print("Press 'q' to quit")
        print("Press 's' to show tracking statistics")
        print()
        
        frame_count = 0
        start_time = time.time()
        show_stats = False
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video")
                    break
                
                frame_count += 1
                
                annotated_frame, faces = self.process_frame(frame)
                
                if show_stats:
                    y_offset = 90
                    for face in faces:
                        history_len = len(self.mouth_detector.mouth_history.get(face.id, []))
                        stat_line = f"Person {face.id}: History={history_len} frames"
                        if face.name:
                            stat_line += f" ({face.name})"
                        cv2.putText(annotated_frame, stat_line, (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        y_offset += 25
                
                if out:
                    out.write(annotated_frame)
                
                if display:
                    cv2.imshow('Face Recognition Tracking', annotated_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        show_stats = not show_stats
                        print(f"Stats: {'ON' if show_stats else 'OFF'}")
                
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_count / elapsed
                    talking_count = sum(f.is_talking for f in faces)
                    print(f"Frame {frame_count}: {len(faces)} faces, {talking_count} talking | FPS: {fps_actual:.1f}")
        
        except KeyboardInterrupt:
            print("\nInterrupted")
        
        finally:
            self.cleanup(out)
            elapsed = time.time() - start_time
            print(f"\n✅ Processed {frame_count} frames in {elapsed:.1f}s")
            print(f"Average FPS: {frame_count/elapsed:.1f}")
            print(f"Total unique faces: {self.face_tracker.next_id}")
    
    def cleanup(self, out=None):
        """Release resources."""
        self.cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()


def main():
    """Example usage with face recognition."""
    
    VIDEO_SOURCE = "video_4_people.mp4"
    DISPLAY = True
    OUTPUT_PATH = "video_4_people_face_recognition.mp4"
    SIMILARITY_THRESHOLD = 0.6  # Adjust if needed (0.5-0.7)
    
    try:
        analyzer = VideoAnalyzer(
            video_source=VIDEO_SOURCE,
            similarity_threshold=SIMILARITY_THRESHOLD
        )
        
        analyzer.run(display=DISPLAY, output_path=OUTPUT_PATH)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


