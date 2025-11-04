"""
Video Analysis Template with Face Tracking
This version maintains consistent face IDs across frames using position-based tracking.

NEW FEATURES:
- ✅ Persistent face IDs across frames
- ✅ Tracks faces even if they move
- ✅ Consistent mouth movement history
- ✅ Handles faces entering/leaving the scene
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import time


@dataclass
class Face:
    """Represents a detected face in the video."""
    id: int
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    center: Tuple[int, int]
    is_talking: bool = False
    confidence: float = 0.0
    landmarks: Optional[np.ndarray] = None
    frames_missing: int = 0  # NEW: Track how long face has been missing


class FaceTracker:
    """
    Tracks faces across frames to maintain consistent IDs.
    Uses position-based tracking with distance matching.
    """
    
    def __init__(self, max_distance=150, max_missing_frames=30):
        """
        Initialize face tracker.
        
        Args:
            max_distance: Maximum pixel distance to consider same face (default: 150px)
            max_missing_frames: Frames before removing a lost face (default: 30 = ~1 second at 30fps)
        """
        self.tracked_faces: Dict[int, Dict] = {}  # {id: {'center': (x,y), 'bbox': (...), 'frames_missing': int}}
        self.next_id = 0
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
    
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
    
    def update(self, new_detections: List[Face]) -> List[Face]:
        """
        Update tracking with new detections and assign consistent IDs.
        
        Args:
            new_detections: List of newly detected faces (with temporary IDs)
            
        Returns:
            List of faces with persistent IDs
        """
        # Mark all tracked faces as missing initially
        for tracked_id in self.tracked_faces:
            self.tracked_faces[tracked_id]['frames_missing'] += 1
        
        # Match new detections with tracked faces
        matched_ids = set()
        
        for detection in new_detections:
            best_match_id = None
            best_score = 0
            
            # Find best matching tracked face
            for tracked_id, tracked_data in self.tracked_faces.items():
                if tracked_id in matched_ids:
                    continue  # Already matched
                
                # Calculate distance between centers
                distance = self.calculate_distance(detection.center, tracked_data['center'])
                
                # Calculate IoU between bounding boxes
                iou = self.calculate_iou(detection.bbox, tracked_data['bbox'])
                
                # Combined score: closer distance + higher IoU = better match
                # Normalize distance to 0-1 range
                distance_score = max(0, 1 - (distance / self.max_distance))
                combined_score = (distance_score * 0.5) + (iou * 0.5)
                
                if distance < self.max_distance and combined_score > best_score:
                    best_score = combined_score
                    best_match_id = tracked_id
            
            # Assign ID
            if best_match_id is not None and best_score > 0.3:
                # Matched with existing tracked face
                detection.id = best_match_id
                self.tracked_faces[best_match_id]['center'] = detection.center
                self.tracked_faces[best_match_id]['bbox'] = detection.bbox
                self.tracked_faces[best_match_id]['frames_missing'] = 0
                matched_ids.add(best_match_id)
            else:
                # New face detected
                detection.id = self.next_id
                self.tracked_faces[self.next_id] = {
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
    """Handles face detection using MediaPipe Face Detection."""
    
    def __init__(self, min_detection_confidence=0.7):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=1  # 1 for full range detection
        )
    
    def detect_faces(self, frame: np.ndarray) -> List[Face]:
        """
        Detect faces in the given frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            List of detected Face objects (with temporary IDs)
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
                
                face = Face(
                    id=idx,  # Temporary ID, will be replaced by tracker
                    bbox=(x, y, width, height),
                    center=(center_x, center_y),
                    confidence=detection.score[0]
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
        # Store mouth aspect ratios per TRACKED face ID
        self.mouth_history: Dict[int, deque] = {}  # {face_id: deque([mar1, mar2, ...])}
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
        """
        Detect if a person is talking based on mouth movement.
        Uses persistent face ID for consistent history.
        
        Args:
            frame: BGR image
            face: Face object with bounding box and TRACKED ID
            threshold: Threshold for mouth movement variance
            
        Returns:
            True if talking detected
        """
        x, y, w, h = face.bbox
        
        # Expand ROI slightly
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
            
            # Initialize history for this tracked face ID
            if face.id not in self.mouth_history:
                self.mouth_history[face.id] = deque(maxlen=self.history_length)
            
            self.mouth_history[face.id].append(mar)
            
            # Check variance in mouth aspect ratio
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
    """Main class for video analysis with face tracking."""
    
    def __init__(self, video_source=0):
        """
        Initialize video analyzer with tracking.
        
        Args:
            video_source: Video source (0 for webcam, or path to video file)
        """
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.face_tracker = FaceTracker(max_distance=150, max_missing_frames=30)
        self.mouth_detector = MouthMovementDetector()
        
        # Video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video initialized: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
        print(f"Face Tracker: max_distance={self.face_tracker.max_distance}px")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Face]]:
        """
        Process a single frame with face tracking.
        
        Args:
            frame: Input frame
            
        Returns:
            Annotated frame and list of tracked faces
        """
        # Step 1: Detect faces (temporary IDs)
        detected_faces = self.face_detector.detect_faces(frame)
        
        # Step 2: Update tracker to get persistent IDs
        tracked_faces = self.face_tracker.update(detected_faces)
        
        # Step 3: Check for talking using persistent IDs
        for face in tracked_faces:
            face.is_talking = self.mouth_detector.detect_talking(frame, face)
        
        # Step 4: Cleanup old mouth histories
        active_ids = {face.id for face in tracked_faces}
        self.mouth_detector.cleanup_old_histories(active_ids)
        
        # Step 5: Annotate frame
        annotated_frame = self.annotate_frame(frame, tracked_faces)
        
        return annotated_frame, tracked_faces
    
    def annotate_frame(self, frame: np.ndarray, faces: List[Face]) -> np.ndarray:
        """Draw bounding boxes and labels with persistent IDs."""
        annotated = frame.copy()
        
        for face in faces:
            x, y, w, h = face.bbox
            
            # Color based on talking status
            color = (0, 255, 0) if face.is_talking else (255, 0, 0)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Label with PERSISTENT ID
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
        
        # Frame info with tracking stats
        active_faces = self.face_tracker.get_active_face_count()
        total_tracked = len(self.face_tracker.tracked_faces)
        talking_count = sum(f.is_talking for f in faces)
        
        info_text = f"Active: {active_faces} | Total tracked: {total_tracked} | Talking: {talking_count}"
        cv2.putText(annotated, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated
    
    def run(self, display=True, output_path=None):
        """
        Run the video analysis pipeline with tracking.
        
        Args:
            display: Whether to display the video
            output_path: Optional path to save output video
        """
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps,
                                (self.frame_width, self.frame_height))
        
        print("\n=== Video Analysis with Face Tracking ===")
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
                    print("End of video or cannot read frame")
                    break
                
                frame_count += 1
                
                # Process frame with tracking
                annotated_frame, faces = self.process_frame(frame)
                
                # Show detailed stats if requested
                if show_stats:
                    y_offset = 60
                    for face in faces:
                        history_len = len(self.mouth_detector.mouth_history.get(face.id, []))
                        stat_line = f"Person {face.id}: History={history_len} frames"
                        cv2.putText(annotated_frame, stat_line, (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        y_offset += 25
                
                # Write output
                if out:
                    out.write(annotated_frame)
                
                # Display
                if display:
                    cv2.imshow('Video Analysis with Tracking', annotated_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        show_stats = not show_stats
                        print(f"Stats display: {'ON' if show_stats else 'OFF'}")
                
                # Print stats every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_count / elapsed
                    talking_count = sum(f.is_talking for f in faces)
                    print(f"Frame {frame_count}: {len(faces)} faces, {talking_count} talking | FPS: {fps_actual:.1f}")
        
        except KeyboardInterrupt:
            print("\nAnalysis interrupted by user")
        
        finally:
            self.cleanup(out)
            elapsed = time.time() - start_time
            print(f"\n✅ Processed {frame_count} frames in {elapsed:.1f}s")
            print(f"Average FPS: {frame_count/elapsed:.1f}")
            print(f"Total unique faces tracked: {self.face_tracker.next_id}")
    
    def cleanup(self, out=None):
        """Release resources."""
        self.cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()


def main():
    """Example usage with face tracking."""
    
    # Configuration
    VIDEO_SOURCE = "video_4_people.mp4"  # Votre vidéo
    DISPLAY = True
    OUTPUT_PATH = "video_4_people_tracked.mp4"  # Sauvegarder le résultat
    
    # Pour webcam, décommenter:
    # VIDEO_SOURCE = 0
    # OUTPUT_PATH = None
    
    try:
        # Initialize analyzer with tracking
        analyzer = VideoAnalyzer(video_source=VIDEO_SOURCE)
        
        # Run analysis
        analyzer.run(display=DISPLAY, output_path=OUTPUT_PATH)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

