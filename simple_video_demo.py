"""
Simplified Video Analysis Demo
A minimal example for quick testing of face and talking detection.
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque


class SimpleVideoAnalyzer:
    """Minimal implementation for face detection and talking detection."""
    
    def __init__(self):
        # Face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.7,
            model_selection=1
        )
        
        # Face mesh for mouth detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Store mouth movement history
        self.mouth_history = deque(maxlen=10)
    
    def calculate_mouth_opening(self, landmarks, img_shape):
        """Calculate how open the mouth is."""
        h, w = img_shape[:2]
        
        # Mouth landmarks: 13 (upper lip), 14 (lower lip)
        upper = landmarks[13]
        lower = landmarks[14]
        left = landmarks[61]
        right = landmarks[291]
        
        # Convert to pixels
        vertical = abs((lower.y - upper.y) * h)
        horizontal = abs((right.x - left.x) * w)
        
        # Return aspect ratio
        return vertical / (horizontal + 1e-6)
    
    def is_talking(self, mouth_ratio):
        """Detect talking based on mouth movement variance."""
        self.mouth_history.append(mouth_ratio)
        
        if len(self.mouth_history) >= 5:
            variance = np.var(list(self.mouth_history))
            return variance > 0.05  # Threshold for movement
        return False
    
    def process_frame(self, frame):
        """Process frame and return annotated result."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        # Detect faces
        face_results = self.face_detection.process(rgb_frame)
        mesh_results = self.face_mesh.process(rgb_frame)
        
        talking = False
        
        # Draw faces
        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Check for talking
                if mesh_results.multi_face_landmarks:
                    for face_landmarks in mesh_results.multi_face_landmarks:
                        ratio = self.calculate_mouth_opening(
                            face_landmarks.landmark, frame.shape
                        )
                        talking = self.is_talking(ratio)
                
                # Draw box
                color = (0, 255, 0) if talking else (255, 0, 0)
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                
                # Label
                label = "TALKING" if talking else "Silent"
                cv2.putText(frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def run(self, source=0):
        """Run the analyzer on video source."""
        cap = cv2.VideoCapture(source)
        
        print("Starting video analysis... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process
            result = self.process_frame(frame)
            
            # Display
            cv2.imshow('Simple Video Analysis', result)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Run the simple demo."""
    analyzer = SimpleVideoAnalyzer()
    
    # Use webcam (0) or provide video path
    analyzer.run(source=0)
    
    # For video file:
    # analyzer.run(source="your_video.mp4")


if __name__ == "__main__":
    main()

