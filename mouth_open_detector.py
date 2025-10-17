"""
Détection d'ouverture de bouche en temps réel
Détecte quand une personne ouvre la bouche dans une vidéo.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List


class MouthOpenDetector:
    """Détecte l'ouverture de la bouche en temps réel."""
    
    def __init__(self, mouth_open_threshold=0.25):
        """
        Initialise le détecteur.
        
        Args:
            mouth_open_threshold: Seuil pour considérer la bouche comme ouverte 
                                  (ratio vertical/horizontal). Valeur typique: 0.2-0.3
        """
        # Initialiser MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.mouth_open_threshold = mouth_open_threshold
        
        # Indices des landmarks de la bouche (MediaPipe Face Mesh)
        # Lèvre supérieure centre: 13
        # Lèvre inférieure centre: 14
        # Coin gauche: 61
        # Coin droit: 291
        self.UPPER_LIP = 13
        self.LOWER_LIP = 14
        self.LEFT_CORNER = 61
        self.RIGHT_CORNER = 291
        
        # Pour une détection plus précise, on peut utiliser plusieurs points
        self.UPPER_LIP_POINTS = [13, 312, 311, 310, 415, 308, 324, 318, 402]
        self.LOWER_LIP_POINTS = [14, 87, 178, 88, 95, 78, 191, 80, 81]
    
    def calculate_mouth_ratio(self, landmarks, img_shape) -> Tuple[float, bool]:
        """
        Calcule le ratio d'ouverture de la bouche.
        
        Args:
            landmarks: Landmarks du visage de MediaPipe
            img_shape: Forme de l'image (height, width)
            
        Returns:
            Tuple (ratio, is_open) où ratio est le MAR et is_open indique si la bouche est ouverte
        """
        h, w = img_shape[:2]
        
        # Récupérer les points clés de la bouche
        upper = landmarks[self.UPPER_LIP]
        lower = landmarks[self.LOWER_LIP]
        left = landmarks[self.LEFT_CORNER]
        right = landmarks[self.RIGHT_CORNER]
        
        # Convertir en coordonnées pixel
        upper_y = upper.y * h
        lower_y = lower.y * h
        left_x = left.x * w
        right_x = right.x * w
        
        # Calculer les distances
        vertical_distance = abs(lower_y - upper_y)
        horizontal_distance = abs(right_x - left_x)
        
        # Calculer le ratio (Mouth Aspect Ratio - MAR)
        mouth_ratio = vertical_distance / (horizontal_distance + 1e-6)
        
        # Déterminer si la bouche est ouverte
        is_mouth_open = mouth_ratio > self.mouth_open_threshold
        
        return mouth_ratio, is_mouth_open
    
    def calculate_advanced_mouth_ratio(self, landmarks, img_shape) -> Tuple[float, bool]:
        """
        Calcule un ratio plus précis en utilisant plusieurs points de la bouche.
        
        Args:
            landmarks: Landmarks du visage de MediaPipe
            img_shape: Forme de l'image
            
        Returns:
            Tuple (ratio, is_open)
        """
        h, w = img_shape[:2]
        
        # Calculer la distance verticale moyenne entre plusieurs points
        vertical_distances = []
        num_points = min(len(self.UPPER_LIP_POINTS), len(self.LOWER_LIP_POINTS))
        
        for i in range(num_points):
            upper_point = landmarks[self.UPPER_LIP_POINTS[i]]
            lower_point = landmarks[self.LOWER_LIP_POINTS[i]]
            
            vertical_dist = abs((lower_point.y - upper_point.y) * h)
            vertical_distances.append(vertical_dist)
        
        avg_vertical = np.mean(vertical_distances)
        
        # Distance horizontale
        left = landmarks[self.LEFT_CORNER]
        right = landmarks[self.RIGHT_CORNER]
        horizontal_distance = abs((right.x - left.x) * w)
        
        # Ratio
        mouth_ratio = avg_vertical / (horizontal_distance + 1e-6)
        is_mouth_open = mouth_ratio > self.mouth_open_threshold
        
        return mouth_ratio, is_mouth_open
    
    def draw_mouth_landmarks(self, frame: np.ndarray, landmarks, img_shape):
        """
        Dessine les landmarks de la bouche sur l'image.
        
        Args:
            frame: Image BGR
            landmarks: Landmarks du visage
            img_shape: Forme de l'image
        """
        h, w = img_shape[:2]
        
        # Dessiner les points de la bouche
        mouth_points = self.UPPER_LIP_POINTS + self.LOWER_LIP_POINTS + [self.LEFT_CORNER, self.RIGHT_CORNER]
        
        for idx in mouth_points:
            landmark = landmarks[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        # Dessiner les lignes principales
        upper = landmarks[self.UPPER_LIP]
        lower = landmarks[self.LOWER_LIP]
        left = landmarks[self.LEFT_CORNER]
        right = landmarks[self.RIGHT_CORNER]
        
        # Ligne verticale (ouverture)
        cv2.line(frame, 
                (int(upper.x * w), int(upper.y * h)),
                (int(lower.x * w), int(lower.y * h)),
                (255, 0, 0), 2)
        
        # Ligne horizontale (largeur)
        cv2.line(frame,
                (int(left.x * w), int(left.y * h)),
                (int(right.x * w), int(right.y * h)),
                (0, 255, 255), 2)
    
    def process_frame(self, frame: np.ndarray, draw_landmarks=True, use_advanced=False) -> Tuple[np.ndarray, List[dict]]:
        """
        Traite une frame et détecte les bouches ouvertes.
        
        Args:
            frame: Image BGR de OpenCV
            draw_landmarks: Si True, dessine les landmarks de la bouche
            use_advanced: Si True, utilise le calcul avancé avec plusieurs points
            
        Returns:
            Tuple (frame annotée, liste des résultats pour chaque visage)
        """
        # Convertir en RGB pour MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Détecter les visages
        results = self.face_mesh.process(rgb_frame)
        
        faces_data = []
        
        if results.multi_face_landmarks:
            for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                landmarks = face_landmarks.landmark
                
                # Calculer le ratio d'ouverture
                if use_advanced:
                    mouth_ratio, is_open = self.calculate_advanced_mouth_ratio(landmarks, frame.shape)
                else:
                    mouth_ratio, is_open = self.calculate_mouth_ratio(landmarks, frame.shape)
                
                # Obtenir la bounding box du visage
                h, w = frame.shape[:2]
                x_coords = [lm.x * w for lm in landmarks]
                y_coords = [lm.y * h for lm in landmarks]
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                # Stocker les données
                face_data = {
                    'id': face_idx,
                    'mouth_ratio': mouth_ratio,
                    'is_open': is_open,
                    'bbox': (x_min, y_min, x_max - x_min, y_max - y_min)
                }
                faces_data.append(face_data)
                
                # Dessiner la bounding box
                color = (0, 255, 0) if is_open else (255, 0, 0)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                
                # Afficher le statut
                status = "BOUCHE OUVERTE" if is_open else "Bouche fermée"
                label = f"Face {face_idx}: {status}"
                
                # Fond pour le texte
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x_min, y_min - text_h - 10), 
                            (x_min + text_w, y_min), color, -1)
                
                # Texte
                cv2.putText(frame, label, (x_min, y_min - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Afficher le ratio
                ratio_text = f"Ratio: {mouth_ratio:.3f}"
                cv2.putText(frame, ratio_text, (x_min, y_max + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Dessiner les landmarks si demandé
                if draw_landmarks:
                    self.draw_mouth_landmarks(frame, landmarks, frame.shape)
        
        # Afficher les informations globales
        total_faces = len(faces_data)
        open_mouths = sum(1 for f in faces_data if f['is_open'])
        info_text = f"Visages: {total_faces} | Bouches ouvertes: {open_mouths}"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame, faces_data
    
    def run(self, source=0, draw_landmarks=True, use_advanced=False):
        """
        Lance la détection en temps réel.
        
        Args:
            source: Source vidéo (0 pour webcam, ou chemin vers fichier vidéo)
            draw_landmarks: Si True, dessine les landmarks de la bouche
            use_advanced: Si True, utilise le calcul avancé
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Erreur: Impossible d'ouvrir la source vidéo {source}")
            return
        
        print("=== Détecteur de bouche ouverte ===")
        print(f"Source: {source}")
        print(f"Seuil d'ouverture: {self.mouth_open_threshold}")
        print("Appuyez sur 'q' pour quitter")
        print("Appuyez sur '+' pour augmenter le seuil")
        print("Appuyez sur '-' pour diminuer le seuil")
        print("Appuyez sur 'l' pour activer/désactiver les landmarks")
        print()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Fin de la vidéo ou erreur de lecture")
                break
            
            # Traiter la frame
            annotated_frame, faces_data = self.process_frame(
                frame, 
                draw_landmarks=draw_landmarks,
                use_advanced=use_advanced
            )
            
            # Afficher le seuil actuel
            threshold_text = f"Seuil: {self.mouth_open_threshold:.2f}"
            cv2.putText(annotated_frame, threshold_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Afficher
            cv2.imshow('Détection de bouche ouverte', annotated_frame)
            
            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                self.mouth_open_threshold += 0.01
                print(f"Seuil augmenté: {self.mouth_open_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                self.mouth_open_threshold = max(0.05, self.mouth_open_threshold - 0.01)
                print(f"Seuil diminué: {self.mouth_open_threshold:.2f}")
            elif key == ord('l'):
                draw_landmarks = not draw_landmarks
                print(f"Landmarks: {'Activés' if draw_landmarks else 'Désactivés'}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Détection terminée")
    
    def __del__(self):
        """Nettoyage."""
        self.face_mesh.close()


def main():
    """Fonction principale."""
    
    # Configuration
    VIDEO_SOURCE = 0  # 0 pour webcam, ou chemin vers fichier vidéo
    MOUTH_OPEN_THRESHOLD = 0.25  # Ajuster selon vos besoins (0.2-0.3 typique)
    DRAW_LANDMARKS = True  # Afficher les points de la bouche
    USE_ADVANCED = False  # Utiliser le calcul avancé avec plusieurs points
    
    # Exemples de configuration:
    # VIDEO_SOURCE = "videoplayback.mp4"  # Pour un fichier vidéo
    # MOUTH_OPEN_THRESHOLD = 0.2  # Seuil plus bas = plus sensible
    
    try:
        # Créer le détecteur
        detector = MouthOpenDetector(mouth_open_threshold=MOUTH_OPEN_THRESHOLD)
        
        # Lancer la détection
        detector.run(
            source=VIDEO_SOURCE,
            draw_landmarks=DRAW_LANDMARKS,
            use_advanced=USE_ADVANCED
        )
        
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

