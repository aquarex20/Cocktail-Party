"""
Détection d'ouverture de bouche AMÉLIORÉE
- Zoom automatique sur les visages éloignés
- Détection multi-échelle
- Filtrage pour réduire les faux positifs
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Optional
from collections import deque


class ImprovedMouthOpenDetector:
    """Détecteur amélioré avec gestion des visages éloignés."""
    
    def __init__(self, mouth_open_threshold=0.25, min_face_size=80):
        """
        Initialise le détecteur amélioré.
        
        Args:
            mouth_open_threshold: Seuil MAR pour bouche ouverte (0.2-0.3)
            min_face_size: Taille minimale du visage en pixels pour traitement (défaut: 80px)
        """
        # MediaPipe avec paramètres optimisés pour petits visages
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=10,  # Augmenté pour détecter plus de visages
            refine_landmarks=True,
            min_detection_confidence=0.3,  # Réduit pour détecter visages éloignés
            min_tracking_confidence=0.3
        )
        
        # Détecteur de visages pour pré-localisation
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5,
            model_selection=1  # Modèle longue distance
        )
        
        self.mouth_open_threshold = mouth_open_threshold
        self.min_face_size = min_face_size
        
        # Landmarks de la bouche
        self.UPPER_LIP = 13
        self.LOWER_LIP = 14
        self.LEFT_CORNER = 61
        self.RIGHT_CORNER = 291
        
        # Points supplémentaires pour détection robuste
        self.UPPER_LIP_POINTS = [13, 312, 311, 310, 415, 308, 324, 318, 402]
        self.LOWER_LIP_POINTS = [14, 87, 178, 88, 95, 78, 191, 80, 81]
        
        # Historique pour filtrage temporel (évite les flickering)
        self.mouth_history = {}  # {face_id: deque([True/False])}
        self.history_length = 5  # Nombre de frames à considérer
        
        # Statistiques
        self.stats = {
            'total_faces': 0,
            'small_faces': 0,
            'upscaled_faces': 0
        }
    
    def detect_faces_locations(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Pré-détecte les emplacements des visages.
        Plus rapide que Face Mesh pour localiser les visages.
        
        Returns:
            Liste de bounding boxes (x, y, width, height)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            h, w = frame.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Vérifier que la bbox est dans l'image
                x = min(x, w - 1)
                y = min(y, h - 1)
                width = min(width, w - x)
                height = min(height, h - y)
                
                faces.append((x, y, width, height))
        
        return faces
    
    def upscale_face_roi(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                         target_size=200) -> Tuple[np.ndarray, float]:
        """
        Agrandit une région du visage pour améliorer la détection.
        
        Args:
            frame: Image complète
            bbox: Bounding box (x, y, width, height)
            target_size: Taille cible en pixels pour le visage
            
        Returns:
            (ROI agrandie, facteur d'échelle)
        """
        x, y, w, h = bbox
        
        # Ajouter du padding
        padding_percent = 0.2
        pad_w = int(w * padding_percent)
        pad_h = int(h * padding_percent)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(frame.shape[1], x + w + pad_w)
        y2 = min(frame.shape[0], y + h + pad_h)
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None, 1.0
        
        # Calculer le facteur d'échelle
        current_size = max(roi.shape[0], roi.shape[1])
        scale_factor = target_size / current_size if current_size < target_size else 1.0
        
        # Agrandir si nécessaire
        if scale_factor > 1.0:
            new_width = int(roi.shape[1] * scale_factor)
            new_height = int(roi.shape[0] * scale_factor)
            roi = cv2.resize(roi, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            self.stats['upscaled_faces'] += 1
        
        return roi, scale_factor
    
    def calculate_mouth_ratio_robust(self, landmarks, img_shape) -> float:
        """
        Calcule le MAR avec moyenne sur plusieurs points pour plus de robustesse.
        """
        h, w = img_shape[:2]
        
        # Méthode 1: Points centraux (rapide)
        upper = landmarks[self.UPPER_LIP]
        lower = landmarks[self.LOWER_LIP]
        left = landmarks[self.LEFT_CORNER]
        right = landmarks[self.RIGHT_CORNER]
        
        vertical_center = abs((lower.y - upper.y) * h)
        horizontal = abs((right.x - left.x) * w)
        
        # Méthode 2: Moyenne sur plusieurs points (robuste)
        vertical_distances = []
        num_points = min(len(self.UPPER_LIP_POINTS), len(self.LOWER_LIP_POINTS))
        
        for i in range(num_points):
            upper_point = landmarks[self.UPPER_LIP_POINTS[i]]
            lower_point = landmarks[self.LOWER_LIP_POINTS[i]]
            vertical_dist = abs((lower_point.y - upper_point.y) * h)
            vertical_distances.append(vertical_dist)
        
        vertical_avg = np.mean(vertical_distances)
        
        # Combiner les deux méthodes
        vertical = (vertical_center + vertical_avg) / 2.0
        
        # MAR
        mar = vertical / (horizontal + 1e-6)
        
        return mar
    
    def smooth_mouth_state(self, face_id: int, is_open: bool) -> bool:
        """
        Lisse l'état de la bouche sur plusieurs frames pour éviter le flickering.
        
        Args:
            face_id: ID du visage
            is_open: État actuel de la bouche
            
        Returns:
            État lissé
        """
        # Initialiser l'historique si nécessaire
        if face_id not in self.mouth_history:
            self.mouth_history[face_id] = deque(maxlen=self.history_length)
        
        # Ajouter l'état actuel
        self.mouth_history[face_id].append(is_open)
        
        # Décision par vote majoritaire
        if len(self.mouth_history[face_id]) >= 3:
            open_count = sum(self.mouth_history[face_id])
            return open_count > len(self.mouth_history[face_id]) / 2
        
        return is_open
    
    def process_single_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                           face_id: int) -> Optional[dict]:
        """
        Traite un seul visage avec upscaling si nécessaire.
        
        Returns:
            Dictionnaire avec les informations du visage ou None
        """
        x, y, w, h = bbox
        face_size = max(w, h)
        
        self.stats['total_faces'] += 1
        
        # Vérifier si le visage est trop petit
        if face_size < self.min_face_size:
            self.stats['small_faces'] += 1
            # Upscaler la région
            roi, scale_factor = self.upscale_face_roi(frame, bbox, target_size=200)
            if roi is None:
                return None
        else:
            # Extraire la ROI normalement avec padding
            padding = int(max(w, h) * 0.2)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            roi = frame[y1:y2, x1:x2]
            scale_factor = 1.0
        
        if roi.size == 0:
            return None
        
        # Détecter les landmarks sur la ROI
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_roi)
        
        if not results.multi_face_landmarks:
            return None
        
        # Prendre le premier visage détecté
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Calculer le MAR
        mar = self.calculate_mouth_ratio_robust(landmarks, roi.shape)
        
        # Déterminer si la bouche est ouverte
        is_open_raw = mar > self.mouth_open_threshold
        
        # Lisser l'état
        is_open = self.smooth_mouth_state(face_id, is_open_raw)
        
        return {
            'id': face_id,
            'bbox': bbox,
            'mouth_ratio': mar,
            'is_open': is_open,
            'is_open_raw': is_open_raw,
            'face_size': face_size,
            'upscaled': scale_factor > 1.0,
            'scale_factor': scale_factor,
            'landmarks': landmarks,
            'roi_shape': roi.shape
        }
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """
        Traite une frame complète avec détection multi-échelle.
        
        Returns:
            (frame annotée, liste des données des visages)
        """
        # Étape 1: Détecter les emplacements des visages (rapide)
        face_bboxes = self.detect_faces_locations(frame)
        
        # Étape 2: Traiter chaque visage individuellement avec upscaling si nécessaire
        faces_data = []
        for face_id, bbox in enumerate(face_bboxes):
            face_info = self.process_single_face(frame, bbox, face_id)
            if face_info:
                faces_data.append(face_info)
        
        # Étape 3: Annoter la frame
        annotated_frame = self.annotate_frame(frame, faces_data)
        
        return annotated_frame, faces_data
    
    def annotate_frame(self, frame: np.ndarray, faces_data: List[dict]) -> np.ndarray:
        """Annote la frame avec les résultats."""
        annotated = frame.copy()
        
        for face in faces_data:
            x, y, w, h = face['bbox']
            is_open = face['is_open']
            mar = face['mouth_ratio']
            face_size = face['face_size']
            upscaled = face['upscaled']
            
            # Couleur selon l'état
            color = (0, 255, 0) if is_open else (255, 0, 0)
            
            # Épaisseur selon la taille du visage
            thickness = 3 if face_size > 150 else 2
            
            # Dessiner la bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
            
            # Label principal
            status = "BOUCHE OUVERTE" if is_open else "Fermée"
            label = f"Face {face['id']}: {status}"
            
            # Ajouter un indicateur si upscalé
            if upscaled:
                label += " [ZOOM]"
            
            # Fond pour le texte
            font_scale = 0.5 if face_size < 100 else 0.6
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                   font_scale, 2)
            cv2.rectangle(annotated, (x, y - text_h - 10), 
                         (x + text_w, y), color, -1)
            
            # Texte
            cv2.putText(annotated, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
            
            # Informations détaillées
            details = f"MAR:{mar:.3f} | Taille:{face_size}px"
            cv2.putText(annotated, details, (x, y + h + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Informations globales
        total_faces = len(faces_data)
        open_mouths = sum(1 for f in faces_data if f['is_open'])
        
        info_text = f"Visages: {total_faces} | Bouches ouvertes: {open_mouths}"
        cv2.putText(annotated, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Statistiques de performance
        stats_text = f"Petits visages: {self.stats['small_faces']}/{self.stats['total_faces']}"
        cv2.putText(annotated, stats_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return annotated
    
    def run(self, source=0):
        """
        Lance la détection en temps réel avec upscaling automatique.
        
        Args:
            source: Source vidéo (0 pour webcam, ou chemin vers fichier)
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Erreur: Impossible d'ouvrir {source}")
            return
        
        print("=== Détecteur de bouche AMÉLIORÉ ===")
        print(f"Source: {source}")
        print(f"Seuil MAR: {self.mouth_open_threshold}")
        print(f"Taille min visage: {self.min_face_size}px")
        print("\n🔍 Fonctionnalités:")
        print("  - Zoom automatique sur visages éloignés")
        print("  - Détection multi-échelle")
        print("  - Filtrage temporel anti-flickering")
        print("\n⌨️  Contrôles:")
        print("  [q] Quitter")
        print("  [+/-] Ajuster seuil MAR")
        print("  [s] Afficher statistiques détaillées")
        print("  [r] Réinitialiser statistiques")
        print()
        
        frame_count = 0
        show_stats = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Fin de la vidéo")
                break
            
            frame_count += 1
            
            # Réinitialiser les stats de la frame
            self.stats['total_faces'] = 0
            self.stats['small_faces'] = 0
            self.stats['upscaled_faces'] = 0
            
            # Traiter la frame
            annotated_frame, faces_data = self.process_frame(frame)
            
            # Afficher statistiques détaillées si demandé
            if show_stats and faces_data:
                y_offset = 90
                for face in faces_data:
                    stat_line = (f"Face {face['id']}: Size={face['face_size']}px, "
                               f"MAR={face['mouth_ratio']:.3f}, "
                               f"Upscaled={face['upscaled']}")
                    cv2.putText(annotated_frame, stat_line, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    y_offset += 20
            
            # Afficher
            cv2.imshow('Détection de bouche - AMÉLIORÉE', annotated_frame)
            
            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                self.mouth_open_threshold += 0.01
                print(f"Seuil MAR: {self.mouth_open_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                self.mouth_open_threshold = max(0.05, self.mouth_open_threshold - 0.01)
                print(f"Seuil MAR: {self.mouth_open_threshold:.2f}")
            elif key == ord('s'):
                show_stats = not show_stats
                print(f"Stats détaillées: {'ON' if show_stats else 'OFF'}")
            elif key == ord('r'):
                self.mouth_history.clear()
                print("Historique réinitialisé")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Détection terminée - {frame_count} frames traitées")
    
    def __del__(self):
        """Nettoyage."""
        self.face_mesh.close()
        self.face_detection.close()


def main():
    """Fonction principale."""
    
    # Configuration
    VIDEO_SOURCE = 0  # 0 = webcam, ou "video.mp4"
    MOUTH_OPEN_THRESHOLD = 0.25  # Seuil MAR
    MIN_FACE_SIZE = 60  # Taille min en pixels (plus bas = détecte visages plus éloignés)
    
    # Exemples:
    # VIDEO_SOURCE = "videoplayback.mp4"  # Fichier vidéo
    # MIN_FACE_SIZE = 40  # Très sensible aux petits visages
    
    try:
        detector = ImprovedMouthOpenDetector(
            mouth_open_threshold=MOUTH_OPEN_THRESHOLD,
            min_face_size=MIN_FACE_SIZE
        )
        
        detector.run(source=VIDEO_SOURCE)
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

