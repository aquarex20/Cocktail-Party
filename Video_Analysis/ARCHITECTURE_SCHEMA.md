# 🏗️ Architecture du Projet CocktailPartyAI

## 📊 Vue d'ensemble hiérarchique

```
CocktailPartyAI/
│
├─ Cocktail-Party/          ← Vous êtes ici (Analyse Vidéo)
│  │
│  ├─ Scripts Principaux/
│  │  ├─ video_analysis_template.py      [Complet - Multi-tâches]
│  │  ├─ simple_video_demo.py            [Simple - Prototype]
│  │  ├─ mouth_open_detector.py          [Basique - Bouche]
│  │  └─ mouth_open_detector_improved.py [Avancé - Bouche + Zoom]
│  │
│  ├─ Documentation/
│  │  ├─ PROJECT_OVERVIEW.md             [Ce fichier - Vue générale]
│  │  ├─ ARCHITECTURE_SCHEMA.md          [Architecture visuelle]
│  │  ├─ README_VIDEO.md                 [Guide utilisateur]
│  │  ├─ VIDEO_ANALYSIS_GUIDE.md         [Guide technique]
│  │  └─ MOUTH_DETECTION_GUIDE.md        [Guide détection bouche]
│  │
│  └─ Données/
│     ├─ videoplayback.mp4               [Vidéo test]
│     ├─ discussion.wav                  [Audio test]
│     ├─ discussion.rttm                 [Annotations]
│     └─ IGN UK Podcast.mp3              [Podcast multi-speakers]
│
└─ diart/                    ← Diarisation Audio (séparé)
   └─ [Bibliothèque speaker diarization]
```

---

## 🔄 Flux de données

### **Pipeline Vidéo Complet**

```
📹 Source Vidéo
    │
    ├─ Webcam (source=0)
    ├─ Fichier vidéo (.mp4, .avi)
    └─ Stream (rtsp://)
    │
    ↓
┌───────────────────────────────┐
│   cv2.VideoCapture()          │  ← Capture frame par frame
└───────────────────────────────┘
    │
    ↓
┌───────────────────────────────┐
│   Frame (image BGR)           │  ← 1280x720 pixels par exemple
└───────────────────────────────┘
    │
    ↓
┌───────────────────────────────┐
│   MediaPipe Face Detection    │  ← Trouve les visages
│   (Détection rapide)          │
└───────────────────────────────┘
    │
    ↓
┌───────────────────────────────┐
│   Liste de bounding boxes     │  ← [(x,y,w,h), (x,y,w,h), ...]
└───────────────────────────────┘
    │
    ↓
    Pour chaque visage:
    │
    ├─ Si petit visage (< 80px) ?
    │   │
    │   ↓ OUI
    │   ┌───────────────────────┐
    │   │  Zoom x2-x3 (ROI)    │  ← Amélioration visages éloignés
    │   └───────────────────────┘
    │
    ↓
┌───────────────────────────────┐
│   MediaPipe Face Mesh         │  ← Détecte 468 landmarks
│   (Détection précise)         │
└───────────────────────────────┘
    │
    ↓
┌───────────────────────────────┐
│   Points de la bouche         │  ← Points 13, 14, 61, 291, etc.
│   (9 points supérieurs)       │
│   (9 points inférieurs)       │
└───────────────────────────────┘
    │
    ↓
┌───────────────────────────────┐
│   Calcul MAR                  │  ← Mouth Aspect Ratio
│   vertical / horizontal       │
└───────────────────────────────┘
    │
    ↓
┌───────────────────────────────┐
│   Filtrage temporel           │  ← Lissage sur 5 frames
│   (Vote majoritaire)          │     (évite flickering)
└───────────────────────────────┘
    │
    ↓
┌───────────────────────────────┐
│   Décision: Bouche ouverte ?  │  ← MAR > seuil (0.25)
│   True / False                │
└───────────────────────────────┘
    │
    ↓
┌───────────────────────────────┐
│   Annotation de la frame      │  ← Dessiner boîtes, labels
│   - Bounding box (vert/rouge) │
│   - Label "BOUCHE OUVERTE"    │
│   - Ratio MAR                 │
└───────────────────────────────┘
    │
    ↓
┌───────────────────────────────┐
│   Affichage / Sauvegarde      │
│   - cv2.imshow()             │  ← Affichage en direct
│   - VideoWriter()            │  ← Sauvegarde en .mp4
└───────────────────────────────┘
```

---

## 🧩 Architecture des Classes

### **1. video_analysis_template.py**

```
VideoAnalyzer
├─ __init__(video_source)
│  ├─ cap = cv2.VideoCapture(source)
│  ├─ face_detector = FaceDetector()
│  ├─ mouth_detector = MouthMovementDetector()
│  └─ audio_analyzer = AudioAnalyzer()
│
├─ process_frame(frame)
│  ├─ faces = face_detector.detect_faces(frame)
│  ├─ for each face:
│  │  └─ face.is_talking = mouth_detector.detect_talking(frame, face)
│  └─ return annotated_frame, faces
│
├─ annotate_frame(frame, faces)
│  ├─ Draw bounding boxes
│  ├─ Draw labels
│  └─ return annotated_frame
│
└─ run(display, output_path)
   ├─ while True:
   │  ├─ ret, frame = cap.read()
   │  ├─ annotated, faces = process_frame(frame)
   │  ├─ cv2.imshow() if display
   │  └─ video_writer.write() if output_path
   └─ cleanup()

FaceDetector
├─ __init__(min_detection_confidence)
│  └─ mp_face_detection.FaceDetection()
│
└─ detect_faces(frame)
   ├─ Convert BGR → RGB
   ├─ results = face_detection.process(rgb_frame)
   └─ return List[Face]

MouthMovementDetector
├─ __init__()
│  ├─ mp_face_mesh.FaceMesh()
│  └─ mouth_history = {}
│
├─ calculate_mouth_aspect_ratio(landmarks, shape)
│  ├─ upper_lip = landmarks[13]
│  ├─ lower_lip = landmarks[14]
│  ├─ left_corner = landmarks[61]
│  ├─ right_corner = landmarks[291]
│  ├─ vertical = |lower.y - upper.y|
│  ├─ horizontal = |right.x - left.x|
│  └─ return vertical / horizontal
│
└─ detect_talking(frame, face, threshold)
   ├─ Extract face ROI
   ├─ results = face_mesh.process(roi)
   ├─ mar = calculate_mouth_aspect_ratio()
   ├─ mouth_history.append(mar)
   ├─ variance = np.var(mouth_history)
   └─ return variance > threshold

AudioAnalyzer
├─ __init__(sample_rate, chunk_size)
└─ detect_speech(audio_chunk, threshold)
   ├─ energy = calculate_audio_energy()
   └─ return energy > threshold
```

---

### **2. mouth_open_detector_improved.py**

```
ImprovedMouthOpenDetector
├─ __init__(mouth_open_threshold, min_face_size)
│  ├─ mp_face_mesh.FaceMesh()
│  ├─ mp_face_detection.FaceDetection()
│  ├─ mouth_history = {}
│  └─ stats = {}
│
├─ detect_faces_locations(frame)
│  ├─ results = face_detection.process(frame)
│  └─ return list of bboxes  [Rapide - Pré-localisation]
│
├─ upscale_face_roi(frame, bbox, target_size=200)
│  ├─ Extract ROI with padding
│  ├─ Calculate scale_factor
│  ├─ if scale > 1: cv2.resize(roi, INTER_CUBIC)
│  └─ return upscaled_roi, scale_factor  [Améliore petits visages]
│
├─ calculate_mouth_ratio_robust(landmarks, shape)
│  ├─ Method 1: Central points (13, 14, 61, 291)
│  ├─ Method 2: Average 9 upper + 9 lower points
│  ├─ vertical = (method1 + method2) / 2
│  └─ return vertical / horizontal  [Plus robuste]
│
├─ smooth_mouth_state(face_id, is_open)
│  ├─ mouth_history[face_id].append(is_open)
│  ├─ open_count = sum(history)
│  └─ return majority_vote  [Anti-flickering]
│
├─ process_single_face(frame, bbox, face_id)
│  ├─ if face_size < min_face_size:
│  │  └─ roi, scale = upscale_face_roi()  [Zoom si nécessaire]
│  ├─ results = face_mesh.process(roi)
│  ├─ mar = calculate_mouth_ratio_robust()
│  ├─ is_open_raw = mar > threshold
│  ├─ is_open = smooth_mouth_state()
│  └─ return face_data
│
├─ process_frame(frame)
│  ├─ bboxes = detect_faces_locations(frame)  [Étape 1: Localisation]
│  ├─ for bbox in bboxes:
│  │  └─ face_data = process_single_face()    [Étape 2: Analyse détaillée]
│  └─ return annotated_frame, faces_data
│
└─ run(source)
   ├─ cap = cv2.VideoCapture(source)
   ├─ while True:
   │  ├─ frame = cap.read()
   │  ├─ annotated, faces = process_frame(frame)
   │  └─ cv2.imshow()
   └─ cleanup()
```

---

## 🎯 Comparaison architecturale

### **Architecture Simple (simple_video_demo.py)**

```
SimpleVideoAnalyzer
├─ Tout dans une seule classe
├─ Pas de séparation FaceDetector / MouthDetector
└─ Méthodes intégrées
```

**Avantages:**

- ✅ Facile à comprendre
- ✅ Code court (~140 lignes)
- ✅ Rapide à modifier

**Inconvénients:**

- ❌ Moins modulaire
- ❌ Difficile à étendre
- ❌ Réutilisation limitée

---

### **Architecture Modulaire (video_analysis_template.py)**

```
VideoAnalyzer
├─ FaceDetector       [Module séparé]
├─ MouthDetector      [Module séparé]
└─ AudioAnalyzer      [Module séparé]
```

**Avantages:**

- ✅ Très modulaire
- ✅ Facile à tester individuellement
- ✅ Réutilisable
- ✅ Extensible

**Inconvénients:**

- ❌ Plus de code (~385 lignes)
- ❌ Plus complexe pour débuter

---

### **Architecture Optimisée (mouth_open_detector_improved.py)**

```
ImprovedMouthOpenDetector
├─ Face Detection (pré-localisation)
├─ Upscaling (traitement adaptatif)
├─ Face Mesh (détection précise)
├─ Calcul robuste (multi-points)
└─ Filtrage temporel (lissage)
```

**Avantages:**

- ✅ Détecte visages éloignés
- ✅ Moins de faux positifs
- ✅ Robuste aux variations

**Inconvénients:**

- ❌ Plus lent (30-50%)
- ❌ Complexe (~450 lignes)

---

## 🔍 Détection multi-échelle expliquée

### **Problème: Visages éloignés**

```
Caméra → Scène → Extraction frame
            │
            ├─ Personne proche (200px)   ✅ Détecté facilement
            ├─ Personne moyenne (100px)  ⚠️ Détecté difficilement
            └─ Personne loin (50px)      ❌ Perdu ou imprécis
```

### **Solution: Zoom adaptatif**

```
Frame complète (1280x720)
    │
    ↓ Face Detection rapide
    │
    ├─ Visage A (200px) ────────────→ Process normal
    │                                  │
    │                                  ↓
    │                               Face Mesh → MAR → Décision
    │
    ├─ Visage B (80px) ─────────────→ Zoom x2 (160px)
    │                                  │
    │                                  ↓
    │                               Face Mesh → MAR → Décision
    │
    └─ Visage C (40px) ─────────────→ Zoom x4 (160px)
                                       │
                                       ↓
                                    Face Mesh → MAR → Décision
```

**Résultat:**

- Tous les visages traités à taille optimale
- Landmarks précis même pour petits visages
- Pas de perte d'information

---

## 📊 Flux de décision MAR

```
Extraction Landmarks
    │
    ↓
Calcul distances
    │
    ├─ Vertical: |lèvre_sup - lèvre_inf|
    └─ Horizontal: |coin_gauche - coin_droit|
    │
    ↓
MAR = Vertical / Horizontal
    │
    ↓
MAR < 0.15 ────────→ Bouche fermée 😐
    │
0.15 ≤ MAR < 0.25 ─→ Légèrement ouverte 🙂
    │
MAR ≥ 0.25 ────────→ Bouche ouverte 😮
    │
MAR > 0.35 ────────→ Très ouverte 😲
```

---

## 🎨 Pipeline d'annotation

```
Frame originale
    │
    ↓
Pour chaque visage détecté:
    │
    ├─ Bouche ouverte?
    │  ├─ OUI → color = (0, 255, 0)    [Vert]
    │  └─ NON → color = (255, 0, 0)     [Rouge]
    │
    ├─ Dessiner bounding box
    │  └─ cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
    │
    ├─ Dessiner label
    │  ├─ text = "Face {id}: BOUCHE OUVERTE"
    │  └─ cv2.putText(frame, text, position, font, color)
    │
    ├─ Dessiner MAR
    │  └─ text = f"MAR: {mar:.3f}"
    │
    └─ (Optionnel) Dessiner landmarks
       └─ for point in mouth_landmarks:
          └─ cv2.circle(frame, point, 2, (0,255,0), -1)
    │
    ↓
Frame annotée
    │
    ├─ Affichage: cv2.imshow()
    └─ Sauvegarde: video_writer.write()
```

---

## 🔄 Intégration Audio-Vidéo (future)

```
Stream Vidéo                    Stream Audio
    │                               │
    ↓                               ↓
Face Detection              VAD (Voice Activity)
    │                               │
    ↓                               ↓
Mouth Movement             Spectral Analysis
    │                               │
    ↓                               ↓
MAR > 0.25?                Energy > threshold?
    │                               │
    └─────────┬─────────────────────┘
              ↓
    ┌─────────────────────┐
    │ Fusion multimodale  │
    │                     │
    │ Bouche ouverte AND  │
    │ Audio présent       │
    │         ↓           │
    │   Parole confirmée  │
    └─────────────────────┘
              │
              ↓
    Attribution speaker
```

**Avantages de la fusion:**

- ✅ Élimine faux positifs (bâillement sans son)
- ✅ Meilleure attribution (qui parle vraiment)
- ✅ Synchronisation audio-vidéo
- ✅ Cocktail party problem résolu

---

## 🎯 Choix architectural selon cas d'usage

```
Cas d'usage                     → Fichier recommandé
─────────────────────────────────────────────────────────
Prototype rapide                → simple_video_demo.py
Application production          → video_analysis_template.py
Détection bouche seule (proche) → mouth_open_detector.py
Détection bouche (éloigné)      → mouth_open_detector_improved.py
Diagnostic installation         → test_setup.py
Apprentissage                   → Lire GUIDES .md
```

---

## 💾 Flux de sauvegarde vidéo

```
Initialisation
    │
    ↓
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(
    filename='output.mp4',
    fourcc=fourcc,
    fps=30,
    frameSize=(1280, 720)
)
    │
    ↓
Boucle traitement
    │
    ├─ frame = cap.read()
    ├─ annotated = process_frame(frame)
    └─ writer.write(annotated)  [Écriture frame annotée]
    │
    ↓
Finalisation
    │
    ├─ writer.release()
    └─ cap.release()
```

---

## 🚀 Optimisations de performance

### **Niveau 1: Réduire résolution**

```python
frame = cv2.resize(frame, (640, 480))  # Au lieu de 1920x1080
```

**Gain:** ~60% plus rapide

### **Niveau 2: Skip frames**

```python
if frame_count % 2 == 0:  # Traiter 1 frame sur 2
    process_frame(frame)
```

**Gain:** ~50% plus rapide

### **Niveau 3: Limiter visages**

```python
FaceMesh(max_num_faces=3)  # Au lieu de 10
```

**Gain:** Variable selon scène

### **Niveau 4: ROI intelligente**

```python
# Ne traiter que les zones avec mouvement
if has_motion(frame):
    process_frame(frame)
```

**Gain:** ~70% plus rapide (scènes statiques)

---

## 📈 Scalabilité

```
1 visage    → 30 FPS (temps réel excellent)
3 visages   → 20 FPS (temps réel bon)
5 visages   → 12 FPS (temps réel acceptable)
10 visages  → 6 FPS  (limite temps réel)
20+ visages → < 3 FPS (offline processing)
```

**Solutions pour scaling:**

- GPU acceleration (MediaPipe supporte GPU)
- Traitement parallèle (multiprocessing)
- Batch processing (traiter frames par groupes)
- Downsampling adaptatif (réduire résolution si trop de visages)

