# 📁 Vue d'ensemble du projet CocktailPartyAI

## 🎯 Objectif global

Ce projet permet l'**analyse multimodale** (audio + vidéo) pour :

- Détecter qui parle dans une conversation (Cocktail Party Problem)
- Analyser les mouvements de bouche
- Combiner analyse audio (diarisation) et visuelle (détection de visage)

---

## 📂 Structure des fichiers

### 🔴 **FICHIERS PRINCIPAUX - Scripts Python**

#### 1. **`video_analysis_template.py`** ⭐ COMPLET

**Rôle:** Template principal complet avec architecture modulaire

**Ce qu'il fait:**

- 🎭 Détecte les visages avec MediaPipe Face Detection
- 👄 Analyse les mouvements de bouche avec MediaPipe Face Mesh
- 🗣️ Détecte qui parle en calculant le MAR (Mouth Aspect Ratio)
- 🎤 Inclut une classe AudioAnalyzer (pour intégration audio future)
- 💾 Peut sauvegarder la vidéo analysée

**Architecture:**

```python
VideoAnalyzer           # Classe principale
├─ FaceDetector        # Détection des visages
├─ MouthMovementDetector  # Détection ouverture bouche
└─ AudioAnalyzer       # Analyse audio (optionnel)
```

**Utilisation:**

```bash
python video_analysis_template.py
```

**Fonctionnalités:**

- Webcam OU fichier vidéo
- Détection multi-visages (jusqu'à 5)
- Annotations visuelles (boîtes vertes = parle, rouges = silencieux)
- Export vidéo annotée

---

#### 2. **`simple_video_demo.py`** ⚡ SIMPLE & RAPIDE

**Rôle:** Version simplifiée pour tests rapides

**Ce qu'il fait:**

- Version minimaliste du template complet
- Même fonctionnalités de base mais code plus court
- Idéal pour prototypage rapide

**Différence avec template complet:**

- ❌ Pas de classes séparées (tout dans une classe)
- ❌ Pas d'AudioAnalyzer
- ❌ Architecture moins modulaire
- ✅ Plus rapide à comprendre
- ✅ Moins de lignes de code

**Utilisation:**

```bash
python simple_video_demo.py
```

---

#### 3. **`mouth_open_detector.py`** 👄 NOUVEAU - Version basique

**Rôle:** Détection SPÉCIFIQUE de l'ouverture de bouche

**Ce qu'il fait:**

- Se concentre UNIQUEMENT sur la détection de bouche ouverte/fermée
- Calcule le MAR en temps réel
- Affiche les landmarks de la bouche
- Contrôles interactifs pour ajuster le seuil

**Avantages:**

- ✅ Code focalisé sur une seule tâche
- ✅ Visualisation des points de la bouche
- ✅ Ajustement seuil en direct (+/-)
- ✅ Rapide et léger

**Limitations:**

- ⚠️ Perd les visages éloignés (< 100px)
- ⚠️ Pas de filtrage temporel (peut flicker)

**Utilisation:**

```bash
python mouth_open_detector.py
```

**Contrôles:**

- `+` / `-` : Ajuster seuil MAR
- `l` : Afficher/cacher landmarks bouche
- `q` : Quitter

---

#### 4. **`mouth_open_detector_improved.py`** 🚀 NOUVEAU - Version avancée

**Rôle:** Détection d'ouverture de bouche avec OPTIMISATIONS pour visages éloignés

**Ce qu'il fait:**

- Tout ce que fait la version basique +
- 🔍 **Zoom automatique** sur les petits visages
- 📊 **Détection multi-échelle**
- 🎯 **Filtrage temporel** (anti-flickering)
- 📈 **Calcul MAR robuste** sur 9 points au lieu de 4
- 📍 **Pré-détection** avec Face Detection avant Face Mesh

**Architecture avancée:**

```python
ImprovedMouthOpenDetector
├─ detect_faces_locations()     # Pré-détection rapide
├─ upscale_face_roi()           # Zoom sur petits visages
├─ calculate_mouth_ratio_robust() # MAR sur 9 points
├─ smooth_mouth_state()         # Filtrage temporel
└─ process_single_face()        # Traitement individuel
```

**Algorithme:**

```
1. Face Detection → Trouve les visages (rapide)
2. Pour chaque visage:
   - Si taille < 80px → Zoom x2-x3
   - Face Mesh sur ROI agrandie
   - Calcul MAR robuste (moyenne 9 points)
   - Filtrage temporel (vote sur 5 frames)
3. Décision lissée
```

**Avantages sur version basique:**

- ✅ Détecte visages jusqu'à 40px (vs 100px)
- ✅ Moins de faux positifs
- ✅ Pas de flickering
- ✅ Plus robuste aux variations

**Inconvénients:**

- ⚠️ Plus lent (~30-50% selon nb visages)
- ⚠️ Plus complexe

**Utilisation:**

```bash
python mouth_open_detector_improved.py
```

**Contrôles:**

- `+` / `-` : Ajuster seuil MAR
- `s` : Afficher stats détaillées
- `r` : Réinitialiser historique
- `q` : Quitter

---

#### 5. **`test_setup.py`** 🧪 TEST

**Rôle:** Script de diagnostic pour vérifier l'installation

**Ce qu'il fait:**

- ✓ Teste si OpenCV est installé
- ✓ Teste si MediaPipe est installé
- ✓ Teste si NumPy est installé
- ✓ Teste si PyAudio est installé
- ✓ Teste l'accès caméra
- ✓ Teste MediaPipe Face Detection
- ✓ Affiche les versions installées

**Utilisation:**

```bash
python test_setup.py
```

**Sortie typique:**

```
Testing imports...
  OpenCV: ✓ OK
  MediaPipe: ✓ OK
  NumPy: ✓ OK
  PyAudio: ✗ FAILED

Installed versions:
  OpenCV: 4.8.1
  MediaPipe: 0.10.5
  NumPy: 1.24.3

Testing camera access...
  Camera: ✓ OK - Resolution: 1280x720

Testing MediaPipe face detection...
  MediaPipe Face Detection: ✓ OK

✓ All tests passed!
```

---

### 📄 **FICHIERS DE CONFIGURATION**

#### 6. **`requirements_video_analysis.txt`** 📦 DÉPENDANCES

**Rôle:** Liste des packages Python nécessaires

**Contenu:**

```
opencv-python>=4.8.0    # Traitement vidéo
mediapipe>=0.10.0       # Détection faciale
numpy>=1.24.0           # Calculs numériques

# Optionnel pour audio:
# scipy>=1.11.0
# librosa>=0.10.0
```

**Installation:**

```bash
pip install -r requirements_video_analysis.txt
```

---

#### 7. **`requirements_video_analysis copy.txt`** 🔄 DOUBLON

**Rôle:** Copie de sauvegarde (probablement ancienne version)

⚠️ **À supprimer** ou renommer si vous voulez garder un historique

---

### 📚 **DOCUMENTATION**

#### 8. **`README.md`** 📖 PRINCIPAL

**Rôle:** Documentation principale du projet

**Contenu minimal actuel:**

```markdown
# Cocktail-Party-Multidisciplinary-Project
```

⚠️ **Pourrait être enrichi** avec une vraie documentation du projet

---

#### 9. **`README_VIDEO.md`** 📖 GUIDE VIDÉO

**Rôle:** Guide d'utilisation détaillé pour l'analyse vidéo

**Sections:**

- Installation
- Quick Start
- Comment ça fonctionne (explication MAR)
- Exemples de code
- Configuration
- Troubleshooting
- Intégration avec diart
- Prochaines étapes

**Très complet !** Contient tout ce qu'il faut pour démarrer.

---

#### 10. **`VIDEO_ANALYSIS_GUIDE.md`** 📖 GUIDE TECHNIQUE APPROFONDI

**Rôle:** Guide technique détaillé (307 lignes)

**Contenu:**

- Architecture des classes
- Explications détaillées de chaque composant
- Cas d'usage avancés
- Personnalisation
- Performance

**Différence avec README_VIDEO.md:**

- README_VIDEO = Guide utilisateur
- VIDEO_ANALYSIS_GUIDE = Documentation technique

---

#### 11. **`MOUTH_DETECTION_GUIDE.md`** 👄 GUIDE DÉTECTION BOUCHE

**Rôle:** Guide complet sur la détection d'ouverture de bouche

**Sections:**

- Comparaison des 2 versions (basique vs améliorée)
- Explication technique du MAR
- Distances de détection
- Modules utilisés
- Cas d'usage
- Paramètres à ajuster
- Tests de performance
- Problèmes courants
- Optimisations possibles

**Très utile pour comprendre les limites et optimisations !**

---

### 🎥 **FICHIERS MÉDIA**

#### 12. **`videoplayback.mp4`** 🎬 VIDÉO TEST

**Rôle:** Fichier vidéo pour tester les scripts

**Utilisation:**

```python
analyzer = VideoAnalyzer(video_source="videoplayback.mp4")
analyzer.run()
```

---

#### 13. **`discussion.wav`** 🔊 AUDIO

**Rôle:** Fichier audio (probablement pour diarisation avec diart)

**Format:** WAV (format non compressé)

**Utilisation potentielle:**

- Diarisation audio avec diart
- Tests de synchronisation audio-vidéo

---

#### 14. **`discussion.rttm`** 📝 ANNOTATIONS AUDIO

**Rôle:** Fichier RTTM (Rich Transcription Time Marked)

**Format:** Annotations temporelles de qui parle quand

**Exemple de contenu:**

```
SPEAKER discussion 1 0.00 2.50 <NA> <NA> speaker1 <NA> <NA>
SPEAKER discussion 1 2.50 1.30 <NA> <NA> speaker2 <NA> <NA>
```

**Usage:**

- Ground truth pour évaluer la diarisation
- Format standard pour la diarisation speaker
- Compatible avec diart

---

#### 15. **`IGN UK Podcast #236 British Laughs and Global Leaks.mp3`** 🎙️ PODCAST

**Rôle:** Fichier audio podcast (test avec plusieurs speakers)

**Utilisation potentielle:**

- Tester diarisation audio multi-speakers
- Cas d'usage complexe (chevauchements, rires, etc.)

---

### 🔄 **FICHIERS DUPLIQUÉS**

#### 16. **`simple_video_demo copy.py`** 🔄

#### 17. **`video_analysis_template copy.py`** 🔄

**Rôle:** Copies de sauvegarde / anciennes versions

⚠️ **Recommandation:**

- Vérifier les différences avec les originaux
- Supprimer si identiques
- Ou renommer en `.backup` ou déplacer dans un dossier `archive/`

---

## 🔄 Flux de travail typique

### **Cas 1: Démarrage rapide**

```bash
# 1. Tester l'installation
python test_setup.py

# 2. Test rapide webcam
python simple_video_demo.py

# 3. Si ça marche, utiliser le template complet
python video_analysis_template.py
```

### **Cas 2: Détection bouche uniquement**

```bash
# Version basique (visages proches)
python mouth_open_detector.py

# Version améliorée (visages éloignés)
python mouth_open_detector_improved.py
```

### **Cas 3: Analyse vidéo complète**

```python
from video_analysis_template import VideoAnalyzer

# Analyser une vidéo
analyzer = VideoAnalyzer(video_source="videoplayback.mp4")
analyzer.run(display=True, output_path="analyzed_output.mp4")
```

---

## 📊 Comparaison des scripts principaux

| Fichier                           | Usage            | Visages éloignés | Performance        | Complexité      |
| --------------------------------- | ---------------- | ---------------- | ------------------ | --------------- |
| `simple_video_demo.py`            | Prototype rapide | ❌               | ⚡⚡⚡ Très rapide | ⭐ Simple       |
| `video_analysis_template.py`      | Production       | ⚠️ Limité        | ⚡⚡ Rapide        | ⭐⭐ Moyen      |
| `mouth_open_detector.py`          | Test bouche      | ❌               | ⚡⚡⚡ Très rapide | ⭐ Simple       |
| `mouth_open_detector_improved.py` | Bouche précise   | ✅               | ⚡ Moyen           | ⭐⭐⭐ Complexe |

---

## 🎯 Quelle version choisir ?

### **Pour débuter / tests rapides:**

→ `simple_video_demo.py`

### **Pour une application complète:**

→ `video_analysis_template.py`

### **Pour détecter bouche ouverte (visages proches):**

→ `mouth_open_detector.py`

### **Pour détecter bouche ouverte (visages éloignés/multiples distances):**

→ `mouth_open_detector_improved.py`

### **Pour diagnostiquer un problème d'installation:**

→ `test_setup.py`

---

## 🔗 Intégration avec diart

Le dossier parent contient `diart/` qui fait la **diarisation audio**.

**Possibilité de combiner:**

```python
# Vidéo: Qui ouvre la bouche ?
video_faces = video_analyzer.process_frame(frame)

# Audio: Qui produit du son ?
audio_speakers = diart_pipeline.process_audio(audio)

# Fusion: Qui parle vraiment ?
# → Bouche ouverte ET son présent = Parole confirmée
```

**Avantage:**

- Réduction faux positifs (bouche ouverte sans parler)
- Meilleure attribution speaker
- Analyse multimodale robuste

---

## 📈 Évolution du projet

**Chronologie probable:**

1. ✅ Base: `video_analysis_template.py` + `simple_video_demo.py`
2. ✅ Test: `test_setup.py`
3. ✅ Documentation: README files + guides
4. ✅ Nouveau: `mouth_open_detector.py` (version basique)
5. ✅ **Aujourd'hui**: `mouth_open_detector_improved.py` (version optimisée)

**Prochaines étapes possibles:**

- 🔄 Fusion audio-vidéo avec diart
- 👤 Face recognition pour identifier les personnes
- 🎭 Détection d'émotions
- 👁️ Gaze detection (où regarde la personne)
- 🎯 Tracking des visages entre frames
- 📊 Dashboard de statistiques

---

## 🧹 Nettoyage recommandé

```bash
# Supprimer les doublons si identiques
rm "simple_video_demo copy.py"
rm "video_analysis_template copy.py"
rm "requirements_video_analysis copy.txt"

# Ou les archiver
mkdir archive
mv *\ copy.* archive/
```

---

## 📦 Fichiers essentiels vs optionnels

### **ESSENTIELS (ne pas supprimer):**

- ✅ `video_analysis_template.py`
- ✅ `simple_video_demo.py`
- ✅ `mouth_open_detector_improved.py`
- ✅ `requirements_video_analysis.txt`
- ✅ `README_VIDEO.md`
- ✅ `MOUTH_DETECTION_GUIDE.md`

### **UTILES:**

- ✅ `test_setup.py`
- ✅ `mouth_open_detector.py`
- ✅ `VIDEO_ANALYSIS_GUIDE.md`

### **OPTIONNELS/DOUBLONS:**

- ⚠️ `*copy.*` (copies de sauvegarde)
- ⚠️ Fichiers média (peuvent être remplacés)

---

## 🎓 Ressources d'apprentissage

**Pour comprendre MediaPipe:**

- Face Detection: https://google.github.io/mediapipe/solutions/face_detection
- Face Mesh: https://google.github.io/mediapipe/solutions/face_mesh
- 468 landmarks: https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md

**Pour comprendre le MAR:**

- Basé sur EAR (Eye Aspect Ratio) pour clignements
- Paper: "Real-Time Eye Blink Detection using Facial Landmarks"
- Adapté pour la bouche

**Pour OpenCV:**

- Tutoriels Python: https://docs.opencv.org/master/d6/d00/tutorial_py_root.html
- VideoCapture: https://docs.opencv.org/master/d8/dfe/classcv_1_1VideoCapture.html

---

## 💡 Tips

### **Performance:**

```python
# Traiter en résolution réduite
frame = cv2.resize(frame, (640, 480))

# Skip frames
if frame_count % 2 == 0:
    process_frame(frame)
```

### **Debugging:**

```python
# Afficher les landmarks
draw_landmarks = True

# Afficher stats détaillées
show_stats = True

# Sauvegarder frames problématiques
if problematic_detection:
    cv2.imwrite(f"debug_frame_{frame_count}.jpg", frame)
```

### **Production:**

```python
# Désactiver affichage pour performance
analyzer.run(display=False, output_path="output.mp4")

# Limiter le nombre de visages
max_num_faces=3  # Au lieu de 10

# Utiliser GPU si disponible (MediaPipe)
# Certaines versions supportent GPU automatiquement
```
