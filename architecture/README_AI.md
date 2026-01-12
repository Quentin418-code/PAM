# 🧠 PAM Project Architecture (Digital Twin)

Ce document décrit l'architecture technique du projet PAM pour permettre à une IA de comprendre rapidement le contexte, les dépendances et le flux de données.

## 📌 État Actuel : "Compatibility Mode"
Le projet utilise actuellement **OpenCV Native (Haar Cascades)** au lieu de MediaPipe pour assurer une compatibilité maximale (problèmes rencontrés avec Python 3.12 + MediaPipe sous Linux).
Nous utilisons une **vidéo pré-enregistrée** (`12099.mp4`) en entrée car la webcam n'est pas détectée sur la machine hôte.

## 📂 Structure des Fichiers

### `main.py` (Orchestrateur)
* **Rôle :** Point d'entrée. Charge la vidéo, initialise les modules, gère la boucle principale et l'affichage (GUI).
* **Logique :**
    1.  Lit une frame de la vidéo.
    2.  Envoie la frame à `FaceDetector`.
    3.  Reçoit les données d'analyse (position, ouverture bouche/yeux).
    4.  Envoie ces données à `AvatarRenderer`.
    5.  Affiche deux fenêtres OpenCV (`Camera` et `Avatar`).
* **Spécificité :** Gère le redimensionnement de l'affichage pour éviter que les vidéos 4K ne dépassent de l'écran.

### `src/face_detector.py` (Vision)
* **Rôle :** Analyse l'image pour extraire les metrics du visage.
* **Technologie :** `cv2.CascadeClassifier` (Haar Cascades).
* **Sortie (Dictionnaire `data`) :**
    * `detected` (bool) : Visage trouvé ?
    * `x, y, w, h` : Bounding box du visage.
    * `frame_w, frame_h` : Dimensions de la vidéo source (pour le ratio).
    * `left_openness`, `right_openness` (0.0 ou 1.0) : Détection binaire des yeux (basée sur `haarcascade_eye`).
    * `mouth_openness` (float 0.0 -> 1.0) : Calculée par **thresholding** (comptage de pixels noirs dans le tiers inférieur du visage).

### `src/avatar.py` (Rendu)
* **Rôle :** Dessine l'avatar vectoriel (cercles, lignes) sur un canvas noir.
* **Logique :**
    * **Centrage forcé :** L'avatar reste au centre de sa fenêtre (300, 300).
    * **Zoom adaptatif :** La taille de la tête dépend du ratio `largeur_visage / largeur_video` (plus on est près, plus c'est gros).
    * **Animation :** Les yeux et la bouche réagissent aux données du détecteur.

## 🔄 Flux de Données (Data Flow)

1.  **Input :** `frame` (Image BGR depuis `12099.mp4`)
2.  **Processing :** `FaceDetector.process(frame)` -> `face_data` (Dict)
3.  **Rendering :** `AvatarRenderer.draw(face_data)` -> `avatar_img` (Image BGR)
4.  **Output :** Affichage via `cv2.imshow`.

## ⚠️ Notes pour l'IA suivante
* Si vous devez repasser sur **MediaPipe**, il faut gérer le conflit de version `protobuf` et l'importation `mp.solutions` sur Python 3.12.
* Le fichier `src/geometry.py` est actuellement **inutilisé** dans cette version Haar Cascade (il servait pour les calculs d'angles Vectoriels de MediaPipe).
