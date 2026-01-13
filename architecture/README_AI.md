# 🧠 PAM Project Architecture - Avatar 3D (Mesh Warp Engine)

## 📌 État Actuel : "Mesh Warp Engine" (Mapping 3D)
Cette branche `Avatar-3D` abandonne les méthodes 2D (Puppet/Slicing) pour utiliser la **déformation de maillage par triangulation**.
L'image de l'avatar est transformée en une "peau" flexible qui est épinglée sur les 468 points du visage de l'utilisateur.

## 🔄 Évolution (Avant / Après)

| Feature | Ancienne Arch. (Puppet) | Nouvelle Arch. (Mesh Warp) |
| :--- | :--- | :--- |
| **Moteur** | OpenCV (Haar Cascades) | **MediaPipe Face Mesh** |
| **Précision** | Rectangle (X, Y, W, H) | **468 Landmarks (3D)** |
| **Rendu** | Découpage d'image (Haut/Bas) | **Déformation Triangulaire (Warp)** |
| **Mouvement** | Parallaxe 2D (Gauche/Droite) | **Suivi 3D complet** (Pitch, Yaw, Roll) |
| **Expressions** | Juste ouverture bouche | **Sourires, Grimaces, Yeux, Bouche** |

## 🛠️ Stack Technique & Versioning (CRITIQUE)
En raison de conflits entre Python 3.12, MediaPipe et Protobuf, les versions suivantes sont **impératives** :

* **Python :** 3.12+
* **MediaPipe :** `0.10.14` (Stabilité)
* **Protobuf :** `<4` (ex: `3.20.3`) - *Incompatible avec v4/v5*
* **OpenCV :** `opencv-python` (Standard) - *Ne pas installer headless*

## 📂 Structure des Modules

### 1. `src/face_mesh.py` (Le Radar)
* **Rôle :** Scanne le visage et retourne une carte de points.
* **Tech :** `mp.solutions.face_mesh` avec `refine_landmarks=True`.
* **Mode Statique :** Utilisé au démarrage pour scanner `mask.png` avec haute précision.
* **Mode Stream :** Utilisé en boucle pour scanner la webcam (rapide).
* **Output :** Liste de 468 tuples `(x, y)`.

### 2. `src/avatar.py` (Le Moteur de Rendu)
C'est le cœur du système. Il fonctionne en deux temps :

#### A. Initialisation (`__init__`)
1.  Charge `mask.png`.
2.  Scanne le visage du Na'vi sur l'image.
3.  Effectue une **Triangulation de Delaunay** sur les points du Na'vi.
4.  Stocke la liste des triangles (indices des points connectés).

#### B. Boucle de Rendu (`draw`)
Pour chaque frame vidéo :
1.  Récupère les landmarks de l'utilisateur.
2.  **Scaling :** Redimensionne et centre les points utilisateurs pour qu'ils rentrent dans la fenêtre Avatar (600x600).
3.  **Warping :** Pour chaque triangle du maillage :
    * Extrait le triangle de texture du Na'vi.
    * Calcule la matrice de transformation affine vers le triangle utilisateur.
    * Déforme et colle le triangle.
4.  **Composition :** Fusionne le visage déformé sur le fond.

## ⚠️ Notes de Maintenance
* **`mask.png` :** Doit impérativement contenir un visage détectable de face. Si l'écran reste noir ou affiche "LOADING", c'est que l'IA ne reconnaît pas le visage sur l'image source.
* **Bords d'écran :** Une sécurité "Clipping" est active dans `warp_triangle` pour éviter les crashs si le visage sort du cadre.


## 📱 Mode Live (Intégration Smartphone/DroidCam)
Le projet supporte désormais l'utilisation d'un smartphone comme caméra HD via **DroidCam** (Linux).

### Pré-requis
1.  **Smartphone :** Installer l'application **DroidCam** (Android/iOS).
2.  **PC (Linux) :** Installer le client et le module vidéo :
    ```bash
    cd /tmp/
    wget -O droidcam_latest.zip [https://files.dev47apps.net/linux/droidcam_2.1.3.zip](https://files.dev47apps.net/linux/droidcam_2.1.3.zip)
    unzip droidcam_latest.zip -d droidcam
    cd droidcam && sudo ./install-client && sudo ./install-video
    ```

### Procédure de Connexion
1.  Lancer DroidCam sur le téléphone. Notez l'IP WiFi (ex: `192.168.x.x`).
    * *Attention : Ne pas utiliser l'IP Mobile `10.x.x.x`.*
2.  Lancer le client PC : `droidcam`.
3.  Entrer l'IP du téléphone et cliquer sur **Connect**.
4.  Vérifier que le flux vidéo apparaît sur le PC.

### Configuration du Code (`main.py`)
Le script détecte automatiquement la source vidéo.
* **`VIDEO_SOURCE = 0`** : Caméra par défaut (souvent DroidCam).
* **`VIDEO_SOURCE = 1`** : À tester si l'écran reste noir (conflit webcam interne).
* **Scaling Auto :** Le fichier `src/avatar.py` redimensionne automatiquement les coordonnées HD du téléphone pour la fenêtre de rendu (600x600).
