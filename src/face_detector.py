import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        # Visage et Profil
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # --- NOUVEAU : Détecteur de Sourire Spécifique ---
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    def process(self, image):
        # Optimisation : On travaille sur une petite image pour aller VITE
        # On divise la taille par 2 pour la détection (4x plus rapide)
        small_frame = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Facteur d'échelle pour remettre les coordonnées à la bonne taille
        scale = 2 

        # 1. Détection Visage
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        detected = False
        orientation = "front"
        x, y, w, h = 0, 0, 0, 0

        if len(faces) > 0:
            (sx, sy, sw, sh) = faces[0]
            # On remet à l'échelle originale
            x, y, w, h = sx*scale, sy*scale, sw*scale, sh*scale
            detected = True
        else:
            # Profil
            profiles = self.profile_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            if len(profiles) > 0:
                (sx, sy, sw, sh) = profiles[0]
                x, y, w, h = sx*scale, sy*scale, sw*scale, sh*scale
                detected = True
                orientation = "left"
            else:
                # Profil inversé
                flipped_gray = cv2.flip(gray, 1)
                profiles_flipped = self.profile_cascade.detectMultiScale(flipped_gray, 1.1, 5, minSize=(30, 30))
                if len(profiles_flipped) > 0:
                    (sx, sy, sw, sh) = profiles_flipped[0]
                    # Calcul inverse un peu plus complexe à cause du flip + resize
                    real_w = image.shape[1]
                    x = real_w - (sx*scale) - (sw*scale)
                    y, w, h = sy*scale, sw*scale, sh*scale
                    detected = True
                    orientation = "right"

        data = {'detected': False}

        if detected:
            # Zone du visage en Gris (Taille réelle pour précision)
            roi_gray = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            
            # --- YEUX ---
            eyes_open = 1.0
            if orientation == "front":
                # On cherche dans la moitié haute du visage
                roi_eyes = roi_gray[0:int(h/2), :]
                eyes = self.eye_cascade.detectMultiScale(roi_eyes, 1.1, 3)
                eyes_open = 1.0 if len(eyes) >= 1 else 0.0

            # --- BOUCHE & SOURIRE ---
            # Zone basse du visage
            mouth_roi_y = int(h * 0.65)
            mouth_roi_h = int(h * 0.35)
            roi_mouth = roi_gray[mouth_roi_y:mouth_roi_y+mouth_roi_h, :]

            # 1. Ouverture (Pixels noirs)
            _, thresh = cv2.threshold(roi_mouth, 60, 255, cv2.THRESH_BINARY_INV)
            non_zero = cv2.countNonZero(thresh)
            mouth_openness = min(1.0, (non_zero / (roi_mouth.size + 1)) * 12)

            # 2. Détection SOURIRE (Haar Cascade)
            # On cherche un sourire dans la zone basse
            # scaleFactor=1.7 (très sévère pour éviter les faux positifs)
            # minNeighbors=20 (il faut que ce soit un sourire franc)
            smiles = self.smile_cascade.detectMultiScale(roi_mouth, scaleFactor=1.7, minNeighbors=20)
            is_smiling = len(smiles) > 0

            data = {
                'detected': True,
                'x': x, 'y': y, 'w': w, 'h': h,
                'frame_w': image.shape[1],
                'orientation': orientation,
                'left_openness': eyes_open,
                'right_openness': eyes_open,
                'mouth_openness': mouth_openness,
                'is_smiling': is_smiling,  # <--- Nouvelle info
                'debug_box': (x, y, w, h)
            }
            
        return data
