import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def process(self, image):
        h_img, w_img, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Détection (Face ou Profil)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        detected = False
        orientation = "front"
        x, y, w, h = 0, 0, 0, 0

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            detected = True
        else:
            profiles = self.profile_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            if len(profiles) > 0:
                (x, y, w, h) = profiles[0]
                detected = True
                orientation = "left"
            else:
                flipped_gray = cv2.flip(gray, 1)
                profiles_flipped = self.profile_cascade.detectMultiScale(flipped_gray, 1.1, 5, minSize=(30, 30))
                if len(profiles_flipped) > 0:
                    (x_f, y_f, w_f, h_f) = profiles_flipped[0]
                    x = w_img - x_f - w_f
                    y, w, h = y_f, w_f, h_f
                    detected = True
                    orientation = "right"

        data = {'detected': False}

        if detected:
            # --- YEUX ---
            eyes_open = 1.0
            if orientation == "front":
                roi_gray_eyes = gray[y:y+int(h/2), x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray_eyes, 1.1, 3) # Seuil baissé à 3
                eyes_open = 1.0 if len(eyes) >= 1 else 0.0

            # --- BOUCHE (Sensibilité accrue) ---
            # On se concentre sur le centre bas du visage pour éviter les ombres du menton
            mouth_roi_y = y + int(h * 0.70)
            mouth_roi_h = int(h * 0.20)
            mouth_roi_x = x + int(w * 0.2)
            mouth_roi_w = int(w * 0.6)
            
            mouth_openness = 0.0
            if mouth_roi_y + mouth_roi_h < h_img:
                mouth_roi = gray[mouth_roi_y:mouth_roi_y+mouth_roi_h, mouth_roi_x:mouth_roi_x+mouth_roi_w]
                # Seuil adaptatif pour mieux voir le contraste dents/lèvres
                _, thresh = cv2.threshold(mouth_roi, 70, 255, cv2.THRESH_BINARY_INV)
                non_zero = cv2.countNonZero(thresh)
                # Multiplicateur x15 : la moindre ouverture sera visible !
                mouth_openness = min(1.0, (non_zero / (mouth_roi.size + 1)) * 15)

            data = {
                'detected': True,
                'x': x, 'y': y, 'w': w, 'h': h,
                'frame_w': w_img,
                'orientation': orientation,
                'left_openness': eyes_open,
                'right_openness': eyes_open,
                'mouth_openness': mouth_openness, # 0.0 à 1.0
                'debug_box': (x, y, w, h)
            }
            
        return data
