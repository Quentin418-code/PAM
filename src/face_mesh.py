import cv2
import numpy as np
import mediapipe as mp

class FaceMeshDetector:
    def __init__(self, static_mode=False):
        # Initialisation standard
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # static_image_mode=True rend la détection BEAUCOUP plus puissante sur les images fixes
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_mode, 
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, image):
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(rgb_image)

        data = {'detected': False}

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            points = []
            for lm in landmarks.landmark:
                pt_x = int(lm.x * w)
                pt_y = int(lm.y * h)
                points.append((pt_x, pt_y))
            
            points = np.array(points, np.int32)
            rect = cv2.boundingRect(points)
            
            # 13: Haut, 14: Bas, 78: Gauche, 308: Droite
            up = points[13]
            down = points[14]
            left = points[78]
            right = points[308]
            
            h_mouth = np.linalg.norm(up - down)
            w_mouth = np.linalg.norm(left - right)
            ratio = h_mouth / w_mouth if w_mouth > 0 else 0
            
            data = {
                'detected': True,
                'landmarks': points,
                'mouth_openness': ratio,
                'debug_box': rect,
                'frame_w': w,
                'frame_h': h
            }

        return data
