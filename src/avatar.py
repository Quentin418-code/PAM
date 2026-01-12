import cv2
import numpy as np

class AvatarRenderer:
    def __init__(self, size=600):
        self.w = size
        self.h = size
        self.bg_color = (20, 20, 25)
        
        # Couleurs
        self.skin_color = (180, 200, 255) 
        self.hair_color = (30, 30, 30)    
        self.eye_white = (245, 245, 245)
        self.iris_color = (100, 50, 0)    
        self.lip_color = (130, 130, 200)  

    def draw(self, data):
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        canvas[:] = self.bg_color

        if not data.get('detected'):
            cv2.putText(canvas, "Searching...", (self.w//2 - 80, self.h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100,100,100), 2)
            return canvas

        # --- GEOMETRIE ---
        center_x = self.w // 2
        center_y = self.h // 2
        vid_w = data.get('frame_w', 640)
        face_ratio = data['w'] / vid_w
        base_scale = int(self.w * face_ratio * 1.3)
        base_scale = max(90, min(base_scale, 280))

        orientation = data.get('orientation', 'front')
        look_shift = 0
        if orientation == 'left': look_shift = -int(base_scale * 0.2)
        elif orientation == 'right': look_shift = int(base_scale * 0.2)

        # 1. CHEVEUX (ARRIERE)
        cv2.circle(canvas, (center_x + look_shift//2, center_y - int(base_scale*0.1)), 
                   int(base_scale * 0.65), self.hair_color, -1)

        # 2. VISAGE
        face_h = int(base_scale * 0.6)
        face_w = int(base_scale * 0.48)
        cv2.ellipse(canvas, (center_x + look_shift//3, center_y), 
                    (face_w, face_h), 
                    0, 0, 360, self.skin_color, -1)

        # 3. YEUX
        eye_y = center_y - int(face_h * 0.15)
        eye_off_x = int(face_w * 0.45)
        eye_w = int(face_w * 0.25)
        eye_h = int(face_h * 0.18)
        
        left_pos = (center_x - eye_off_x + look_shift, eye_y)
        right_pos = (center_x + eye_off_x + look_shift, eye_y)

        def draw_eye(pos, is_open):
            if is_open:
                cv2.ellipse(canvas, pos, (eye_w, eye_h), 0, 0, 360, self.eye_white, -1)
                iris_pos = (pos[0] + look_shift//2, pos[1])
                cv2.circle(canvas, iris_pos, int(eye_h * 0.6), self.iris_color, -1)
                cv2.circle(canvas, iris_pos, int(eye_h * 0.25), (0,0,0), -1)
                cv2.circle(canvas, (iris_pos[0]-2, iris_pos[1]-2), 3, (255,255,255), -1)
            else:
                start = (pos[0] - eye_w + 5, pos[1] + 5)
                end = (pos[0] + eye_w - 5, pos[1] + 5)
                cv2.line(canvas, start, end, (50,40,30), 3)

        draw_eye(left_pos, data['left_openness'] > 0.5)
        draw_eye(right_pos, data['right_openness'] > 0.5)

        # 4. CHEVEUX (MECHE)
        hair_y = center_y - face_h + int(face_h*0.4)
        pts = np.array([
            [center_x - face_w + look_shift, center_y - int(face_h*0.6)], 
            [center_x + face_w + look_shift, center_y - int(face_h*0.6)], 
            [center_x + look_shift, hair_y],             
        ], np.int32)
        cv2.fillPoly(canvas, [pts], self.hair_color)

        # 5. BOUCHE INTELLIGENTE (Fixée en haut)
        mouth_fixed_y = center_y + int(face_h * 0.45) 
        mouth_w = int(face_w * 0.5)
        openness = data['mouth_openness']
        is_smiling = data.get('is_smiling', False) # On récupère le détecteur de sourire

        # --- LOGIQUE DE DESSIN ---
        
        # CAS 1 : Bouche Grande Ouverte (Parler / Rire)
        if openness > 0.25:
            drop = int(openness * (face_h * 0.4))
            # Fond sombre
            pts_mouth = np.array([
                [center_x - mouth_w + look_shift//2, mouth_fixed_y], 
                [center_x + mouth_w + look_shift//2, mouth_fixed_y], 
                [center_x + look_shift//2, mouth_fixed_y + drop]
            ], np.int32)
            cv2.ellipse(canvas, (center_x + look_shift//2, mouth_fixed_y), 
                        (mouth_w, drop), 0, 0, 180, (50, 0, 0), -1)
            cv2.line(canvas, (center_x - mouth_w + look_shift//2, mouth_fixed_y),
                             (center_x + mouth_w + look_shift//2, mouth_fixed_y), (50,0,0), 2)
            # Dents
            cv2.rectangle(canvas, 
                          (center_x + look_shift//2 - int(mouth_w*0.7), mouth_fixed_y), 
                          (center_x + look_shift//2 + int(mouth_w*0.7), mouth_fixed_y + 8), 
                          (240,240,240), -1)

        # CAS 2 : Sourire détecté (même bouche fermée)
        elif is_smiling:
            # On dessine un arc vers le HAUT (U)
            center_mouth_x = center_x + look_shift//2
            
            # Ligne du sourire
            cv2.ellipse(canvas, (center_mouth_x, mouth_fixed_y), 
                        (mouth_w, int(face_h * 0.15)), 
                        0, 0, 180, self.lip_color, 4)
            # Fossettes (optionnel)
            cv2.line(canvas, (center_mouth_x - mouth_w - 5, mouth_fixed_y - 5),
                             (center_mouth_x - mouth_w, mouth_fixed_y), (150,100,100), 2)
            cv2.line(canvas, (center_mouth_x + mouth_w + 5, mouth_fixed_y - 5),
                             (center_mouth_x + mouth_w, mouth_fixed_y), (150,100,100), 2)

        # CAS 3 : Tirer la gueule (Entre ouvert, mais pas de sourire détecté)
        elif openness > 0.08:
            center_mouth_x = center_x + look_shift//2
            # Arc vers le BAS
            cv2.ellipse(canvas, (center_mouth_x, mouth_fixed_y + 10), 
                        (mouth_w, int(face_h * 0.15)), 
                        0, 180, 360, self.lip_color, 4)
            # Ombre
            cv2.ellipse(canvas, (center_mouth_x, mouth_fixed_y + 15), 
                        (int(mouth_w*0.6), int(face_h * 0.1)), 
                        0, 180, 360, (150,100,100), 2)

        # CAS 4 : Neutre
        else:
            start_p = (center_x - mouth_w + look_shift//2, mouth_fixed_y)
            end_p = (center_x + mouth_w + look_shift//2, mouth_fixed_y)
            cv2.line(canvas, start_p, end_p, self.lip_color, 4)

        return canvas
