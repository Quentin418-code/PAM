import cv2
import numpy as np
import src.face_mesh as fm

class AvatarRenderer:
    def __init__(self, size=600):
        self.w = size
        self.h = size
        self.bg_color = (5, 5, 10)
        
        self.mask_img = None
        self.mask_landmarks = None
        self.triangles = None
        self.is_ready = False
        
        # Indices des points qui forment le contour INTERIEUR de la bouche
        # C'est une boucle standard de MediaPipe
        self.inner_mouth_indices = [
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, # Lèvre haut (gauche vers droite)
            324, 318, 402, 317, 14, 87, 178, 88, 95          # Lèvre bas (droite vers gauche)
        ]
        
        self.mask_scanner = fm.FaceMeshDetector(static_mode=True)
        self.load_and_mesh_mask()

    def load_and_mesh_mask(self):
        try:
            print("--- CHARGEMENT AVATAR (mask.png) ---")
            img = cv2.imread('mask.png', cv2.IMREAD_UNCHANGED)
            if img is None: 
                print("ERREUR: 'mask.png' introuvable.")
                return

            if img.shape[2] == 4:
                rgb_for_scan = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            else:
                rgb_for_scan = img

            data = self.mask_scanner.process(rgb_for_scan)
            if not data['detected']:
                print("ERREUR: Visage non détecté sur mask.png.")
                return

            self.mask_img = img
            self.mask_landmarks = data['landmarks']

            rect = (0, 0, self.mask_img.shape[1], self.mask_img.shape[0])
            subdiv = cv2.Subdiv2D(rect)
            
            for p in self.mask_landmarks:
                if 0 <= p[0] < self.mask_img.shape[1] and 0 <= p[1] < self.mask_img.shape[0]:
                    subdiv.insert((float(p[0]), float(p[1])))
                
            triangle_list = subdiv.getTriangleList()
            
            self.triangles = []
            for t in triangle_list:
                pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
                indices = []
                for pt in pts:
                    if pt[0] < 0 or pt[0] >= self.mask_img.shape[1] or pt[1] < 0 or pt[1] >= self.mask_img.shape[0]:
                        continue
                    min_dist = 2.0 
                    found_idx = -1
                    for i, lm in enumerate(self.mask_landmarks):
                        dist = abs(lm[0] - pt[0]) + abs(lm[1] - pt[1])
                        if dist < min_dist:
                            found_idx = i
                            break
                    if found_idx != -1:
                        indices.append(found_idx)
                
                if len(indices) == 3:
                    self.triangles.append(indices)
            
            print(f"--- MESH GÉNÉRÉ : {len(self.triangles)} polygones ---")
            self.is_ready = True

        except Exception as e:
            print(f"CRASH INIT: {e}")

    def warp_triangle(self, img1, img2, t1, t2):
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))

        if r2[2] <= 0 or r2[3] <= 0: return 
        h, w = img2.shape[:2]
        x1, y1 = max(0, r2[0]), max(0, r2[1])
        x2, y2 = min(w, r2[0]+r2[2]), min(h, r2[1]+r2[3])
        if x1 >= x2 or y1 >= y2: return

        off_x, off_y = x1 - r2[0], y1 - r2[1]
        w_crop, h_crop = x2 - x1, y2 - y1

        t1_rect = []
        t2_rect = []
        t2_rect_int = []

        for i in range(3):
            t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
            t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
            t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

        img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        size = (r2[2], r2[3])
        
        warp_mat = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
        img2_rect = cv2.warpAffine(img1_rect, warp_mat, size, None, 
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        img2_crop = img2_rect[off_y : off_y+h_crop, off_x : off_x+w_crop]
        mask_crop = mask[off_y : off_y+h_crop, off_x : off_x+w_crop]
        
        dest = img2[y1:y2, x1:x2]
        dest[:] = dest * ((1.0, 1.0, 1.0) - mask_crop)
        dest[:] = dest + (img2_crop * mask_crop)

    def draw(self, data):
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        canvas[:] = self.bg_color

        if not self.is_ready or not data['detected']:
            return canvas

        # Auto-Scaling
        cam_w = data.get('frame_w', self.w)
        cam_h = data.get('frame_h', self.h)
        scale = self.h / cam_h
        scaled_w = cam_w * scale
        offset_x = (self.w - scaled_w) // 2

        landmarks_scaled = []
        for lm in data['landmarks']:
            lx = int(lm[0] * scale + offset_x)
            ly = int(lm[1] * scale)
            landmarks_scaled.append((lx, ly))

        img_source = self.mask_img[:, :, :3]
        warped_face = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # 1. Warping du visage
        for triangle_indices in self.triangles:
            t1 = []
            t2 = []
            for index in triangle_indices:
                t1.append(self.mask_landmarks[index])
                t2.append(landmarks_scaled[index])
            self.warp_triangle(img_source, warped_face, t1, t2)

        # --- FIX BOUCHE : LE TROU NOIR ---
        # On récupère les points du contour intérieur de la bouche (scalés)
        mouth_hole_points = []
        for idx in self.inner_mouth_indices:
            mouth_hole_points.append(landmarks_scaled[idx])
            
        # On remplit cette zone avec une couleur sombre (Bleu nuit Na'vi)
        # Couleur BGR : (Bleu, Vert, Rouge) -> (40, 20, 30)
        if len(mouth_hole_points) > 0:
             cv2.fillPoly(warped_face, [np.array(mouth_hole_points, np.int32)], (40, 20, 30))
        # ---------------------------------

        warped_gray = cv2.cvtColor(warped_face, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(warped_gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        bg = cv2.bitwise_and(canvas, canvas, mask=mask_inv)
        fg = cv2.bitwise_and(warped_face, warped_face, mask=mask)
        
        return cv2.add(bg, fg)
