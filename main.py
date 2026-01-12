import cv2
import numpy as np
from src.face_mesh import FaceMeshDetector
from src.avatar import AvatarRenderer

def main():
    VIDEO_FILE = '12100.mp4'
    
    # Initialisation du moteur Mesh
    detector = FaceMeshDetector()
    
    # Le renderer va scanner mask.png au démarrage, surveille le terminal !
    renderer = AvatarRenderer(size=600)
    
    cap = cv2.VideoCapture(VIDEO_FILE)

    print("--- PAM MESH ENGINE STARTED ---")
    print("Si ça rame, c'est normal : le calcul 3D est intensif.")

    cv2.namedWindow('PAM - Avatar', cv2.WINDOW_NORMAL)
    cv2.namedWindow('PAM - Camera', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # 1. Analyse Mesh (468 points)
        face_data = detector.process(frame)
        
        # 2. Rendu Warp
        avatar_img = renderer.draw(face_data)

        # 3. Debug Caméra
        if face_data['detected']:
            pts = face_data['landmarks']
            # On dessine quelques points pour montrer que le tracking marche
            # (Ex: Contour visage et lèvres)
            for i in range(0, len(pts), 5): # Un point sur 5 pour pas surcharger
                cv2.circle(frame, (pts[i][0], pts[i][1]), 1, (0, 255, 0), -1)
            
            (x, y, w, h) = face_data['debug_box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('PAM - Avatar', avatar_img)
        
        h_cam, w_cam = frame.shape[:2]
        if h_cam > 600:
            frame = cv2.resize(frame, (int(w_cam * (600/h_cam)), 600))
        cv2.imshow('PAM - Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
