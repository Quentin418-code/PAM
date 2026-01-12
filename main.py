import cv2
import numpy as np
from src.face_detector import FaceDetector
from src.avatar import AvatarRenderer

def main():
    # --- CONFIGURATION ---
    VIDEO_FILE = '12100.mp4'
    
    detector = FaceDetector()
    renderer = AvatarRenderer(size=600) # Fenêtre de l'avatar (carrée)
    
    cap = cv2.VideoCapture(VIDEO_FILE)

    print(f"--- LECTURE VIDEO : {VIDEO_FILE} ---")
    print("Appuie sur 'q' pour quitter.")

    # On crée une fenêtre "intelligente" qu'on pourra redimensionner
    cv2.namedWindow('PAM - Camera', cv2.WINDOW_NORMAL)
    cv2.namedWindow('PAM - Avatar', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        success, frame = cap.read()
        
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # 1. Analyse
        face_data = detector.process(frame)
        
        # 2. Création de l'Avatar
        avatar_img = renderer.draw(face_data)

        # 3. Debug visuel (Carré vert)
        if face_data['detected']:
            (x, y, w, h) = face_data['debug_box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # --- AFFICHAGE ADAPTATIF ---
        
        # On calcule un facteur de réduction pour que la vidéo tienne sur l'écran
        # Si la hauteur dépasse 600px, on la réduit
        target_height = 600
        h_cam, w_cam, _ = frame.shape
        
        if h_cam > target_height:
            scale = target_height / h_cam
            new_w = int(w_cam * scale)
            new_h = int(h_cam * scale)
            display_frame = cv2.resize(frame, (new_w, new_h))
        else:
            display_frame = frame

        cv2.imshow('PAM - Camera', display_frame)
        cv2.imshow('PAM - Avatar', avatar_img)

        # Placement des fenêtres (facultatif, pour éviter le chevauchement)
        # cv2.moveWindow('PAM - Camera', 0, 0)
        # cv2.moveWindow('PAM - Avatar', display_frame.shape[1] + 10, 0)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
