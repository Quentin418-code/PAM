import cv2
import numpy as np
from src.face_mesh import FaceMeshDetector
from src.avatar import AvatarRenderer

def main():
    # --- CONFIGURATION CAMÉRA ---
    # 0 = Généralement la webcam par défaut ou DroidCam
    # 1 = Essaie ça si le 0 ne marche pas (ou si tu as une autre webcam branchée)
    # '12100.mp4' = Remets le nom du fichier si tu veux revenir à la vidéo
    VIDEO_SOURCE = 0
    
    detector = FaceMeshDetector()
    renderer = AvatarRenderer(size=600)
    
    print("--- PAM LIVE SYSTEM ---")
    print(f"Connexion à la source vidéo {VIDEO_SOURCE}...")
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    # On demande une bonne résolution au téléphone (HD 720p)
    # Si le téléphone ne supporte pas, il prendra sa résolution par défaut
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("ERREUR CRITIQUE: Impossible d'ouvrir la caméra.")
        print("1. Vérifie que DroidCam (PC) et l'appli (Tel) sont connectés.")
        print("2. Essaie de changer VIDEO_SOURCE = 1 dans le code.")
        return

    cv2.namedWindow('PAM - Avatar', cv2.WINDOW_NORMAL)
    cv2.namedWindow('PAM - Live', cv2.WINDOW_NORMAL)

    print("--- SYSTÈME PRÊT ---")
    print("Appuie sur 'q' pour quitter.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Fin du flux ou erreur de lecture.")
            # Si c'est une vidéo en boucle, on rembobine. Sinon on quitte.
            if isinstance(VIDEO_SOURCE, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        # EFFET MIROIR (Indispensable pour le direct)
        # 1 = Flip horizontal (comme un miroir)
        if isinstance(VIDEO_SOURCE, int): 
            frame = cv2.flip(frame, 1)

        # 1. Analyse (Détection 3D)
        face_data = detector.process(frame)
        
        # 2. Rendu (Warping + Adaptation à la taille de la caméra)
        avatar_img = renderer.draw(face_data)

        # 3. Affichage
        cv2.imshow('PAM - Avatar', avatar_img)
        
        # Affichage du retour caméra (en petit pour contrôle)
        h, w = frame.shape[:2]
        display_w = 480
        display_h = int(h * (display_w / w))
        small_frame = cv2.resize(frame, (display_w, display_h))
        cv2.imshow('PAM - Live', small_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
