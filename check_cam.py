import cv2

print("--- RECHERCHE DE CAMERA ---")
available_cams = []

# On teste les ports de 0 à 5
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"[✅] Caméra trouvée à l'index {i}")
            available_cams.append(i)
        else:
            print(f"[❌] Index {i} s'ouvre mais ne renvoie pas d'image (Occupée ?)")
        cap.release()
    else:
        print(f"[ ] Pas de caméra à l'index {i}")

if len(available_cams) == 0:
    print("\n[ERREUR CRITIQUE] Aucune caméra détectée.")
    print("Vérifie :")
    print("1. Si un bouton physique bloque la cam")
    print("2. Si tu es sur une Machine Virtuelle (WSL/VirtualBox) ?")
else:
    print(f"\n[SUCCÈS] Utilise l'index {available_cams[0]} dans ton code !")
