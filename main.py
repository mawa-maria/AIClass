import cv2
import numpy as np
import os
import pandas as pd
import time
from datetime import datetime

# --- Paramètres ---
image_folder = "etudiants_images"
MAX_DURATION = 30  # Temps max en secondes avant arrêt automatique

# Détection de visages
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

faces = []
labels = []
label_dict = {}
current_label = 0

# Charger les images des étudiants
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        student_name = os.path.splitext(filename)[0]
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces_rect:
            faces.append(gray[y:y+h, x:x+w])
            labels.append(current_label)
            label_dict[current_label] = student_name
            break
        current_label += 1

if not faces:
    print("Aucun visage trouvé dans le dataset.")
    exit()

# Entraînement du modèle de reconnaissance
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

# Dictionnaire de présence
presence = {nom: "Non" for nom in label_dict.values()}

# --- Détection en temps réel ---
cap = cv2.VideoCapture(0)
start_time = time.time()  # Démarrer le chronomètre

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_rect = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces_rect:
        face_roi = gray_frame[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_roi)
        if confidence < 100:
            student_name = label_dict.get(label, "Inconnu")
            presence[student_name] = "Oui"
            cv2.putText(frame, student_name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Detection de Presence", frame)
    
    # Vérifier le temps écoulé
    elapsed_time = time.time() - start_time
    if elapsed_time > MAX_DURATION:
        print(f"Arrêt automatique après {MAX_DURATION} secondes.")
        break
    
    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Arrêt manuel par l'utilisateur.")
        break

cap.release()
cv2.destroyAllWindows()

# --- Enregistrement de la liste de présence ---
date_str = datetime.now().strftime("%Y-%m-%d")
csv_filename = f"presence_{date_str}.csv"
df = pd.DataFrame({
    "Nom": list(presence.keys()),
    "Présence": list(presence.values()),
    "Date": date_str
})
df.to_csv(csv_filename, index=False)
print(f"Présence enregistrée dans {csv_filename}")
