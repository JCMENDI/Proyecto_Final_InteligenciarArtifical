import cv2
import os
import time
from esp32_client import adelante, atras, derecha, izquierda, frenar


CAM_URL = "http://192.168.0.20:8000/video"

# Carpetas del dataset
CLASES = ["RECTA", "CURVA_IZQ", "CURVA_DER", "GIRO_90_IZQ", "GIRO_90_DER", "CRUCE_T"]
BASE_DIR = "dataset"

for clase in CLASES:
    os.makedirs(os.path.join(BASE_DIR, clase), exist_ok=True)


cap = cv2.VideoCapture(CAM_URL)
frame_count = {c: 0 for c in CLASES}
clase_actual = None

print("""
=== CAPTURA DE DATASET ===
W = RECTA
A = CURVA_IZQ
D = CURVA_DER
Z = GIRO_90_IZQ
X = GIRO_90_DER
T = CRUCE_T
F = Frenar (sin guardar)
Q = Salir
      """)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error con la camara")
        break

    # Redimensionar para la CNN
    frame_resized = cv2.resize(frame, (224, 224))

    # Mostrar info en pantalla
    info = f"Clase: {clase_actual} | " + " | ".join({f"{c}:{frame_count[c]}" for c in CLASES})
    cv2.putText(frame, info[:80], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 1)
    cv2.imshow("Captura Dataset", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('w'):
        clase_actual = "RECTA"
        adelante()
    elif key == ord('a'):
        clase_actual = "CURVA_IZQ"
        izquierda()
    elif key == ord('d'):
        clase_actual = "CURVA_DER"
        derecha()
    elif key == ord('z'):
        clase_actual = "GIRO_90_IZQ"
        izquierda()
    elif key == ord('x'):
        clase_actual = "GIRO_90_DER"
        derecha()
    elif key == ord('t'):
        clase_actual = "CRUCE_T"
        frenar()
    elif key == ord('f'):
        clase_actual = None
        frenar()
    elif key == ord('q'):
        frenar()
        break

    # Guardar frame si hay clase activa
    if clase_actual:
        nombre = f"{clase_actual}_{int(time.time()*1000)}.jpg"
        ruta = os.path.join(BASE_DIR, clase_actual, nombre)
        cv2.imwrite(ruta, frame_resized)
        frame_count[clase_actual] += 1

cap.release()
cv2.destroyAllWindows()

print("\nResumen final:")
for c in CLASES:
    print(f"{c}: {frame_count[c]} imágenes")