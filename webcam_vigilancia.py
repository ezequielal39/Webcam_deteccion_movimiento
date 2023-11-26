import cv2
import numpy as np
import os
from datetime import datetime

# Crear la carpeta si no existe
if not os.path.exists('imagenes_capturadas'):
    os.makedirs('imagenes_capturadas')

# Inicializar la cámara web
cap = cv2.VideoCapture(0)

# parámetros para la grabación de video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = 20  # Ajusta esto según la cámara

# Leer dos marcos iniciales
_, frame1 = cap.read()
_, frame2 = cap.read()

# Variables para controlar la grabación
recording = False
record_end_time = None

while True:
    # Convertir imágenes a escala de grises
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calcular la diferencia y aplicar un umbral
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Buscar contornos en el umbral
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            if not recording:
                # Comenzar la grabación
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_filename = f"imagenes_capturadas/movimiento_{timestamp}.avi"
                out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
                record_end_time = datetime.now() + timedelta(seconds=5)
                recording = True
            break

    if recording:
        # Grabar el marco actual
        out.write(frame1)
        if datetime.now() >= record_end_time:
            # Detener la grabación después de 5 segundos
            out.release()
            recording = False

    # Mostrar el marco
    cv2.imshow("Frame", frame1)

    # Leer un nuevo marco
    _, frame1 = cap.read()

    # Esperar que se presione la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Hacer que el nuevo marco sea el marco anterior
    frame2 = frame1

# Liberar la cámara y cerrar todas las ventanas
if recording:
    out.release()
cap.release()
cv2.destroyAllWindows()
