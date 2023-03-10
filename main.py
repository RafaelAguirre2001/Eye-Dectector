import cv2
import numpy as np

# Carga los clasificadores pre-entrenados de OpenCV para la detección de ojos
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
eye_glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# Crea un objeto de captura de video utilizando la cámara
cap = cv2.VideoCapture(0)

while True:
    # Lee el cuadro de video actual
    ret, frame = cap.read()

    # Convierte el cuadro a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta los ojos en el cuadro
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    glasses = eye_glasses_cascade.detectMultiScale(gray, 1.3, 5)

    # Dibuja un rectángulo alrededor de cada ojo detectado
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for (x, y, w, h) in glasses:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    # Muestra el cuadro de video con los rectángulos dibujados
    cv2.imshow('Detector de Ojo', frame)

    # Si se presiona la tecla 'q', sale del bucle
    if cv2.waitKey(1) == ord('q'):
        break

# Libera el objeto de captura y cierra todas las ventanas
cap.release()
cv2.destroyAllWindows()
