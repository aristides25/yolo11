from ultralytics import YOLO
import cv2

# 1. Cargar el modelo entrenado
print("Cargando modelo entrenado...")
# Intentar cargar el último modelo entrenado, si no existe, usar el preentrenado
try:
    model = YOLO("runs/detect/train/weights/best.pt")  # Usar el mejor modelo entrenado
    print("Modelo entrenado cargado exitosamente!")
except:
    print("No se encontró modelo entrenado, usando modelo preentrenado...")
    model = YOLO("yolo11n.pt")
    print("Modelo preentrenado cargado exitosamente!")

# 2. Inicializar la cámara USB
print("Iniciando cámara...")
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

print("Iniciando detección en tiempo real... Presiona 'q' para salir.")
while True:
    # Capturar un frame de la cámara
    ret, frame = cap.read()

    # Si no se pudo capturar el frame, salir del bucle
    if not ret:
        print("Error: No se pudo capturar el frame.")
        break

    # Realizar la detección de objetos en el frame actual
    results = model(frame, conf=0.45)  # Umbral de confianza del 45%

    # Mostrar los resultados en una ventana
    annotated_frame = results[0].plot()
    cv2.imshow("Detección en tiempo real", annotated_frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()