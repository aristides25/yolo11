from ultralytics import YOLO
import cv2

# 1. Cargar el modelo YOLO para pose estimation
print("Cargando modelo YOLO11-pose...")
try:
    # Intentar cargar el modelo entrenado, si existe
    model = YOLO("runs/pose/train/weights/best.pt")
    print("Modelo entrenado de poses cargado exitosamente!")
except:
    # Si no existe, usar el modelo preentrenado
    print("Usando modelo preentrenado de poses...")
    model = YOLO("yolo11n-pose.pt")
    print("Modelo preentrenado de poses cargado!")

# 2. Inicializar la cámara
print("Iniciando cámara...")
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Lista de keypoints para referencia
keypoints = {
    0: "nariz", 1: "ojo_izq", 2: "ojo_der", 
    3: "oreja_izq", 4: "oreja_der",
    5: "hombro_izq", 6: "hombro_der",
    7: "codo_izq", 8: "codo_der",
    9: "muñeca_izq", 10: "muñeca_der",
    11: "cadera_izq", 12: "cadera_der",
    13: "rodilla_izq", 14: "rodilla_der",
    15: "tobillo_izq", 16: "tobillo_der"
}

print("Iniciando detección de poses en tiempo real... Presiona 'q' para salir.")
while True:
    # Capturar frame
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el frame.")
        break

    # Realizar la detección de poses
    results = model.predict(frame, conf=0.45, show=False)
    
    # Obtener el frame con las anotaciones
    annotated_frame = results[0].plot()
    
    # Mostrar información de keypoints detectados
    for result in results:
        if result.keypoints is not None:
            keypoints_data = result.keypoints.data[0]
            for idx, kp in enumerate(keypoints_data):
                x, y, conf = kp
                if conf > 0.5:  # Solo mostrar keypoints con confianza > 50%
                    cv2.putText(annotated_frame, f"{keypoints[idx]}", 
                              (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 0), 1)

    # Mostrar el frame
    cv2.imshow("YOLO11 Pose Estimation", annotated_frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows() 