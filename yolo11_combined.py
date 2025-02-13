from ultralytics import YOLO
import cv2

# 1. Cargar ambos modelos YOLO
print("Cargando modelos YOLO11...")

# Modelo para detección de objetos
try:
    detect_model = YOLO("runs/detect/train/weights/best.pt")
    print("Modelo entrenado de detección cargado exitosamente!")
except:
    print("Usando modelo preentrenado de detección...")
    detect_model = YOLO("yolo11n.pt")
    print("Modelo preentrenado de detección cargado!")

# Modelo para poses
try:
    pose_model = YOLO("runs/pose/train/weights/best.pt")
    print("Modelo entrenado de poses cargado exitosamente!")
except:
    print("Usando modelo preentrenado de poses...")
    pose_model = YOLO("yolo11n-pose.pt")
    print("Modelo preentrenado de poses cargado!")

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

# 2. Inicializar la cámara
print("Iniciando cámara...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

print("Iniciando detección combinada en tiempo real... Presiona 'q' para salir.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el frame.")
        break

    # Realizar ambas detecciones
    detect_results = detect_model(frame, conf=0.45, show=False)
    pose_results = pose_model(frame, conf=0.45, show=False)
    
    # Dibujar detecciones de objetos
    frame_with_detections = detect_results[0].plot()
    
    # Añadir poses sobre el frame con detecciones
    for result in pose_results:
        if result.keypoints is not None:
            # Dibujar el esqueleto
            frame_with_detections = result.plot(img=frame_with_detections)
            
            # Añadir etiquetas de keypoints
            keypoints_data = result.keypoints.data[0]
            for idx, kp in enumerate(keypoints_data):
                x, y, conf = kp
                if conf > 0.5:  # Solo mostrar keypoints con confianza > 50%
                    cv2.putText(frame_with_detections, f"{keypoints[idx]}", 
                              (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 0), 1)
    
    # Mostrar información de detecciones
    for r in detect_results:
        for box in r.boxes:
            # Obtener clase y confianza
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            # Mostrar en la esquina superior
            cv2.putText(frame_with_detections, 
                       f"Detectado: {r.names[cls]} ({conf:.2f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)

    # Mostrar el frame final
    cv2.imshow("YOLO11 Detección Combinada", frame_with_detections)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 