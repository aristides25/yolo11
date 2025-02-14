from ultralytics import YOLO
import cv2

def list_available_cameras():
    """Enumera las cámaras disponibles y permite al usuario seleccionar una."""
    available_cameras = []
    for i in range(10):  # Buscar hasta 10 cámaras posibles
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

def select_camera():
    """Permite al usuario seleccionar una cámara de las disponibles."""
    cameras = list_available_cameras()
    
    if not cameras:
        print("No se detectaron cámaras. Saliendo...")
        exit()
    
    if len(cameras) == 1:
        print(f"Solo se detectó una cámara (índice {cameras[0]}). Usando esta...")
        return cameras[0]
    
    print("\nCámaras disponibles:")
    for i, cam_idx in enumerate(cameras):
        print(f"{i + 1}. Cámara {cam_idx}")
    
    while True:
        try:
            selection = int(input("\nSeleccione el número de la cámara a usar (1-{}): ".format(len(cameras))))
            if 1 <= selection <= len(cameras):
                return cameras[selection - 1]
            else:
                print("Selección inválida. Intente de nuevo.")
        except ValueError:
            print("Por favor, ingrese un número válido.")

# 1. Cargar modelos YOLOv11 (nano)
pose_model = YOLO("yolo11n-pose.pt")  # Personas + Poses (clase 0)
object_model = YOLO("yolo11n.pt")      # Objetos generales (clases 1-79)

# 2. Filtrar TODAS las clases no humanas (excluyendo clase 0)
non_human_classes = list(range(1, 80))  # IDs 1-79 (todas las clases COCO excepto 'person')

# 3. Seleccionar cámara
selected_camera = select_camera()

# 4. Procesar frames
cap = cv2.VideoCapture(selected_camera)
print(f"\nIniciando detección con cámara {selected_camera}...")
print("Presione 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer frame de la cámara. Saliendo...")
        break
    
    # Procesar ambos modelos (sin redundancia)
    pose_results = pose_model(frame, conf=0.5, max_det=5)       # Solo personas (clase 0)
    object_results = object_model(frame, classes=non_human_classes, conf=0.5)  # Objetos no humanos
    
    # Combinar resultados base
    combined_frame = pose_results[0].plot()                    # Dibuja personas y poses
    combined_frame = object_results[0].plot(img=combined_frame)  # Dibuja objetos
    
    # Mostrar resultado final
    cv2.imshow("Sistema Integral YOLOv11", combined_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()