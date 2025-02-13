from ultralytics import YOLO
import cv2
from deepface import DeepFace
import numpy as np

# 1. Cargar modelos YOLOv11 (nano)
pose_model = YOLO("yolo11n-pose.pt")  # Personas + Poses (clase 0)
object_model = YOLO("yolo11n.pt")      # Objetos generales (clases 1-79)

# 2. Filtrar TODAS las clases no humanas (excluyendo clase 0)
non_human_classes = list(range(1, 80))  # IDs 1-79 (todas las clases COCO excepto 'person')

# 3. Función para analizar emociones
def analyze_emotion(face_img):
    try:
        result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        return emotion
    except Exception as e:
        return None

# 4. Procesar frames
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Procesar ambos modelos (sin redundancia)
    pose_results = pose_model(frame, conf=0.5, max_det=5)       # Solo personas (clase 0)
    object_results = object_model(frame, classes=non_human_classes, conf=0.5)  # Objetos no humanos
    
    # Combinar resultados base
    combined_frame = pose_results[0].plot()                    # Dibuja personas y poses
    combined_frame = object_results[0].plot(img=combined_frame)  # Dibuja objetos
    
    # Analizar emociones para cada persona detectada
    if pose_results[0].keypoints is not None:
        for person in pose_results[0].boxes:
            # Obtener coordenadas del cuadro delimitador
            x1, y1, x2, y2 = map(int, person.xyxy[0])
            
            # Extraer región de la cara (ajustada arriba de los hombros)
            face_y2 = int(y1 + (y2 - y1) * 0.3)  # 30% del altura total
            face_img = frame[y1:face_y2, x1:x2]
            
            if face_img.size > 0:  # Verificar que la imagen no esté vacía
                emotion = analyze_emotion(face_img)
                if emotion:
                    # Dibujar emoción sobre la persona
                    cv2.putText(combined_frame, f"Emocion: {emotion}", 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, (255, 255, 255), 2)
    
    # Mostrar resultado final
    cv2.imshow("Sistema Integral YOLOv11 + Emociones", combined_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()