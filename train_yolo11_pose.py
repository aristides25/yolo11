from ultralytics import YOLO

# 1. Cargar el modelo YOLO-pose
print("Iniciando configuración del modelo de poses...")
model = YOLO("yolo11n-pose.pt")  # Cargar modelo preentrenado (recomendado)

# 2. Entrenar el modelo
print("Iniciando entrenamiento del modelo de poses...")
print("Este proceso puede tomar tiempo...")

results = model.train(
    data="coco8-pose.yaml",  # Dataset específico para poses
    epochs=100,
    imgsz=640,
    batch=8,    # Batch size pequeño para CPU
    workers=4,  # Workers reducidos para CPU
    patience=50 # Parar si no hay mejora en 50 épocas
)

print("Entrenamiento completado!")
print("El modelo entrenado se ha guardado en: 'runs/pose/train/weights/best.pt'") 