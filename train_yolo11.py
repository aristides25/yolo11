from ultralytics import YOLO

# Cargar el modelo preentrenado YOLO
print("Iniciando entrenamiento del modelo...")
model = YOLO("yolo11n.pt")

# Entrenar el modelo
print("Entrenando el modelo (esto tomará tiempo)...")
results = model.train(
    data="coco8.yaml",
    epochs=100,
    imgsz=640,
    batch=8,  # Batch size más pequeño para CPU
    workers=4  # Menos workers para no sobrecargar la CPU
)
print("Entrenamiento completado. Modelo guardado en: 'runs/detect/train'") 