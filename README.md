# YOLO11 - Sistema de Detección y Pose Estimation

Sistema de detección de objetos y estimación de poses en tiempo real usando YOLO11 de Ultralytics.

## Características

- Detección de objetos y personas
- Estimación de poses (17 keypoints)
- Detección combinada (objetos + poses)
- Soporte para CPU
- Visualización en tiempo real

## Archivos

- `yolo11.py`: Detección básica de objetos
- `yolo11_pose.py`: Estimación de poses
- `yolo11_combined.py`: Sistema combinado (detección + poses)
- `train_yolo11_pose.py`: Script de entrenamiento para poses

## Requisitos

```bash
pip install ultralytics
pip install opencv-python
```

## Uso

1. Detección básica:
```bash
python yolo11.py
```

2. Estimación de poses:
```bash
python yolo11_pose.py
```

3. Sistema combinado:
```bash
python yolo11_combined.py
```

4. Entrenamiento (opcional):
```bash
python train_yolo11_pose.py
```

## Keypoints Detectados

- 0: Nariz
- 1-2: Ojos (izquierdo/derecho)
- 3-4: Orejas (izquierda/derecha)
- 5-6: Hombros (izquierdo/derecho)
- 7-8: Codos (izquierdo/derecho)
- 9-10: Muñecas (izquierda/derecha)
- 11-12: Caderas (izquierda/derecha)
- 13-14: Rodillas (izquierda/derecha)
- 15-16: Tobillos (izquierdo/derecho)

## Controles

- Presiona 'q' para salir de cualquier visualización

## Autor

- Aristides 