# Detector de Flores OAK-1

Detector de flores para cámara OAK-1 que clasifica flores como "listas para corte" o "no listas".

## Archivos principales

- `detector_flores.py` - Script principal del detector
- `best.blob` - Modelo YOLO entrenado
- `best.pt` / `best.onnx` - Otros formatos del modelo (opcionales)

## Uso

```bash
python detector_flores.py
```

## Controles en tiempo real

- **'q'** - Salir
- **'r'** - Reiniciar estadísticas
- **'ESPACIO'** - Pausar/reanudar
- **'+'** - Aumentar umbral de confianza
- **'-'** - Disminuir umbral de confianza

## Configuración

El script está optimizado para:
- Entrada: 640x640 píxeles
- Clases: flores listas para corte, flores no listas
- Umbral inicial: 75% (ajustable)
- Máximo 20 detecciones por frame

## Requisitos

- DepthAI
- OpenCV
- NumPy
- Cámara OAK-1 conectada
