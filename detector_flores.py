#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detector de flores OAK-1 optimizado
Modelo: best.blob (2 clases: flores listas para corte vs no listas)
Configuración específica para detección de flores con cajas pequeñas
"""

import cv2
import depthai as dai
import numpy as np
from pathlib import Path
import time

# Configuración para detección de flores
BLOB_PATH = "best.blob"
INPUT_SIZE = (640, 640)
CLASSES = ["flor_lista_corte", "flor_no_lista"]  # Nombres específicos para flores
COLORS = [(0, 255, 0), (0, 165, 255)]  # Verde brillante, Naranja
CONFIDENCE_THRESHOLD = 0.75  # Umbral alto para evitar detecciones aleatorias
NMS_THRESHOLD = 0.4

def create_pipeline():
    """Pipeline optimizado para detección de flores"""
    pipeline = dai.Pipeline()
    
    # Cámara con configuración optimizada
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(*INPUT_SIZE)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)
    
    # Red neuronal
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(BLOB_PATH)
    nn.setNumPoolFrames(4)
    nn.setNumInferenceThreads(2)
    
    # Conexiones
    cam.preview.link(nn.input)
    
    # Salidas
    cam_out = pipeline.create(dai.node.XLinkOut)
    cam_out.setStreamName("cam")
    cam.preview.link(cam_out.input)
    
    nn_out = pipeline.create(dai.node.XLinkOut)
    nn_out.setStreamName("nn")
    nn.out.link(nn_out.input)
    
    return pipeline

def parse_yolo_detections(output, confidence_threshold=0.7, nms_threshold=0.4):
    """
    Parsear salida YOLO optimizada para detección de flores
    """
    output = np.array(output)
    
    # YOLO formato: 50400 valores = 7200 detecciones × 7 valores
    # Para flores: [x, y, w, h, objectness, class_0_prob, class_1_prob]
    num_detections = len(output) // 7
    detections = output.reshape(num_detections, 7)
    
    valid_detections = []
    
    for detection in detections:
        x, y, w, h, objectness, class_0_prob, class_1_prob = detection
        
        # Normalizar probabilidades de clase (softmax)
        class_probs = np.array([class_0_prob, class_1_prob])
        exp_probs = np.exp(class_probs - np.max(class_probs))  # Para estabilidad numérica
        class_probs = exp_probs / np.sum(exp_probs)
        
        class_id = np.argmax(class_probs)
        class_confidence = class_probs[class_id]
        
        # Normalizar objectness (sigmoid)
        objectness = 1 / (1 + np.exp(-objectness))
        
        # Confianza final
        final_confidence = objectness * class_confidence
        
        # Filtros para flores:
        # 1. Confianza mínima más alta
        if final_confidence < confidence_threshold:
            continue
            
        # 2. Tamaño de caja apropiado para flores (no muy grande, no muy pequeña)
        if w < 0.02 or w > 0.5 or h < 0.02 or h > 0.5:  # Entre 2% y 50% del frame
            continue
            
        # 3. Relación de aspecto razonable para flores
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:  # No muy alargadas
            continue
        
        # 4. Coordenadas dentro de límites válidos
        if x < 0 or x > 1 or y < 0 or y > 1:
            continue
        
        # Convertir coordenadas de centro a esquinas
        x1 = max(0, min(1, x - w/2))
        y1 = max(0, min(1, y - h/2))
        x2 = max(0, min(1, x + w/2))
        y2 = max(0, min(1, y + h/2))
        
        # Verificar que la caja tenga área válida
        if (x2 - x1) > 0.01 and (y2 - y1) > 0.01:  # Área mínima del 1%
            valid_detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': final_confidence,
                'class_id': class_id,
                'class_name': CLASSES[class_id],
                'objectness': objectness,
                'class_prob': class_confidence
            })
    
    # Aplicar Non-Maximum Suppression más estricto
    if len(valid_detections) > 0:
        # Ordenar por confianza
        valid_detections = sorted(valid_detections, key=lambda x: x['confidence'], reverse=True)
        
        # NMS manual para mejor control
        final_detections = []
        for i, det_a in enumerate(valid_detections):
            keep = True
            for j, det_b in enumerate(final_detections):
                # Calcular IoU
                box_a = det_a['bbox']
                box_b = det_b['bbox']
                
                # Intersección
                x1 = max(box_a[0], box_b[0])
                y1 = max(box_a[1], box_b[1])
                x2 = min(box_a[2], box_b[2])
                y2 = min(box_a[3], box_b[3])
                
                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
                    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
                    union = area_a + area_b - intersection
                    
                    if union > 0:
                        iou = intersection / union
                        if iou > nms_threshold:
                            keep = False
                            break
            
            if keep:
                final_detections.append(det_a)
        
        # Limitar número máximo de detecciones para flores
        valid_detections = final_detections[:20]  # Máximo 20 flores por frame
    
    return valid_detections

def get_overall_classification(detections):
    """
    Obtener clasificación general basada en las detecciones de flores
    """
    if not detections:
        return 0, 0.5, [0.5, 0.5]  # Por defecto flores listas
    
    # Contar detecciones por clase, ponderadas por confianza
    class_scores = [0, 0]
    total_confidence = 0
    
    for det in detections:
        class_id = det['class_id']
        confidence = det['confidence']
        class_scores[class_id] += confidence
        total_confidence += confidence
    
    if total_confidence > 0:
        probs = [score / total_confidence for score in class_scores]
    else:
        probs = [0.5, 0.5]
    
    # Normalizar probabilidades
    total_prob = sum(probs)
    if total_prob > 0:
        probs = [p / total_prob for p in probs]
    
    pred_class = np.argmax(probs)
    confidence = probs[pred_class]
    
    return pred_class, confidence, probs

def draw_results(frame, pred_class, confidence, probs, fps, detections=None):
    """Dibujar resultados optimizado para detección de flores"""
    h, w = frame.shape[:2]
    
    # Dibujar detecciones individuales de flores
    if detections:
        for i, det in enumerate(detections):
            bbox = det['bbox']
            det_confidence = det['confidence']
            class_id = det['class_id']
            class_name = det['class_name']
            
            # Convertir coordenadas normalizadas a píxeles
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)
            
            # Color según la clase
            color = COLORS[class_id]
            
            # Caja más gruesa para flores con alta confianza
            thickness = 3 if det_confidence > 0.8 else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Punto central de la flor
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 3, color, -1)
            
            # Label más informativo
            status = "✓ CORTE" if class_id == 0 else "✗ NO CORTE"
            label = f"Flor {i+1}: {status} ({det_confidence:.0%})"
            
            # Tamaño del label
            font_scale = 0.5
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            
            # Posición del label (arriba de la caja)
            label_x = x1
            label_y = y1 - 10 if y1 > 30 else y2 + 25
            
            # Background del label
            cv2.rectangle(frame, (label_x - 2, label_y - label_size[1] - 5), 
                         (label_x + label_size[0] + 4, label_y + 3), color, -1)
            
            # Texto del label
            text_color = (255, 255, 255) if class_id == 1 else (0, 0, 0)
            cv2.putText(frame, label, (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
    
    # Información general
    class_name = CLASSES[pred_class]
    color = COLORS[pred_class]
    
    # Panel de información
    panel_height = 90
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w-10, panel_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)
    
    # Status general
    general_status = "LISTAS PARA CORTE" if pred_class == 0 else "NO LISTAS"
    cv2.putText(frame, f"Estado General: {general_status}", (20, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    if detections:
        # Contar flores por tipo
        listas = sum(1 for det in detections if det['class_id'] == 0)
        no_listas = sum(1 for det in detections if det['class_id'] == 1)
        
        cv2.putText(frame, f"Flores: {len(detections)} total | Listas: {listas} | No listas: {no_listas}", 
                   (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Confianza promedio
        avg_conf = np.mean([det['confidence'] for det in detections])
        cv2.putText(frame, f"Confianza promedio: {avg_conf:.0%}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "No se detectaron flores", (20, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, h-20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Indicador visual principal
    center = (w-60, 45)
    cv2.circle(frame, center, 25, color, -1)
    cv2.circle(frame, center, 25, (255, 255, 255), 2)
    
    # Símbolo en el indicador
    if pred_class == 0:  # Listas para corte
        cv2.putText(frame, "✓", (center[0]-8, center[1]+8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    else:  # No listas
        cv2.line(frame, (center[0]-10, center[1]-10), (center[0]+10, center[1]+10), (255, 255, 255), 3)
        cv2.line(frame, (center[0]-10, center[1]+10), (center[0]+10, center[1]-10), (255, 255, 255), 3)
    
    # Información de umbral actual
    cv2.putText(frame, f"Umbral: {CONFIDENCE_THRESHOLD:.0%}", (w-120, h-20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    return frame

def main():
    """Función principal"""
    global CONFIDENCE_THRESHOLD
    
    print("🌸 DETECTOR DE FLORES OAK-1")
    print("=" * 45)
    
    if not Path(BLOB_PATH).exists():
        print(f"❌ {BLOB_PATH} no encontrado")
        return
    
    print(f"✅ Modelo: {BLOB_PATH}")
    print(f"📏 Entrada: {INPUT_SIZE[0]}x{INPUT_SIZE[1]}")
    print(f"🌺 Clases: {', '.join(CLASSES)}")
    print(f"🎯 Umbral de confianza: {CONFIDENCE_THRESHOLD:.0%}")
    print(f"🔧 Umbral NMS: {NMS_THRESHOLD:.0%}")
    
    try:
        pipeline = create_pipeline()
        
        print("🔌 Conectando...")
        with dai.Device(pipeline) as device:
            print("✅ ¡Conectado!")
            
            q_cam = device.getOutputQueue("cam", maxSize=4, blocking=False)
            q_nn = device.getOutputQueue("nn", maxSize=4, blocking=False)
            
            # Variables para FPS y estadísticas
            fps = 0
            fps_counter = 0
            fps_time = time.time()
            
            # Estadísticas para flores
            stats = {"flor_lista_corte": 0, "flor_no_lista": 0}
            total_predictions = 0
            
            print("\n🌸 DETECTANDO FLORES EN VIVO")
            print("Controles:")
            print("  'q' - Salir")
            print("  'r' - Reiniciar estadísticas")
            print("  'ESPACIO' - Pausa")
            print("  '+' - Aumentar umbral de confianza")
            print("  '-' - Disminuir umbral de confianza")
            print("-" * 50)
            
            paused = False
            last_prediction = None
            detections = []
            
            while True:
                # Obtener datos
                cam_frame = q_cam.get()
                nn_data = q_nn.tryGet()
                
                if cam_frame is not None:
                    frame = cam_frame.getCvFrame()
                    
                    # Procesar inferencia
                    if nn_data is not None:
                        output = nn_data.getFirstLayerFp16()
                        
                        # Parsear detecciones YOLO con umbral más estricto
                        detections = parse_yolo_detections(output, 
                                                         confidence_threshold=CONFIDENCE_THRESHOLD,
                                                         nms_threshold=NMS_THRESHOLD)
                        
                        # Obtener clasificación general
                        pred_class, confidence, probs = get_overall_classification(detections)
                        
                        class_name = CLASSES[pred_class]
                        
                        # Actualizar estadísticas
                        stats[class_name] += 1
                        total_predictions += 1
                        
                        # Mostrar solo cambios significativos o detecciones importantes
                        if (last_prediction != pred_class or confidence > 0.7 or 
                            (detections and len(detections) > 0)):
                            det_info = f" ({len(detections)} flores)" if detections else ""
                            print(f"🌸 {class_name}: {confidence:.1%}{det_info} | Total: LC={stats['flor_lista_corte']}, NL={stats['flor_no_lista']}")
                            last_prediction = pred_class
                    
                    else:
                        # Sin nueva inferencia, usar última predicción
                        pred_class = 0  # Por defecto flores listas
                        confidence = 0.5
                        probs = [0.5, 0.5]
                        detections = []
                    
                    # Calcular FPS
                    fps_counter += 1
                    if fps_counter % 30 == 0:
                        fps = fps_counter / (time.time() - fps_time)
                        fps_counter = 0
                        fps_time = time.time()
                    
                    # Dibujar resultados
                    frame = draw_results(frame, pred_class, confidence, probs, fps, detections)
                    
                    # Mostrar solo si no está pausado
                    if not paused:
                        cv2.imshow("🌸 Detector de Flores OAK-1", frame)
                    
                    # Manejar teclas
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        stats = {"flor_lista_corte": 0, "flor_no_lista": 0}
                        total_predictions = 0
                        print("📊 Estadísticas reiniciadas")
                    elif key == ord(' '):
                        paused = not paused
                        print(f"⏸️ {'Pausado' if paused else 'Reanudado'}")
                    elif key == ord('+') or key == ord('='):
                        CONFIDENCE_THRESHOLD = min(0.95, CONFIDENCE_THRESHOLD + 0.05)
                        print(f"🔺 Umbral aumentado: {CONFIDENCE_THRESHOLD:.0%}")
                    elif key == ord('-') or key == ord('_'):
                        CONFIDENCE_THRESHOLD = max(0.10, CONFIDENCE_THRESHOLD - 0.05)
                        print(f"🔻 Umbral reducido: {CONFIDENCE_THRESHOLD:.0%}")
        
        # Mostrar estadísticas finales
        print("\n📊 ESTADÍSTICAS FINALES DE FLORES:")
        print(f"  Total de predicciones: {total_predictions}")
        if total_predictions > 0:
            for class_name, count in stats.items():
                percentage = (count / total_predictions) * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        print("\n✅ Sesión de detección terminada")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
