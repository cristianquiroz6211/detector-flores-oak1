# 📝 Changelog - Detector de Flores OAK-1

Todas las mejoras importantes del proyecto se documentan en este archivo.

El formato se basa en [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
y este proyecto sigue [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-19

### ✨ Agregado
- **Sistema completo de detección** de flores en tiempo real con cámara OAK-1
- **Modelo YOLO personalizado** (`best.blob`) para clasificación binaria
- **Controles interactivos** en tiempo real:
  - `q` - Salir del programa
  - `r` - Reiniciar estadísticas
  - `ESPACIO` - Pausar/reanudar detección
  - `+/-` - Ajustar umbral de confianza
- **Sistema de filtros avanzado**:
  - Filtro por confianza mínima (75% por defecto)
  - Filtro por tamaño de flor (2%-50% del frame)
  - Filtro por relación de aspecto (0.3-3.0)
  - Non-Maximum Suppression (NMS) con umbral 0.4
- **Visualización completa**:
  - Cajas de detección con colores por clase
  - Etiquetas con ID, estado y confianza
  - Panel de estadísticas en tiempo real
  - Contador de FPS y rendimiento
- **Análisis inteligente**:
  - Clasificación general del frame actual
  - Estadísticas acumuladas por sesión
  - Cálculo de confianza promedio
- **Documentación profesional**:
  - README.md completo con guía de instalación
  - Docstrings detallados en todo el código
  - Comentarios explicativos paso a paso

### 🛠️ Configuración del Proyecto
- **requirements.txt** con dependencias específicas y documentadas
- **setup.py** para instalación como paquete Python
- **.gitignore** completo para desarrollo profesional
- **LICENSE** MIT para uso libre
- **CHANGELOG.md** para seguimiento de versiones

### 🎯 Características Técnicas
- **Entrada**: Resolución nativa de cámara OAK-1 (1080p)
- **Procesamiento**: Redimensión a 640x640px para YOLO
- **Salida**: Hasta 20 detecciones simultáneas
- **Rendimiento**: 25-30 FPS en hardware recomendado
- **Clases**: 
  - 🌸 `flor_lista_corte` (verde) - Flores maduras para corte
  - 🌿 `flor_no_lista` (naranja) - Flores inmaduras

### 🔧 Arquitectura
- **Pipeline optimizado**:
  1. Captura → Preprocesamiento → Inferencia
  2. Post-procesamiento → Filtrado → Visualización
  3. Análisis → Estadísticas → Retroalimentación
- **Gestión de memoria** eficiente para procesamiento en tiempo real
- **Manejo de errores** robusto con recuperación automática

### 📋 Compatibilidad
- **Python**: 3.8+ (probado hasta 3.12)
- **Sistemas**: Windows 10+, Ubuntu 18.04+, macOS 10.15+
- **Hardware**: Cámara OAK-1, USB 3.0+, 4GB RAM mínimo

## [Próximas Versiones]

### 🚀 Planificado para v1.1.0
- [ ] **Grabación de video** con detecciones marcadas
- [ ] **Exportación de estadísticas** a CSV/JSON
- [ ] **Configuración desde archivo** JSON/YAML
- [ ] **Interfaz web** básica para monitoreo remoto
- [ ] **Métricas avanzadas** (precisión, recall, F1-score)

### 🎯 Ideas para v1.2.0
- [ ] **Entrenamiento online** con feedback del usuario
- [ ] **Detección de múltiples especies** de flores
- [ ] **Análisis de calidad** (tamaño, forma, color)
- [ ] **API RESTful** para integración con otros sistemas
- [ ] **Dashboard** completo con gráficos históricos

### 🔮 Visión a largo plazo
- [ ] **Modelo de segmentación** para análisis más preciso
- [ ] **Detección 3D** usando capacidades de profundidad OAK-1
- [ ] **Sistema multi-cámara** para cobertura completa
- [ ] **Integración IoT** con sensores ambientales
- [ ] **Mobile app** para monitoreo desde dispositivos móviles

---

## 📊 Métricas de Desarrollo

| Métrica | Valor |
|---------|-------|
| Líneas de código | ~500 |
| Funciones documentadas | 100% |
| Cobertura de tests | 0% (próxima versión) |
| Tiempo de desarrollo | ~8 horas |
| Dependencias principales | 4 |

## 🤝 Contribuciones

Si quieres contribuir al proyecto:

1. **Fork** el repositorio
2. **Crea** una rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. **Commit** tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. **Push** a la rama (`git push origin feature/nueva-funcionalidad`)
5. **Abre** un Pull Request

## 📞 Contacto

**Cristian Quiroz** - [@cristianquiroz6211](https://github.com/cristianquiroz6211)

Proyecto: [detector-flores-oak1](https://github.com/cristianquiroz6211/detector-flores-oak1)

---

*Última actualización: 19 de Diciembre, 2024*
