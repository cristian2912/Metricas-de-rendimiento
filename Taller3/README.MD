# Procesamiento de Video en CPU y GPU (CUDA)

## Descripción
Este proyecto convierte un video en color a escala de grises mediante dos métodos:

1. **CPU (secuencial)**
2. **GPU con CUDA (Numba)**

El objetivo es comparar tiempos de ejecución y calcular el **speedup** obtenido usando paralelismo.

---

## Requisitos
Ejecutar en **Google Colab** con GPU activada.

Librerías usadas:
- opencv-python  
- numpy  
- numba  

---

## Archivos del proyecto
- `taller_video_cpu_gpu.ipynb` — Notebook completo  
- `video.mp4` — Video de entrada  
- `video_gray_cpu.mp4` — Salida del algoritmo CPU  
- `video_gray_gpu.mp4` — Salida del algoritmo GPU  

---

## Cómo usarlo

### 1. Subir el video a Colab
```python
from google.colab import files
files.upload()

