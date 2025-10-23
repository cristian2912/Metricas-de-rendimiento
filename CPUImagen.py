import cv2
from google.colab import files
import matplotlib.pyplot as plt
import numpy as np

# Cargar imagen
uploaded = files.upload()
img_name = list(uploaded.keys())[0]
img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)

# Conversión a gris en CPU
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Delineado BALANCEADO: limpio pero con detalles
smooth = cv2.GaussianBlur(gray, (3, 3), 0)

med = np.median(smooth)
lower = int(max(25, 0.45 * med))
upper = int(min(190, 1.15 * med))

edges = cv2.Canny(smooth, lower, upper, apertureSize=5, L2gradient=True)

adaptive = cv2.adaptiveThreshold(
    smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY_INV, blockSize=11, C=3
)

combined = cv2.bitwise_or(edges, adaptive)

# Limpieza SUAVE (sin perder detalles)
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=1)

kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open, iterations=1)

# Filtrar SOLO manchas muy pequeñas
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined, connectivity=8)

clean = np.zeros_like(combined)

min_size = 20
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if area > min_size:
        clean[labels == i] = 255

lineart = 255 - clean

# Mostrar
plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.imshow(img); plt.title("Original"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(lineart, cmap='gray', vmin=0, vmax=255)
plt.title("Delineado Limpio"); plt.axis("off")
plt.show()
