import cv2
import cupy as cp
from google.colab import files
import matplotlib.pyplot as plt
import numpy as np

# Cargar imagen
uploaded = files.upload()
img_name = list(uploaded.keys())[0]
img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)

# Conversión a gris en GPU
kernel_code = r'''
extern "C" __global__
void rgb_to_gray(const unsigned char* rgb, unsigned char* gray, int w, int h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (idx < w && idy < h) {
        int i = idy * w + idx;
        int k = i * 3;
        gray[i] = (unsigned char)(0.299f*rgb[k] + 0.587f*rgb[k+1] + 0.114f*rgb[k+2]);
    }
}
'''

rgb_to_gray_kernel = cp.RawKernel(kernel_code, "rgb_to_gray")
h, w, _ = img.shape

img_gpu = cp.asarray(img, dtype=cp.uint8).ravel()
gray_gpu = cp.zeros(w*h, dtype=cp.uint8)

threads_per_block = (16, 16)
blocks_per_grid = (
    (w + threads_per_block[0] - 1) // threads_per_block[0],
    (h + threads_per_block[1] - 1) // threads_per_block[1]
)

rgb_to_gray_kernel(blocks_per_grid, threads_per_block, (img_gpu, gray_gpu, w, h))
gray = cp.asnumpy(gray_gpu).reshape(h, w)

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
# 1) Solo cerrar gaps pequeños
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=1)

# 2) Limpieza MUY ligera (mantiene más detalles)
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open, iterations=1)

# 3) Filtrar SOLO manchas muy pequeñas (ruido obvio)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined, connectivity=8)

clean = np.zeros_like(combined)

# Umbral MUY bajo para mantener casi todo
min_size = 20  # Más bajo = mantiene más detalles
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
