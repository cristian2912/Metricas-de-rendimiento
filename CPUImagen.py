import cv2, numpy as np, matplotlib.pyplot as plt, time
from google.colab import files

# Entrada
uploaded = files.upload()
img_name = list(uploaded.keys())[0]
t0 = time.perf_counter()
img_bgr = cv2.imread(img_name)
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

# Pipeline cronometrado
t = time.perf_counter(); gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY); t_gray = time.perf_counter()-t
t = time.perf_counter(); smooth = cv2.GaussianBlur(gray, (3, 3), 0);     t_blur = time.perf_counter()-t

med = np.median(smooth); lower = int(max(25, 0.45 * med)); upper = int(min(190, 1.15 * med))

t = time.perf_counter(); edges = cv2.Canny(smooth, lower, upper, apertureSize=5, L2gradient=True); t_canny = time.perf_counter()-t
t = time.perf_counter()
adaptive = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
combined = cv2.bitwise_or(edges, adaptive); t_thresh = time.perf_counter()-t

t = time.perf_counter()
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=1)
kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel_open,  iterations=1)
t_morph = time.perf_counter()-t

t = time.perf_counter()
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
clean = np.zeros_like(combined)
min_size = 20
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] > min_size:
        clean[labels == i] = 255
lineart = 255 - clean
t_cc = time.perf_counter()-t

total = time.perf_counter()-t0
edge_pixels = int(np.sum(lineart == 0))
white_pixels = int(np.sum(lineart == 255))

# Salida visual y archivos
cv2.imwrite("CPU_lineart.png", lineart)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.imshow(img); plt.title("Original"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(lineart, cmap='gray', vmin=0, vmax=255); plt.title("CPU Delineado"); plt.axis("off")
plt.show()

# Reporte
print("== CPU ==")
print(f"Imagen: {w}x{h}")
print(f"Gray: {t_gray:.4f}s  Blur: {t_blur:.4f}s  Canny: {t_canny:.4f}s  Thresh+OR: {t_thresh:.4f}s  Morph: {t_morph:.4f}s  CC+Invert: {t_cc:.4f}s")
print(f"Total: {total:.4f}s")
print(f"Pixels borde (negro): {edge_pixels}  blancos: {white_pixels}")
print(f"CSV: algo=CPU,shape={w}x{h},total_s={total:.6f},kernel_ms=NaN,edges={edge_pixels}")
