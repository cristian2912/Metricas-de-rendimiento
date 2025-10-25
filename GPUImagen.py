import cv2, numpy as np, matplotlib.pyplot as plt, time
from google.colab import files
try:
    import cupy as cp
except Exception:
    import sys, subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cupy-cuda12x"])
    import cupy as cp

# --- Entrada ---
uploaded = files.upload()
img_name = list(uploaded.keys())[0]
t0 = time.perf_counter()
img_bgr = cv2.imread(img_name)
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

# --- Kernel CUDA: RGB->Gray ---
kernel_code = r'''
extern "C" __global__
void rgb_to_gray(const unsigned char* rgb, unsigned char* gray, int w, int h) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < w && y < h) {
        int i = y * w + x;
        int k = i * 3;
        gray[i] = (unsigned char)(0.299f*rgb[k] + 0.587f*rgb[k+1] + 0.114f*rgb[k+2]);
    }
}
'''
rgb_to_gray_kernel = cp.RawKernel(kernel_code, "rgb_to_gray")

# --- Copia H->D ---
t = time.perf_counter()
img_gpu = cp.asarray(img, dtype=cp.uint8).ravel()
gray_gpu = cp.zeros(w*h, dtype=cp.uint8)
cp.cuda.Stream.null.synchronize()
t_h2d = time.perf_counter()-t

# --- Lanzamiento ---
threads = (16, 16)
blocks = ((w + threads[0] - 1)//threads[0], (h + threads[1] - 1)//threads[1])

start = cp.cuda.Event(); end = cp.cuda.Event()
start.record()
rgb_to_gray_kernel(blocks, threads, (img_gpu, gray_gpu, w, h))
end.record(); end.synchronize()
kernel_ms = cp.cuda.get_elapsed_time(start, end)

# --- Copia D->H ---
t = time.perf_counter()
gray = cp.asnumpy(gray_gpu).reshape(h, w)
t_d2h = time.perf_counter()-t

# --- Resto del pipeline en CPU 
t = time.perf_counter(); smooth = cv2.GaussianBlur(gray, (3, 3), 0); t_blur = time.perf_counter()-t
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

# --- Salida visual y archivos ---
cv2.imwrite("GPU_lineart.png", lineart)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.imshow(img); plt.title("Original"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(lineart, cmap='gray', vmin=0, vmax=255); plt.title("GPU Delineado"); plt.axis("off")
plt.show()

# --- Reporte ---
print("== GPU ==")
print(f"Imagen: {w}x{h}")
print(f"H2D: {t_h2d:.4f}s  Kernel: {kernel_ms/1000:.4f}s  D2H: {t_d2h:.4f}s")
print(f"Blur: {t_blur:.4f}s  Canny: {t_canny:.4f}s  Thresh+OR: {t_thresh:.4f}s  Morph: {t_morph:.4f}s  CC+Invert: {t_cc:.4f}s")
print(f"Total: {total:.4f}s")
print(f"Pixels borde (negro): {edge_pixels}  blancos: {white_pixels}")
print(f"CSV: algo=GPU,shape={w}x{h},total_s={total:.6f},kernel_ms={kernel_ms:.3f},edges={edge_pixels}")
