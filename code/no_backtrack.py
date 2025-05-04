"""
denoising methods: Pixel L1, Wavelet L1, TV, and Wavelet+TV
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import argparse
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pywt
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# -------------------- config --------------------
NOISY_PATH = 'noised/img.jpg'
CLEAN_PATH = 'original/img.jpg'

# Regularization and algorithm parameters
LAMBDA = 0.0625
LAMBDA_STAGES = [LAMBDA]
# LAMBDA_STAGES = [LAMBDA * (0.5**i) for i in range(8)]
MAX_ITERS = {name: 1000 for name in [
    'ista_l1_pixel','fista_l1_pixel','ista_l1_wavelet','fista_l1_wavelet',
    'ista_tv','fista_tv','ista_wavelet_tv','fista_wavelet_tv']}
EARLY_STOP_TOL = 1e-4
L = 20
TV_ITERS = 100
TV_TAU = 0.249
WAVELET = 'db1'
LEVEL = 3

# -------------------- helpers --------------------
def load_image(path: str) -> np.ndarray:
    # normalize
    img = Image.open(path).convert('L')
    return np.asarray(img, dtype=np.float32) / 255.0


def save_image(arr: np.ndarray, name: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    base, _ = os.path.splitext(name)
    out_name = f"{base}.jpg"
    out = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(os.path.join(out_dir, out_name), format='JPEG')


def soft_threshold(x: np.ndarray, t: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)

# # -------------------- wavelet-domain L1 prox --------------------
def prox_wavelet(u: np.ndarray, lam: float) -> np.ndarray:
    coeffs = pywt.wavedec2(u, wavelet=WAVELET, level=LEVEL)
    cA, cDs = coeffs[0], coeffs[1:]
    new_cDs = []
    for (cH, cV, cD) in cDs:
        new_cDs.append((soft_threshold(cH, lam),
                        soft_threshold(cV, lam),
                        soft_threshold(cD, lam)))
    return pywt.waverec2([cA] + new_cDs, wavelet=WAVELET)

# -------------------- TV proximal operator --------------------
def prox_tv(u: np.ndarray, lam: float, n_iters: int = TV_ITERS, tau: float = TV_TAU) -> np.ndarray:
    m, n = u.shape
    p = np.zeros((2, m, n), dtype=u.dtype)
    def div(p):
        px, py = p
        d = np.zeros_like(u)
        d[:-1, :] += px[:-1, :]
        d[1:, :]  -= px[:-1, :]
        d[:, :-1] += py[:, :-1]
        d[:, 1:]  -= py[:, :-1]
        return d
    def grad(d):
        gx = np.diff(d, axis=0, append=0)
        gy = np.diff(d, axis=1, append=0)
        return np.stack((gx, gy), axis=0)
    for _ in range(n_iters):
        d = div(p) - u / lam
        g = grad(d)
        norm = np.sqrt(g[0]**2 + g[1]**2 + 1e-8)
        p = (p + tau * g) / (1 + tau * norm)
    return u - lam * div(p)

# -------------------- wavelet + TV prox --------------------
def prox_wavelet_tv(u: np.ndarray, lam: float) -> np.ndarray:
    return prox_tv(prox_wavelet(u, lam), lam)

# -------------------- ISTA and FISTA --------------------
def ista_iter(b: np.ndarray, lam: float, grad_f, prox, L: float, max_it: int, tol: float) -> (np.ndarray, int):
    """
    ISTA
    b: noisy image
    lam: regularizatin
    grad_f: gradient of the fidelity term
    prox: proximal operator
    L: Lipschitz
    max_it: maximum outer iterations
    tol: relative change stopping threshold
    """
    x = b.copy()
    t = 1.0 / L

    for k in range(1, max_it + 1):
        x_old = x.copy()
        # gradient + proximal
        x = np.clip(prox(x - t * grad_f(x), lam * t), 0, 1)

        # compute actual change
        diff = np.linalg.norm(x - x_old)
        # print(f"[ISTA] Iter {k:3d}, diff = {diff:.3e}")

        # stop when relative change below tol
        if diff < tol * np.linalg.norm(x_old):
            print(f"[ISTA] Converged at iter {k}, diff = {diff:.3e}")
            return x, k

    return x, max_it



def fista_iter(b: np.ndarray, lam: float, grad_f, prox, L: float, max_it: int, tol: float) -> (np.ndarray, int):
    """
    FISTA
    """
    x = y = b.copy()
    t = 1.0 / L
    tp = 1.0
    y_prev = b.copy()

    for k in range(1, max_it + 1):
        x_old = x.copy()
        # gradient + proximal
        g = grad_f(y)
        z = np.clip(prox(y - t * g, lam * t), 0, 1)

        # Nesterov momentum update
        tn = (1 + np.sqrt(1 + 4 * tp**2)) / 2
        y = z + ((tp - 1) / tn) * (z - x_old)
        tp = tn

        x = z
        y_prev = y.copy()

        diff = np.linalg.norm(x - x_old)
        # print(f"[FISTA] Iter {k:3d}, diff = {diff:.3e}")

        if diff < tol * np.linalg.norm(x_old):
            print(f"[FISTA] Converged at iter {k}, diff = {diff:.3e}")
            return x, k

    return x, max_it



# -------------------- execution --------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='Subdir name')
    args = parser.parse_args()

    out_base = os.path.join('restored', args.name)
    os.makedirs(out_base, exist_ok=True)
    shutil.copy2(NOISY_PATH, os.path.join(out_base, "noisy.jpg"))
    shutil.copy2(CLEAN_PATH, os.path.join(out_base, "clean.jpg"))

    b = load_image(NOISY_PATH)
    orig = load_image(CLEAN_PATH)
    grad_f = lambda x: x - b

    methods = [
        ('ista_l1_pixel', ista_iter, soft_threshold),
        ('fista_l1_pixel', fista_iter, soft_threshold),
        ('ista_l1_wavelet', ista_iter, prox_wavelet),
        ('fista_l1_wavelet', fista_iter, prox_wavelet),
        ('ista_tv', ista_iter, prox_tv),
        ('fista_tv', fista_iter, prox_tv),
        ('ista_wavelet_tv', ista_iter, prox_wavelet_tv),
        ('fista_wavelet_tv', fista_iter, prox_wavelet_tv)
    ]

    recs, names, times, iters = [], [], {}, {}
    start_all = time.time()

    for name, func, prox_op in methods:
        print(f"Running {name}...")
        t0 = time.time()
        x = b.copy()
        last_iter = 0
        for lam in LAMBDA_STAGES:
            x, it_count = func(b, lam, grad_f, prox_op, L, MAX_ITERS[name], EARLY_STOP_TOL)
            last_iter = it_count
            psnr = compute_psnr(orig, x, data_range=1)
            ssim = compute_ssim(orig, x, data_range=1)
            print(f"  {name} λ={lam:.4f}: PSNR={psnr:.2f}, SSIM={ssim:.4f}, Iters={it_count}")
        recs.append(x)
        names.append(name)
        times[name] = time.time() - t0
        iters[name] = last_iter
        save_image(x, f"{name}.jpg", out_base)

    all_imgs = [orig, b] + recs
    all_titles = ['Original', 'Noised'] + names
    cols = min(len(all_imgs), 4)
    rows = int(np.ceil(len(all_imgs) / cols))
    plt.figure(figsize=(4 * cols, 3 * rows))
    for i, (im, ttl) in enumerate(zip(all_imgs, all_titles)):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(im, cmap='gray')
        ax.set_title(ttl)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_base, 'comparison.jpg'))

    total_time = time.time() - start_all
    print(f"Total time: {total_time:.2f}s")
    for nm in names:
        psnr = compute_psnr(orig, recs[names.index(nm)], data_range=1)
        ssim = compute_ssim(orig, recs[names.index(nm)], data_range=1)
        print(f"{nm}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}, Time={times[nm]:.2f}s, Iters={iters[nm]}")
    print("Done.")
