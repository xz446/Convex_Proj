"""
denoising methods: Pixel L1, Wavelet L1, TV, and Wavelet+TV
"""
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
TV_ITERS = 100
TV_TAU = 0.249
WAVELET = 'db1'
LEVEL = 3

# -------------------- helpers --------------------
def load_image(path):
    img = Image.open(path).convert('L')
    return np.asarray(img, dtype=np.float32) / 255.0

def save_image(arr, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    base, _ = os.path.splitext(name)
    out_name = f"{base}.jpg"
    out = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(os.path.join(out_dir, out_name), format='JPEG')

def soft_threshold(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)

def prox_wavelet(u, lam):
    coeffs = pywt.wavedec2(u, wavelet=WAVELET, level=LEVEL)
    cA, cDs = coeffs[0], coeffs[1:]
    new_cDs = []
    for (cH, cV, cD) in cDs:
        new_cDs.append((soft_threshold(cH, lam),
                        soft_threshold(cV, lam),
                        soft_threshold(cD, lam)))
    return pywt.waverec2([cA] + new_cDs, wavelet=WAVELET)

def prox_tv(u, lam, n_iters=TV_ITERS, tau=TV_TAU):
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

def prox_wavelet_tv(u, lam):
    return prox_tv(prox_wavelet(u, lam), lam)

# -------------------- backtracking ISTA and FISTA --------------------
def ista_backtracking(b, lam, grad_f, prox, max_it, tol, L0=1.0, eta=2.0):
    x = b.copy()
    for k in range(1, max_it + 1):
        x_old = x.copy()
        L = L0
        while True:
            t = 1.0 / L
            x_new = np.clip(prox(x_old - t * grad_f(x_old), lam * t), 0, 1)
            diff = x_new - x_old
            lhs = 0.5 * np.linalg.norm(x_new - b)**2
            rhs = 0.5 * np.linalg.norm(x_old - b)**2 + np.sum(diff * grad_f(x_old)) + (L / 2) * np.linalg.norm(diff)**2
            if lhs <= rhs:
                break
            L *= eta
        x = x_new
        if np.linalg.norm(x - x_old) < tol * np.linalg.norm(x_old):
            print(f"[ISTA-BT] Converged at iter {k}, diff = {np.linalg.norm(x - x_old):.3e}")
            return x, k
    return x, max_it

def fista_backtracking(b, lam, grad_f, prox, max_it, tol, L0=1.0, eta=2.0):
    x = y = b.copy()
    t_prev = 1.0
    for k in range(1, max_it + 1):
        x_old = x.copy()
        L = L0
        while True:
            t = 1.0 / L
            x_new = np.clip(prox(y - t * grad_f(y), lam * t), 0, 1)
            diff = x_new - y
            lhs = 0.5 * np.linalg.norm(x_new - b)**2
            rhs = 0.5 * np.linalg.norm(y - b)**2 + np.sum(diff * grad_f(y)) + (L / 2) * np.linalg.norm(diff)**2
            if lhs <= rhs:
                break
            L *= eta
        t_curr = (1 + np.sqrt(1 + 4 * t_prev**2)) / 2
        y = x_new + ((t_prev - 1) / t_curr) * (x_new - x_old)
        t_prev = t_curr
        x = x_new
        if np.linalg.norm(x - x_old) < tol * np.linalg.norm(x_old):
            print(f"[FISTA-BT] Converged at iter {k}, diff = {np.linalg.norm(x - x_old):.3e}")
            return x, k
    return x, max_it

# -------------------- execution --------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    args = parser.parse_args()

    out_base = os.path.join('restored', args.name)
    os.makedirs(out_base, exist_ok=True)
    shutil.copy2(NOISY_PATH, os.path.join(out_base, "noisy.jpg"))
    shutil.copy2(CLEAN_PATH, os.path.join(out_base, "clean.jpg"))

    b = load_image(NOISY_PATH)
    orig = load_image(CLEAN_PATH)
    grad_f = lambda x: x - b

    methods = [
        ('ista_l1_pixel', ista_backtracking, soft_threshold),
        ('fista_l1_pixel', fista_backtracking, soft_threshold),
        ('ista_l1_wavelet', ista_backtracking, prox_wavelet),
        ('fista_l1_wavelet', fista_backtracking, prox_wavelet),
        ('ista_tv', ista_backtracking, prox_tv),
        ('fista_tv', fista_backtracking, prox_tv),
        ('ista_wavelet_tv', ista_backtracking, prox_wavelet_tv),
        ('fista_wavelet_tv', fista_backtracking, prox_wavelet_tv)
    ]

    recs, names, times, iters = [], [], {}, {}
    start_all = time.time()

    for name, func, prox_op in methods:
        print(f"Running {name}...")
        t0 = time.time()
        x = b.copy()
        last_iter = 0
        for lam in LAMBDA_STAGES:
            x, it_count = func(b, lam, grad_f, prox_op, MAX_ITERS[name], EARLY_STOP_TOL)
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
    print(f"\n=== Summary ===\nTotal time: {total_time:.2f}s")
    for nm in names:
        psnr = compute_psnr(orig, recs[names.index(nm)], data_range=1)
        ssim = compute_ssim(orig, recs[names.index(nm)], data_range=1)
        print(f"{nm}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}, Time={times[nm]:.2f}s, Iters={iters[nm]}")
    print("Done.")
