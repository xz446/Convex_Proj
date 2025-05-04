"""
code to add specified noise to image
"""
import os
import glob
import cv2
import numpy as np
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio  

# -------------- config ---------------
INPUT_DIR = './original'
OUTPUT_DIR = './noised'
MODE = 'gaussian'
INTENSITY = 0.05
RANDOM_PERCENT = 15
BASELINE = 100
SEED = 42

def add_noise(image: np.ndarray, mode: str, intensity: float) -> np.ndarray:
    if mode == 'gaussian':
        return random_noise(image, mode='gaussian', mean=0, var=intensity)
    elif mode in ('s&p', 'salt', 'pepper'):
        return random_noise(image, mode=mode, amount=intensity)
    elif mode == 'speckle':
        return random_noise(image, mode='speckle', mean=0, var=intensity)
    elif mode == 'poisson':
        return random_noise(image, mode='poisson')
    else:
        raise ValueError(f"ERROR: noise type: {mode}")


def process_images(input_dir: str, output_dir: str, mode: str, intensity: float, random_pct: float, baseline: float, seed=None) -> None:
    if seed is not None:
        np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    file_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    files = []
    for pattern in file_patterns:
        files.extend(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        print(f"no images found in {input_dir}")
        return

    for file_path in files:
        try:
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                raise IOError(f"reading failed: {file_path}")
            if img.ndim == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()

            norm_img = gray.astype(np.float32) / 255.0

            actual_pct = (random_pct / 100.0) * baseline
            actual_pct = np.clip(actual_pct, 0.0, 100.0)
            num_pixels = gray.size
            num_noisy = int(num_pixels * actual_pct / 100.0)

            mask = np.zeros(num_pixels, dtype=bool)
            if num_noisy > 0:
                noisy_indices = np.random.choice(num_pixels, size=num_noisy, replace=False)
                mask[noisy_indices] = True
            mask = mask.reshape(gray.shape)

            noisy_full = add_noise(norm_img, mode, intensity)

            psnr_value = peak_signal_noise_ratio(norm_img, noisy_full, data_range=1.0)
            
            result = np.where(mask, noisy_full, norm_img)

            output = np.clip(result * 255.0, 0, 255).astype(np.uint8)
            filename = os.path.basename(file_path)
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, output)

            print(f"Processed: {filename} | Noisy pixels: {num_noisy}/{num_pixels} ({actual_pct:.1f}%) | PSNR (full noise): {psnr_value:.2f} dB")

        except Exception as e:
            print(f"ERROR: processing {file_path}: {e}")


def main():
    process_images(INPUT_DIR, OUTPUT_DIR, MODE, INTENSITY, RANDOM_PERCENT, BASELINE, SEED)

if __name__ == '__main__':
    main()
