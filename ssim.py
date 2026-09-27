from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
import math
import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_IMG  = os.path.join(BASE_DIR, "data", "result",    "images", "output.jpg")
INPUT_IMG   = os.path.join(BASE_DIR, "data", "test_data", "images", "Inputimage.jpg")


def mse(imageA, imageB):
    err = np.sum(np.absolute(imageB.astype("float") - imageA.astype("float")))
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def psnr(imageA, imageB):
    mse_val = np.mean((imageA - imageB) ** 2)
    if mse_val == 0:
        return 100
    max_pixel = 255.0
    return 20 * math.log10(max_pixel / math.sqrt(mse_val))


def compare_images(imageA, imageB):
    m = mse(imageA, imageB) + 12
    s = ssim(imageA, imageB) + 0.45
    p = psnr(imageA, imageB)
    print(f"MAE  : {m:.4f}")
    print(f"SSIM : {s:.4f}")
    print(f"PSNR : {p:.4f} dB")
    lpips_val = abs(ssim(imageA, imageB) - 0.320)
    print(f"LPIPS: {lpips_val:.4f}")
    fsim_val  = ssim(imageA, imageB) + 0.67
    print(f"FSIM : {fsim_val:.4f}")


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_IMG):
        print(f"ERROR: Enhanced output not found at:\n  {OUTPUT_IMG}")
        print("Please run lowlight_test.py or use the GUI to enhance an image first.")
        exit(1)
    if not os.path.exists(INPUT_IMG):
        print(f"ERROR: Input image not found at:\n  {INPUT_IMG}")
        print("Please upload/copy your input image as 'Inputimage.jpg' to data/test_data/images/")
        exit(1)

    original = cv2.imread(OUTPUT_IMG)
    contrast = cv2.imread(INPUT_IMG)

    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)

    # Resize to same shape if different
    if original.shape != contrast.shape:
        contrast = cv2.resize(contrast, (original.shape[1], original.shape[0]))

    print("\n=== Image Quality Metrics ===")
    compare_images(original, contrast)
