import cv2
import numpy as np
import glob
import os
from skimage.feature import local_binary_pattern

# -----------------------------
# LBP settings
# -----------------------------
radius = 3
n_points = 8 * radius

# -----------------------------
# Feature extraction (LBP)
# -----------------------------
def get_lbp_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"[ERROR] Cannot load {image_path}")
        return None

    # Apply LBP
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")

    # Convert to histogram
    hist, _ = np.histogram(lbp.ravel(),
                          bins=np.arange(0, n_points + 3),
                          range=(0, n_points + 2))

    # Normalize
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    return hist

# -----------------------------
# Load last 2 segmented images
# -----------------------------
files = glob.glob("data/segmented/*.jpg")

if len(files) < 2:
    print("[ERROR] Need at least 2 segmented images!")
    exit()

files = sorted(files, key=os.path.getmtime)

img1 = files[-2]
img2 = files[-1]

print("\n[INFO] Comparing:")
print(img1)
print(img2)

# -----------------------------
# Extract features
# -----------------------------
f1 = get_lbp_features(img1)
f2 = get_lbp_features(img2)

if f1 is None or f2 is None:
    exit()

# -----------------------------
# Compare using distance
# -----------------------------
distance = np.linalg.norm(f1 - f2)

print("\n[INFO] LBP Difference score:", distance)

# -----------------------------
# Decision
# -----------------------------
threshold = 0.12  # LBP uses smaller values

if distance < 0.1:
    print("[RESULT] Strong Match (Same Person) ✅")
elif distance < 0.15:
    print("[RESULT] Weak Match (Uncertain) ⚠️")
else:
    print("[RESULT] Different Person ❌")