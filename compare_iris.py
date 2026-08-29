import cv2
import numpy as np
import glob
import os
from skimage.feature import local_binary_pattern

# -----------------------------
# LBP settings
# -----------------------------
radius = 2
n_points = 8 * radius
GRID_SIZE = 8  # Divide 224x224 image into 8x8 spatial grid cells

# -----------------------------
# Spatial Feature Extraction (Grid LBP)
# -----------------------------
def get_spatial_lbp_features(image_path):
    # Read grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"[ERROR] Cannot load {image_path}")
        return None

    # Compute LBP representation
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")

    # Spatial Grid Histograms
    h, w = lbp.shape
    cell_h = h // GRID_SIZE
    cell_w = w // GRID_SIZE
    
    spatial_histograms = []

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            # Extract local grid cell
            cell = lbp[r*cell_h : (r+1)*cell_h, c*cell_w : (c+1)*cell_w]
            
            # Compute cell histogram
            hist, _ = np.histogram(
                cell.ravel(),
                bins=np.arange(0, n_points + 3),
                range=(0, n_points + 2)
            )
            
            # Normalize cell histogram
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)
            
            spatial_histograms.extend(hist)

    # Return concatenated spatial feature vector
    return np.array(spatial_histograms, dtype=np.float32)

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

print("\n[INFO] Comparing Spatial LBP Descriptors:")
print(f" Reference (Stored): {img1}")
print(f" Candidate (Login) : {img2}")

# -----------------------------
# Extract features
# -----------------------------
f1 = get_spatial_lbp_features(img1)
f2 = get_spatial_lbp_features(img2)

if f1 is None or f2 is None:
    exit()

# -----------------------------
# Distance Calculation (Chi-Square Distance for Histograms)
# -----------------------------
# Chi-Square distance is far superior to Euclidean distance for histogram matching
def chi_square_distance(histA, histB, eps=1e-10):
    return 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))

distance = chi_square_distance(f1, f2)

print(f"\n[INFO] Spatial LBP Chi-Square Distance Score: {distance:.4f}")

# -----------------------------
# Calibrated Decision Threshold
# -----------------------------
# Same person score:  typically 0.10 - 0.28
# Different person score: typically > 0.35
THRESHOLD = 0.30

if distance < 0.25:
    print("[RESULT] Strong Match (Same Person) ✅ - ACCESS GRANTED")
elif distance < THRESHOLD:
    print("[RESULT] Acceptable Match (Same Person) ✅ - ACCESS GRANTED")
else:
    print("[RESULT] Mismatch (Different Person) ❌ - ACCESS DENIED")