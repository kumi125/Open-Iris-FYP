import os
import glob
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# -----------------------------
# Feature extraction (same as main)
# -----------------------------
def extract_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    radius = 3
    n_points = 8 * radius

    lbp = local_binary_pattern(img, n_points, radius, method="uniform")

    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    edges = cv2.Canny(img, 50, 150)
    edge_hist, _ = np.histogram(edges.ravel(), bins=2, range=(0, 256))
    edge_hist = edge_hist.astype("float")
    edge_hist /= (edge_hist.sum() + 1e-6)

    intensity_hist, _ = np.histogram(img.ravel(), bins=32, range=(0, 256))
    intensity_hist = intensity_hist.astype("float")
    intensity_hist /= (intensity_hist.sum() + 1e-6)

    return np.hstack([lbp_hist, edge_hist, intensity_hist])


def compare(img1, img2):
    f1 = extract_features(img1)
    f2 = extract_features(img2)

    if f1 is None or f2 is None:
        return 999

    return np.linalg.norm(f1 - f2)

# -----------------------------
# Evaluation
# -----------------------------
USER_DB = "data/users"

threshold = 0.18

genuine_total = 0
genuine_fail = 0

impostor_total = 0
impostor_accept = 0

users = os.listdir(USER_DB)

for user in users:
    user_path = os.path.join(USER_DB, user)
    images = glob.glob(os.path.join(user_path, "*.jpg"))

    # -----------------------------
    # Genuine comparisons (same user)
    # -----------------------------
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            score = compare(images[i], images[j])
            genuine_total += 1

            if score > threshold:
                genuine_fail += 1

    # -----------------------------
    # Impostor comparisons (different users)
    # -----------------------------
    for other_user in users:
        if other_user == user:
            continue

        other_path = os.path.join(USER_DB, other_user)
        other_images = glob.glob(os.path.join(other_path, "*.jpg"))

        for img1 in images:
            for img2 in other_images:
                score = compare(img1, img2)
                impostor_total += 1

                if score < threshold:
                    impostor_accept += 1

# -----------------------------
# Results
# -----------------------------
FAR = impostor_accept / impostor_total if impostor_total > 0 else 0
FRR = genuine_fail / genuine_total if genuine_total > 0 else 0

print("\n===== SYSTEM EVALUATION =====")
print(f"Total Genuine Tests: {genuine_total}")
print(f"Total Impostor Tests: {impostor_total}")

print(f"\nFAR (False Acceptance Rate): {FAR:.2f}")
print(f"FRR (False Rejection Rate): {FRR:.2f}")