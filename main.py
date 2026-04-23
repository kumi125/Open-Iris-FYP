import os
import glob

from preprocess_iris import preprocess_latest
from segment_iris import segment_iris

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# -----------------------------
# Paths
# -----------------------------
USER_DB = "data/users"
os.makedirs(USER_DB, exist_ok=True)

# -----------------------------
# HYBRID FEATURE COMPARISON
# -----------------------------
def compare_images(img1_path, img2_path):

    def extract_features(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return None

        # -----------------------------
        # 1. LBP (Texture)
        # -----------------------------
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

        # -----------------------------
        # 2. Edge (Shape)
        # -----------------------------
        edges = cv2.Canny(img, 50, 150)

        edge_hist, _ = np.histogram(edges.ravel(), bins=2, range=(0, 256))
        edge_hist = edge_hist.astype("float")
        edge_hist /= (edge_hist.sum() + 1e-6)

        # -----------------------------
        # 3. Intensity (Brightness)
        # -----------------------------
        intensity_hist, _ = np.histogram(img.ravel(), bins=32, range=(0, 256))
        intensity_hist = intensity_hist.astype("float")
        intensity_hist /= (intensity_hist.sum() + 1e-6)

        # -----------------------------
        # Combine all
        # -----------------------------
        features = np.hstack([lbp_hist, edge_hist, intensity_hist])

        return features

    f1 = extract_features(img1_path)
    f2 = extract_features(img2_path)

    if f1 is None or f2 is None:
        return 999

    return np.linalg.norm(f1 - f2)

# -----------------------------
# REGISTER (MULTI-SAMPLE)
# -----------------------------
def register_user():

    username = input("Enter username: ").strip()
    user_path = os.path.join(USER_DB, username)

    if username == "":
        print("[ERROR] Empty username")
        return

    os.makedirs(user_path, exist_ok=True)

    print("[INFO] Registering user with 3 samples...")

    for i in range(5):
        print(f"\n[INFO] Capture sample {i+1}/5")

        os.system("python test_camera.py")
        preprocess_latest()

        saved_iris = segment_iris(user_path)

        if saved_iris is None:
            print("[WARNING] Sample failed, try again")
            continue

    print(f"\n[SUCCESS] User {username} registered")

# -----------------------------
# LOGIN (AVERAGE MATCHING)
# -----------------------------
def login_user():

    username = input("Enter username: ").strip()
    user_path = os.path.join(USER_DB, username)

    if not os.path.exists(user_path):
        print("[ERROR] User not found!")
        return

    print("[INFO] Capturing login image...")
    os.system("python test_camera.py")

    print("[INFO] Preprocessing...")
    preprocess_latest()

    print("[INFO] Segmenting iris...")

    temp_folder = "data/temp_login"
    os.makedirs(temp_folder, exist_ok=True)

    test_iris = segment_iris(temp_folder)

    print("[DEBUG] Segment output:", test_iris)

    if test_iris is None:
        print("[ERROR] Segmentation failed")
        return

    user_iris = glob.glob(os.path.join(user_path, "*.jpg"))

    if len(user_iris) == 0:
        print("[ERROR] No iris found for user")
        return

    print("[INFO] Comparing iris...")

    scores = []

    for ref in user_iris:
        score = compare_images(ref, test_iris)
        print(f"[DEBUG] {ref} → {score}")
        scores.append(score)

    avg_score = sum(scores) / len(scores)

    print(f"[INFO] Average Score: {avg_score}")
    
    min_score = min(scores)
    print(f"[INFO] Best Score: {min_score}")

    # -----------------------------
    # DECISION (UPDATED THRESHOLD)
    # -----------------------------
    if avg_score < 0.15 and len(close_matches) >= 2:
        print(f"[RESULT] LOGIN SUCCESS ✅ Welcome {username}")
    elif min_score < 0.20:
        print("[RESULT] UNCERTAIN MATCH ⚠️ Try again")
    else:
        print("[RESULT] LOGIN FAILED ❌ Not your iris")

# -----------------------------
# MENU
# -----------------------------
def menu():
    print("\n===== IRIS RECOGNITION SYSTEM =====")
    print("1. Register User")
    print("2. Login User")
    print("3. Exit")

# -----------------------------
# MAIN LOOP
# -----------------------------
if __name__ == "__main__":
    while True:
        menu()
        choice = input("Enter choice: ")

        if choice == "1":
            register_user()

        elif choice == "2":
            login_user()

        elif choice == "3":
            print("[INFO] Exiting...")
            break

        else:
            print("[ERROR] Invalid option")