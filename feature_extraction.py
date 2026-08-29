import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# -----------------------------
# Configuration
# -----------------------------
RADIUS = 2
N_POINTS = 8 * RADIUS
GRID_SIZE = 8
POLAR_HEIGHT = 64
POLAR_WIDTH = 256

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def is_eye_open(gray_img):
    """
    Quality Gate: Validates if the captured frame contains an open eye with a visible pupil/iris.
    Rejects closed eyes, eyelids, or blank crops.
    """
    if gray_img is None or gray_img.size == 0:
        return False

    h, w = gray_img.shape

    # 1. Edge & Variance Check (Closed eyelids are smoother than open irises)
    laplacian_var = cv2.Laplacian(gray_img, cv2.CV_64F).var()
    if laplacian_var < 80.0:  # Threshold for blur / smooth skin (eyelid)
        print(f"[QUALITY GATE] Frame rejected: Image too smooth / eyelid closed (Variance: {laplacian_var:.1f})")
        return False

    # 2. Pupil Detection Check (A closed eye has no dark central pupil)
    blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=50,
        param2=18,
        minRadius=int(min(h, w) * 0.08),
        maxRadius=int(min(h, w) * 0.45)
    )

    if circles is None:
        print("[QUALITY GATE] Frame rejected: No pupil/iris circle detected (Eye Closed/Blinking)")
        return False

    return True


def center_iris_crop(img):
    """Detects pupil center dynamically and aligns crop matrix."""
    if img is None or img.size == 0:
        return img

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=50,
        param2=18,
        minRadius=int(min(h, w) * 0.08),
        maxRadius=int(min(h, w) * 0.45)
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        cx, cy, r = int(circles[0][0][0]), int(circles[0][0][1]), int(circles[0][0][2])

        dx = (w // 2) - cx
        dy = (h // 2) - cy

        M = np.float32([[1, 0, dx], [0, 1, dy]])
        centered = cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return centered

    return gray


def preprocess_frame(img):
    """Validates eye openness, centers crop, and enhances local contrast."""
    if img is None or img.size == 0:
        return None

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Step 1: Open Eye Quality Gate
    if not is_eye_open(gray):
        return None

    # Step 2: Dynamic centering
    centered = center_iris_crop(gray)

    # Step 3: Resize standardization
    if centered.shape != (224, 224):
        centered = cv2.resize(centered, (224, 224))

    # Step 4: CLAHE Contrast Enhancement
    return clahe.apply(centered)


def extract_features(input_source):
    """Extracts features if quality gate passes."""
    try:
        if isinstance(input_source, str):
            if not os.path.exists(input_source):
                return None
            image = cv2.imread(input_source, cv2.IMREAD_GRAYSCALE)
        elif isinstance(input_source, np.ndarray):
            image = input_source.copy()
        else:
            return None

        image = preprocess_frame(image)
        if image is None:  # Failed quality gate (closed eye / invalid)
            return None

        h, w = image.shape
        r_outer = int(min(h, w) * 0.48)

        try:
            polar = cv2.warpPolar(image, (POLAR_WIDTH, POLAR_HEIGHT), (w // 2, h // 2), r_outer, cv2.WARP_POLAR_LINEAR)
        except Exception:
            polar = cv2.resize(image, (POLAR_WIDTH, POLAR_HEIGHT))

        lbp = local_binary_pattern(polar, N_POINTS, RADIUS, method="uniform")

        # Global LBP Histogram
        global_hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, N_POINTS + 3),
            range=(0, N_POINTS + 2)
        )
        global_hist = global_hist.astype("float32")
        global_hist /= (global_hist.sum() + 1e-6)

        # Spatial Grid LBP Histograms
        cell_h, cell_w = lbp.shape[0] // GRID_SIZE, lbp.shape[1] // GRID_SIZE
        spatial_histograms = []

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                cell = lbp[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
                hist, _ = np.histogram(cell.ravel(), bins=np.arange(0, N_POINTS + 3), range=(0, N_POINTS + 2))
                hist = hist.astype("float32")
                hist /= (hist.sum() + 1e-6)
                spatial_histograms.extend(hist)

        spatial_vec = np.array(spatial_histograms, dtype=np.float32)
        spatial_vec /= (np.linalg.norm(spatial_vec) + 1e-6)

        return {
            "global": global_hist,
            "spatial": spatial_vec
        }

    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        return None


def chi_square(h1, h2, eps=1e-10):
    return 0.5 * np.sum(((h1 - h2) ** 2) / (h1 + h2 + eps))


def compare_images(ref_path, test_path):
    f1 = extract_features(ref_path)
    f2 = extract_features(test_path)

    if f1 is None or f2 is None:
        print("[QUALITY GATE ALERT] Closed eye or poor quality frame detected during comparison!")
        return 1.0  # Safe fallback (denies access)

    global_dist = chi_square(f1["global"], f2["global"])
    spatial_dist = chi_square(f1["spatial"], f2["spatial"])

    final_score = (0.7 * global_dist) + (0.3 * spatial_dist)
    return float(np.clip(final_score, 0.0, 1.0))