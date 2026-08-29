import os
import cv2

import numpy as np
from secure_vault import SecureVault

def segment_iris(target_folder, processed_file_path, username=None):

    os.makedirs(target_folder, exist_ok=True)

    # 1. Load preprocessed frame
    img = cv2.imread(processed_file_path)
    if img is None:
        print(f"[ERROR] Could not read processed image: {processed_file_path}")
        return None



    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape  # 224x224
    
    # Force coordinates to start around the frame center (112, 112)
    center_x, center_y = w // 2, h // 2

    # 2. LOCATE PUPIL INSIDE CENTRAL INNER SQUARE ONLY (Ignore all edges)
    # Define a strict 80x80 central window
    crop_size = 40
    central_roi = gray[center_y - crop_size : center_y + crop_size, 
                       center_x - crop_size : center_x + crop_size]

    # Find local minimum (darkest point) inside the central box
    blurred_roi = cv2.GaussianBlur(central_roi, (5, 5), 0)
    _, _, min_loc, _ = cv2.minMaxLoc(blurred_roi)

    # Convert ROI relative location back to image coordinates
    x_pupil = (center_x - crop_size) + min_loc[0]
    y_pupil = (center_y - crop_size) + min_loc[1]

    # Set realistic biometric radii (Pupil ~ 18-24px, Iris ~ 50-65px)
    r_pupil = 20
    r_iris = 58

    # 3. STRICT MASKING
    mask = np.zeros((h, w), dtype=np.uint8)

    # Outer Iris Ring (White)
    cv2.circle(mask, (x_pupil, y_pupil), r_iris, 255, -1)
    
    # Inner Pupil (Blackout center)
    cv2.circle(mask, (x_pupil, y_pupil), r_pupil, 0, -1)

    # Isolate iris ring pixels
    isolated_iris = cv2.bitwise_and(gray, gray, mask=mask)

    # Crop square ROI
    y1 = max(0, y_pupil - r_iris)
    y2 = min(h, y_pupil + r_iris)
    x1 = max(0, x_pupil - r_iris)
    x2 = min(w, x_pupil + r_iris)

    iris_crop = isolated_iris[y1:y2, x1:x2]

    if iris_crop.size == 0:
        print("[ERROR] Extracted iris matrix is empty.")
        return None

    iris_crop = cv2.resize(iris_crop, (224, 224))

    # --------------------------------------------------------
    # Cryptographic Storage & Backup
    # --------------------------------------------------------

    prefix = username if username else "user"


    existing_files = [f for f in os.listdir(target_folder) if f.startswith(f"{prefix}_iris_") and f.endswith(".enc")]
    next_index = len(existing_files) + 1


    filename = f"{prefix}_iris_{next_index}.enc"
    full_user_path = os.path.join(target_folder, filename)


    try:

        vault = SecureVault()


        success, encoded_image = cv2.imencode('.jpg', iris_crop)
        if not success:

            return None


        raw_bytes = encoded_image.tobytes()


        encrypted_data = vault.encrypt_template(raw_bytes)


        with open(full_user_path, "wb") as secure_file:
            secure_file.write(encrypted_data)
        
        print(f"[SECURE] Biometric template encrypted: {full_user_path}")

    except Exception as e:
        print(f"[CRITICAL ERROR] Encryption failed: {str(e)}")
        return None


    central_segmented_folder = "data/segmented"
    os.makedirs(central_segmented_folder, exist_ok=True)
    central_path = os.path.join(central_segmented_folder, f"{prefix}_iris_{next_index}.jpg")
    cv2.imwrite(central_path, iris_crop)
    print(f"[INFO] Central backup saved to: {central_path}")

    return full_user_path