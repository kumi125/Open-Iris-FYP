import cv2
import os
import glob
import numpy as np

def preprocess_latest():

    capture_folder = "data/captured"
    processed_folder = "data/processed"
    os.makedirs(processed_folder, exist_ok=True)

    # Latest captured image
    list_of_files = glob.glob(os.path.join(capture_folder, "*.jpg"))

    if len(list_of_files) == 0:
        print("[ERROR] No captured images found!")
        return None

    latest_file = max(list_of_files, key=os.path.getctime)
    image = cv2.imread(latest_file)

    if image is None:
        print("[ERROR] Failed to load image!")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape

    # Eye detection cascade
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=8,
        minSize=(70, 70)
    )

    if len(eyes) == 0:
        print("[ERROR] No eyes detected in frame!")
        return None

    # FIX 1: CONSISTENT EYE LOCKING
    # Sort eyes left-to-right based on x-coordinate, and consistently pick the Left Eye (x min)
    eyes = sorted(eyes, key=lambda e: e[0])
    (x, y, w, h) = eyes[0]  # Always lock onto the primary left-hand side eye frame

    eye_roi = gray[y:y+h, x:x+w]

    # Resize to standard size
    resized = cv2.resize(eye_roi, (224, 224))

    # FIX 2: CLOSED-EYE & PUPIL PRESENCE GUARDRAIL
    # Analyze central ROI for dark pupil structure
    center_crop = resized[60:164, 60:164]
    
    # Eyelids/skin have higher average intensity and low dark-pixel ratio
    mean_val = np.mean(center_crop)
    dark_pixel_ratio = np.sum(center_crop < 60) / center_crop.size

    # Check for closed eye / lack of open pupil
    if dark_pixel_ratio < 0.05 or mean_val > 150:
        print("[ERROR] Eye appears CLOSED or pupil is obscured. Please open eyes wide and look directly into camera.")
        return None

    # Apply adaptive contrast enhancement & mild blur for segmentation readiness
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Save processed frame
    existing_files = [f for f in os.listdir(processed_folder) if f.endswith(".png")]
    count = len(existing_files)

    filename = f"processed_{count}.png"
    processed_path = os.path.join(processed_folder, filename)

    cv2.imwrite(processed_path, blurred)

    print(f"[INFO] Preprocessed image saved as: {processed_path}")
    
    return processed_path