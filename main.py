import os
import glob



import cv2
import numpy as np


# Project Module Imports
from test_camera import capture_frame
from preprocess_iris import preprocess_latest
from segment_iris import segment_iris
from secure_vault import SecureVault
from liveness_detector import LivenessDetector
from feature_extraction import compare_images  # Updated to Spatial LBP Chi-Square Engine

# -----------------------------
# Paths
# -----------------------------
USER_DB = "data/users"
os.makedirs(USER_DB, exist_ok=True)




# -----------------------------
# REGISTER USER
# -----------------------------
def register_user():
    """
    Enrolls a new user by capturing exactly 3 high-quality live iris templates.
    Encrypts templates using AES via SecureVault.
    """
    username = input("Enter username: ").strip()


    if username == "":
        print("[ERROR] Empty username")
        return

    user_path = os.path.join(USER_DB, username)
    os.makedirs(user_path, exist_ok=True)

    # Clean existing templates if re-registering (*.enc files)
    for existing_file in glob.glob(os.path.join(user_path, "*.enc")):
        try:
            os.remove(existing_file)
        except OSError:
            pass

    target_samples = 3
    successful_samples = 0
    liveness_engine = LivenessDetector()

    print(f"[INFO] Registering user '{username}' with {target_samples} valid live samples...")

    while successful_samples < target_samples:
        print(f"\n[INFO] Capture sample {successful_samples + 1}/{target_samples}")

        # Capture frame from webcam
        raw_image_path = capture_frame()
        if raw_image_path is None:
            print("[ERROR] Capture canceled or camera failed. Retaking this sample!")
            break

        # Anti-Spoofing Check on Registration Frame
        liveness_result = liveness_engine.analyze_frame(raw_image_path)
        if not liveness_result['passed']:
            print(f"[SECURITY REJECT] 🛑 {liveness_result['reason']}. Retaking sample...")
            continue

        # Extract eye ROI
        processed_path = preprocess_latest()
        if processed_path is None:
            print("[ERROR] Preprocessing failed (No eyes detected). Retaking this sample!")
            continue

        # Segment iris and save/encrypt template
        saved_iris = segment_iris(user_path, processed_file_path=processed_path, username=username)
        if saved_iris is None:
            print("[ERROR] Iris localization failed. Retaking this sample!")
            continue

        successful_samples += 1
        print(f"[INFO] Sample {successful_samples}/{target_samples} successfully saved!")

    print(f"\n[SUCCESS] User '{username}' registered with {target_samples} valid encrypted templates!")


# -----------------------------
# LOGIN USER
# -----------------------------
def login_user():
    """
    Captures a login sample, executes liveness verification, decrypts user templates 
    in-memory, and evaluates Spatial LBP Chi-Square scores.
    """
    username = input("Enter username: ").strip()
    user_path = os.path.join(USER_DB, username)

    if not os.path.exists(user_path):
        print("[ERROR] User not found!")
        return

    print("[INFO] Capturing login image...")

    raw_image_path = capture_frame()
    if raw_image_path is None:
        print("[ERROR] Login capture canceled. Authentication aborted.")
        return

    # Anti-Spoofing / Liveness Firewall
    print("[INFO] Executing Liveness & Anti-Spoofing Analysis...")
    liveness_engine = LivenessDetector()
    liveness_result = liveness_engine.analyze_frame(raw_image_path)

    print(f"[DEBUG] Anti-Spoof Status: {liveness_result['reason']}")
    
    if not liveness_result['passed']:
        print(f"\n[SECURITY ALERT] 🛑 {liveness_result['reason']}")
        print("[SECURITY ALERT] Authentication rejected due to presentation attack detection.")
        return

    print("[INFO] Preprocessing...")
    processed_path = preprocess_latest()
    if processed_path is None:
        print("[ERROR] Eye detection failed. Authentication aborted.")
        return

    print("[INFO] Segmenting iris...")

    temp_folder = "data/temp_login"
    os.makedirs(temp_folder, exist_ok=True)


    test_iris = segment_iris(temp_folder, processed_file_path=processed_path)


    if test_iris is None:
        print("[ERROR] Segmentation failed. Please align your eye properly.")
        return

    vault = SecureVault()

    # Save temporary decrypted login frame for comparison
    test_iris_jpg = os.path.join(temp_folder, "login_test_iris.jpg")
    
    if test_iris.endswith(".enc"):
        with open(test_iris, "rb") as sf:
            ciphertext = sf.read()
        decrypted_bytes = vault.decrypt_template(ciphertext)
        np_arr = np.frombuffer(decrypted_bytes, dtype=np.uint8)
        decrypted_matrix = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(test_iris_jpg, decrypted_matrix)
    else:

        img = cv2.imread(test_iris, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(test_iris_jpg, img)

    test_iris = test_iris_jpg


    # Locate secure encrypted templates (.enc)
    user_iris_encrypted = glob.glob(os.path.join(user_path, "*.enc"))

    if not user_iris_encrypted:
        print("[ERROR] No secure cryptographic templates found for this user.")
        return

    print("[INFO] Decrypting Vault & Extracting Spatial LBP Features...")
    scores = []







    decrypted_temp_path = os.path.join(temp_folder, "secure_decrypted_temp.jpg")

    for secure_ref in user_iris_encrypted:
        try:

            with open(secure_ref, "rb") as sf:
                ciphertext = sf.read()







            decrypted_bytes = vault.decrypt_template(ciphertext)



            np_arr = np.frombuffer(decrypted_bytes, dtype=np.uint8)
            decrypted_matrix = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
         
            cv2.imwrite(decrypted_temp_path, decrypted_matrix)
            
            # Compute Spatial LBP Chi-Square Score
            score = compare_images(decrypted_temp_path, test_iris)
            print(f"[DEBUG] Template ({os.path.basename(secure_ref)}) Chi-Square Score: {score:.4f}")
            scores.append(score)


        except Exception as e:
            print(f"[ERROR] Failed to verify template {secure_ref}: {str(e)}")
        finally:
            if os.path.exists(decrypted_temp_path):
                os.remove(decrypted_temp_path)


    # Cleanup login temporary test sample
    if os.path.exists(test_iris):
        try:
            os.remove(test_iris)
        except OSError:
            pass

    if not scores:
        print("[CRITICAL] Verification failed. Could not process templates.")
        return

    avg_score = sum(scores) / len(scores)
    

    min_score = min(scores)
    close_matches = [score for score in scores if score < 0.28]

    print(f"\n[INFO] Average Distance: {avg_score:.4f}")
    print(f"[INFO] Best Distance: {min_score:.4f}")
    print(f"[INFO] Valid Sample Matches (<0.28): {len(close_matches)}/{len(scores)}")

    # -----------------------------
    # CALIBRATED DECISION BOUNDARIES
    # -----------------------------
    if min_score < 0.28 and len(close_matches) >= 2:
        print(f"\n[RESULT] LOGIN SUCCESS ✅ Welcome {username}!")
    elif min_score < 0.35:
        print("\n[RESULT] UNCERTAIN MATCH ⚠️ Try again with better alignment.")
    else:
        print("\n[RESULT] LOGIN FAILED ❌ Access Denied: Mismatch!")


# -----------------------------
# CLI MENU
# -----------------------------
def menu():
    print("\n===== OPEN-IRIS AUTHENTICATION SYSTEM =====")
    print("1. Register User")
    print("2. Login User")
    print("3. Exit")


# -----------------------------
# MAIN LOOP
# -----------------------------
if __name__ == "__main__":
    while True:
        menu()
        choice = input("Enter choice: ").strip()

        if choice == "1":
            register_user()

        elif choice == "2":


            login_user()

        elif choice == "3":
            print("[INFO] Exiting system...")
            break

        else:
            print("[ERROR] Invalid option. Selection must be 1, 2, or 3.")