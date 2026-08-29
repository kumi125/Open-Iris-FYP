import os
import glob
import cv2
import numpy as np

# Project Module Imports
from feature_extraction import compare_images
from secure_vault import SecureVault

# -----------------------------
# Configuration & Paths
# -----------------------------
USER_DB = "data/users"
TEMP_DIR = "data/temp_eval"
os.makedirs(TEMP_DIR, exist_ok=True)

THRESHOLD = 0.50  # Calibrated Chi-Square threshold for Spatial Polar LBP


def decrypt_template_to_temp(encrypted_path, vault, temp_output_path):
    """
    Decrypts an encrypted .enc template directly to a temporary JPEG file for evaluation.
    """
    try:
        with open(encrypted_path, "rb") as f:
            ciphertext = f.read()
            
        decrypted_bytes = vault.decrypt_template(ciphertext)
        np_arr = np.frombuffer(decrypted_bytes, dtype=np.uint8)
        decrypted_matrix = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        
        cv2.imwrite(temp_output_path, decrypted_matrix)
        return True
    except Exception as e:
        print(f"[ERROR] Decryption failed for {encrypted_path}: {e}")
        return False


def evaluate():
    if not os.path.exists(USER_DB):
        print(f"[ERROR] User database folder '{USER_DB}' not found.")
        return

    users = [u for u in os.listdir(USER_DB) if os.path.isdir(os.path.join(USER_DB, u))]
    
    if len(users) < 2:
        print("[WARNING] Evaluation requires at least 2 registered users to calculate False Acceptance Rate (FAR).")

    vault = SecureVault()
    
    genuine_total = 0
    genuine_fail = 0    # False Rejections

    impostor_total = 0
    impostor_accept = 0  # False Acceptances

    print("[INFO] Loading encrypted templates from database...")

    # Iterate through all registered users
    for user in users:
        user_path = os.path.join(USER_DB, user)
        enc_templates = glob.glob(os.path.join(user_path, "*.enc"))

        if not enc_templates:
            continue

        # Decrypt current user's templates to temporary evaluation folder
        user_temp_files = []
        for idx, enc_path in enumerate(enc_templates):
            temp_path = os.path.join(TEMP_DIR, f"{user}_template_{idx}.jpg")
            if decrypt_template_to_temp(enc_path, vault, temp_path):
                user_temp_files.append(temp_path)

        # -----------------------------
        # 1. Genuine Comparisons (Same User)
        # -----------------------------
        for i in range(len(user_temp_files)):
            for j in range(i + 1, len(user_temp_files)):
                score = compare_images(user_temp_files[i], user_temp_files[j])
                genuine_total += 1

                if score >= THRESHOLD:
                    genuine_fail += 1

        # -----------------------------
        # 2. Impostor Comparisons (Different Users)
        # -----------------------------
        for other_user in users:
            if other_user == user:
                continue

            other_path = os.path.join(USER_DB, other_user)
            other_enc_templates = glob.glob(os.path.join(other_path, "*.enc"))

            for img1_path in user_temp_files:
                for idx, other_enc in enumerate(other_enc_templates):
                    other_temp_path = os.path.join(TEMP_DIR, f"{other_user}_other_template_{idx}.jpg")
                    
                    if decrypt_template_to_temp(other_enc, vault, other_temp_path):
                        score = compare_images(img1_path, other_temp_path)
                        impostor_total += 1

                        if score < THRESHOLD:
                            impostor_accept += 1

                        # Clean up temporary decrypted impostor image
                        if os.path.exists(other_temp_path):
                            os.remove(other_temp_path)

        # Clean up temporary user templates
        for temp_file in user_temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    # -----------------------------
    # Calculate System Metrics
    # -----------------------------
    FAR = (impostor_accept / impostor_total * 100) if impostor_total > 0 else 0.0
    FRR = (genuine_fail / genuine_total * 100) if genuine_total > 0 else 0.0
    
    total_tests = genuine_total + impostor_total
    correct_decisions = (genuine_total - genuine_fail) + (impostor_total - impostor_accept)
    accuracy = (correct_decisions / total_tests * 100) if total_tests > 0 else 0.0

    print("\n" + "="*45)
    print("      BIOMETRIC SYSTEM PERFORMANCE REPORT")
    print("="*45)
    print(f" Evaluation Threshold       : {THRESHOLD}")
    print(f" Total Genuine Tests        : {genuine_total}")
    print(f" Total Impostor Tests       : {impostor_total}")
    print("-" * 45)
    print(f" False Acceptance Rate (FAR): {FAR:.2f}% ({impostor_accept}/{impostor_total if impostor_total > 0 else 1})")
    print(f" False Rejection Rate (FRR) : {FRR:.2f}% ({genuine_fail}/{genuine_total if genuine_total > 0 else 1})")
    print(f" Overall System Accuracy    : {accuracy:.2f}%")
    print("="*45 + "\n")

if __name__ == "__main__":
    evaluate()