import os
import glob
import logging
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

# Project Imports & Config
import config
from test_camera import capture_frame
from preprocess_iris import preprocess_latest
from segment_iris import segment_iris
from feature_extraction import compare_images
from secure_vault import SecureVault
from liveness_detector import LivenessDetector

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S"
)

# Initialize Liveness Engine
liveness_engine = LivenessDetector(model_path="models/liveness_mobilenet.pth")


# -----------------------------
# ANTI-SPOOFING / QUALITY HELPER
# -----------------------------
def check_screen_spoof(image_path: str) -> tuple[bool, str]:
    """
    Detects digital screen presentation attacks (Moiré patterns) using 2D FFT.


    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, "Invalid image frame"

    # Crop central region (focus on eye/face ROI)
    h_orig, w_orig = img.shape
    crop = img[int(h_orig * 0.2):int(h_orig * 0.8), int(w_orig * 0.2):int(w_orig * 0.8)]

    resized = cv2.resize(crop, (300, 300))
    f_transform = np.fft.fft2(resized)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)

    h, w = magnitude_spectrum.shape
    center_h, center_w = h // 2, w // 2
    
    # Filter out low frequencies
    magnitude_spectrum[center_h - 35:center_h + 35, center_w - 35:center_w + 35] = 0
    high_freq_score = float(np.mean(magnitude_spectrum))

    logging.info(f"Anti-Spoof FFT Score: {high_freq_score:.2f} (Threshold: {config.SPOOF_THRESHOLD})")

    if high_freq_score > config.SPOOF_THRESHOLD:
        return False, f"Screen presentation attack detected (Score: {high_freq_score:.1f})"

    return True, "Passed texture analysis"


# -----------------------------
# UI STATUS UPDATER HELPER
# -----------------------------
def set_status(text: str, state: str = "info") -> None:
    """Updates the dynamic status monitor card in GUI."""
    colors = {
        "info": ("#1E293B", "#38BDF8"),     # Dark Slate / Cyan
        "success": ("#064E3B", "#34D399"),  # Dark Emerald / Green
        "warning": ("#451A03", "#FBBF24"),  # Dark Amber / Gold
        "error": ("#4C0519", "#FB7185")     # Dark Rose / Red
    }
    bg_color, fg_color = colors.get(state, colors["info"])
    status_card.config(bg=bg_color)
    status_label.config(text=text, fg=fg_color, bg=bg_color)
    root.update()


# -----------------------------
# USER REGISTRATION WORKFLOW
# -----------------------------
def register_user() -> None:
    """Handles multi-sample user registration and AES template encryption."""
    username = entry_username.get().strip()

    if not username:
        messagebox.showerror("Validation Error", "Please enter a valid username.")
        return

    user_path = os.path.join(config.USER_DB_DIR, username)
    os.makedirs(user_path, exist_ok=True)

    # Clean existing templates if re-registering
    for existing_file in glob.glob(os.path.join(user_path, "*.*")):
        try:
            os.remove(existing_file)
        except OSError:
            pass

    set_status("Initializing Registration Pipeline...", "info")




    successful_samples = 0
    while successful_samples < config.REGISTRATION_SAMPLES:
        set_status(f"Capture Sample {successful_samples + 1}/{config.REGISTRATION_SAMPLES} → Press 's'", "warning")

        raw_image_path = capture_frame()
        if raw_image_path is None:
            messagebox.showwarning("Registration Warning", "Capture canceled. Retrying current sample.")
            continue

        processed_path = preprocess_latest()
        if processed_path is None:
            messagebox.showwarning("Localization Error", f"Eye localization failed. Retrying sample {successful_samples + 1}!")
            continue


        saved_iris = segment_iris(user_path, processed_file_path=processed_path, username=username)

        if saved_iris is None:
            messagebox.showwarning("Segmentation Error", f"Iris segmentation failed. Retrying sample {successful_samples + 1}!")
            continue


        successful_samples += 1
        set_status(f"Sample {successful_samples}/{config.REGISTRATION_SAMPLES} Encrypted & Saved!", "success")

    set_status(f"Registration Complete for '{username}'", "success")
    logging.info(f"User '{username}' registered with {config.REGISTRATION_SAMPLES} templates.")
    messagebox.showinfo("Registration Complete", f"User '{username}' successfully registered.")


# -----------------------------
# USER AUTHENTICATION WORKFLOW
# -----------------------------
def login_user() -> None:
    """Executes full liveness verification, vault decryption, and biometric matching."""
    username = entry_username.get().strip()
    user_path = os.path.join(config.USER_DB_DIR, username)

    if not os.path.exists(user_path):
        messagebox.showerror("Authentication Error", f"Username '{username}' not found in database.")
        set_status("User record missing", "error")
        return

    set_status("Capturing biometric frame...", "info")
    raw_image_path = capture_frame()
    if raw_image_path is None:
        set_status("Authentication canceled", "error")
        return
    
    # 1. Anti-Spoofing: FFT Analysis
    set_status("Executing Liveness & Anti-Spoofing Checks...", "warning")
    passed_spoof, spoof_msg = check_screen_spoof(raw_image_path)
    if not passed_spoof:
        set_status("Anti-Spoofing Alert: Screen Photo Detected", "error")
        messagebox.showerror("Security Violation", f"🛑 Access Denied: {spoof_msg}")
        return

    # 2. Anti-Spoofing: PyTorch CNN Liveness Engine
    frame = cv2.imread(raw_image_path)
    is_live, confidence = liveness_engine.predict(frame, confidence_threshold=0.35)
    if not is_live:
        set_status("Liveness Verification Failed", "error")
        messagebox.showerror("Liveness Error", f"🛑 Access Denied: Spoof / Presentation attack detected (Confidence: {confidence * 100:.1f}%)")
        return

    set_status("Preprocessing & Localizing Iris Region...", "info")
    processed_path = preprocess_latest()
    if processed_path is None:
        set_status("Preprocessing Error: Eye not clear", "error")
        messagebox.showerror("Error", "Eye preprocessing failed. Keep eye centered and open.")
        return

    set_status("Segmenting Iris Boundaries...", "info")
    test_iris = segment_iris(config.TEMP_LOGIN_DIR, processed_file_path=processed_path)
    if test_iris is None:
        set_status("Segmentation Error", "error")
        messagebox.showerror("Error", "Segmentation failed. Could not isolate iris.")
        return

    vault = SecureVault()

    # 3. Resolve test iris file path
    test_iris_final = test_iris
    if test_iris.endswith(".enc"):
        test_iris_jpg = os.path.join(config.TEMP_LOGIN_DIR, "login_test_decrypted.jpg")
        try:
            with open(test_iris, "rb") as sf:
                ciphertext = sf.read()
            decrypted_bytes = vault.decrypt_template(ciphertext)
            np_arr = np.frombuffer(decrypted_bytes, dtype=np.uint8)
            decrypted_matrix = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            if decrypted_matrix is None:
                raise ValueError("Decrypted test sample matrix is empty.")
            cv2.imwrite(test_iris_jpg, decrypted_matrix)
            test_iris_final = test_iris_jpg
        except Exception as e:
            logging.error(f"Failed to decrypt login test sample: {str(e)}")
            set_status("Vault Decryption Error", "error")
            messagebox.showerror("Error", "Failed to decrypt authentication sample.")
            return

    # Quality Gate Check
    test_img = cv2.imread(test_iris_final, cv2.IMREAD_GRAYSCALE)
    if test_img is None or np.std(test_img) < 14.0:
        set_status("Quality Gate: Closed Eye / Blank Crop", "error")
        messagebox.showerror("Quality Error", "❌ Invalid sample. Please open eyes clearly.")
        return

    # Retrieve stored user templates
    user_iris_encrypted = glob.glob(os.path.join(user_path, "*.enc"))
    if not user_iris_encrypted:
        user_iris_encrypted = glob.glob(os.path.join(user_path, "*.jpg"))

    if not user_iris_encrypted:
        set_status("No templates found", "error")

        messagebox.showerror("Error", "No registered templates found for this user.")

        return

    # 4. Decrypt and compare each stored user template
    set_status("Decrypting Vault & Extracting Features...", "info")
    scores = []
    decrypted_temp_path = os.path.join(config.TEMP_LOGIN_DIR, "secure_decrypted_ref.jpg")

    for secure_ref in user_iris_encrypted:
        try:
            if secure_ref.endswith(".enc"):
                with open(secure_ref, "rb") as sf:
                    ciphertext = sf.read()
                decrypted_bytes = vault.decrypt_template(ciphertext)
                np_arr = np.frombuffer(decrypted_bytes, dtype=np.uint8)
                decrypted_matrix = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
                if decrypted_matrix is None:
                    continue
                cv2.imwrite(decrypted_temp_path, decrypted_matrix)
                ref_path = decrypted_temp_path
            else:
                ref_path = secure_ref

            score = compare_images(ref_path, test_iris_final)
            logging.info(f"Template Compare Score ({os.path.basename(secure_ref)}): {score:.4f}")
            scores.append(score)



        except Exception as e:
            logging.error(f"Verification error on template {secure_ref}: {str(e)}")
        finally:
            if os.path.exists(decrypted_temp_path):
                try:
                    os.remove(decrypted_temp_path)
                except OSError:
                    pass

    if not scores:
        set_status("Template processing failure", "error")
        messagebox.showerror("Error", "Could not process or decrypt user templates.")
        return

    min_score = min(scores)
    avg_score = sum(scores) / len(scores)

    # Decision Boundaries for Chi-Square Spatial LBP
    MATCH_LIMIT = 0.40
    UNCERTAIN_LIMIT = 0.50

    valid_matches = [s for s in scores if s < MATCH_LIMIT]
    logging.info(f"Auth Decision -> User: {username} | Min Score: {min_score:.4f} | Avg Score: {avg_score:.4f} | Valid Matches (<{MATCH_LIMIT}): {len(valid_matches)}/{len(scores)}")

    # Update UI based on decision thresholds
    if min_score < MATCH_LIMIT:
        set_status(f"ACCESS GRANTED: Welcome {username}!", "success")
        messagebox.showinfo("Authentication Success", f"✅ Biometric Identity Verified!\nWelcome back, {username}.\n\nDistance Score: {min_score:.3f}")
    elif min_score < UNCERTAIN_LIMIT:
        set_status("UNCERTAIN MATCH: Adjust distance & lighting", "warning")
        messagebox.showwarning("Confidence Warning", f"⚠️ Low Confidence Match (Score: {min_score:.3f}).\nEnsure direct lighting and look straight into camera.")
    else:
        set_status("ACCESS DENIED: Biometric Mismatch", "error")
        messagebox.showerror("Authentication Failure", f"❌ Access Denied. Biometric template mismatch.\n\nBest Match Score: {min_score:.3f} (Required < {MATCH_LIMIT})")

    # Clean up temporary login files after authentication completion
    for f in glob.glob(os.path.join(config.TEMP_LOGIN_DIR, "*")):
        try:
            os.remove(f)
        except OSError:
            pass




# -----------------------------
# GUI INTERFACE INITIALIZATION
# -----------------------------
root = tk.Tk()
root.title("Iris Recognition & Verification System")
root.geometry("600x580")
root.minsize(500, 500)
root.configure(bg="#0B0F19")

root.columnconfigure(0, weight=1)
root.rowconfigure(1, weight=1)

# Header Banner
header_frame = tk.Frame(root, bg="#111827", pady=20, padx=20)
header_frame.grid(row=0, column=0, sticky="ew")

title_lbl = tk.Label(header_frame, text="👁️ OPEN-IRIS AUTHENTICATOR", font=("Segoe UI", 18, "bold"), fg="#F9FAFB", bg="#111827")
title_lbl.pack()

subtitle_lbl = tk.Label(header_frame, text="AES-256 Encrypted Biometric Access Control", font=("Segoe UI", 10), fg="#9CA3AF", bg="#111827")
subtitle_lbl.pack(pady=(4, 0))

# Main Container
main_container = tk.Frame(root, bg="#0B0F19", padx=30, pady=25)
main_container.grid(row=1, column=0, sticky="nsew")
main_container.columnconfigure(0, weight=1)

# Input Card
card_frame = tk.Frame(main_container, bg="#1F2937", padx=25, pady=25, relief="flat")
card_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
card_frame.columnconfigure(0, weight=1)

lbl_user = tk.Label(card_frame, text="USER IDENTITY / USERNAME", font=("Segoe UI", 9, "bold"), fg="#D1D5DB", bg="#1F2937")
lbl_user.grid(row=0, column=0, sticky="w", pady=(0, 8))

entry_username = tk.Entry(
    card_frame, 
    font=("Segoe UI", 13), 
    bg="#111827", 
    fg="#F9FAFB", 
    insertbackground="#F9FAFB", 
    relief="solid", 
    bd=1,
    justify='center'
)
entry_username.grid(row=1, column=0, sticky="ew", ipady=8)
entry_username.focus()

# Action Buttons
btn_frame = tk.Frame(main_container, bg="#0B0F19")
btn_frame.grid(row=1, column=0, sticky="ew", pady=(0, 20))
btn_frame.columnconfigure(0, weight=1)
btn_frame.columnconfigure(1, weight=1)

btn_login = tk.Button(
    btn_frame, 
    text="🔒 LOGIN / AUTHENTICATE", 
    font=("Segoe UI", 11, "bold"), 
    bg="#2563EB", 
    fg="#FFFFFF", 
    activebackground="#1D4ED8", 
    activeforeground="#FFFFFF", 
    relief="flat", 
    cursor="hand2", 
    command=login_user
)
btn_login.grid(row=0, column=0, sticky="ew", ipady=10, padx=(0, 8))

btn_register = tk.Button(
    btn_frame, 
    text="➕ REGISTER USER", 
    font=("Segoe UI", 11, "bold"), 
    bg="#059669", 
    fg="#FFFFFF", 
    activebackground="#047857", 
    activeforeground="#FFFFFF", 
    relief="flat", 
    cursor="hand2", 
    command=register_user
)
btn_register.grid(row=0, column=1, sticky="ew", ipady=10, padx=(8, 0))

# Status Monitor Panel
status_card = tk.Frame(main_container, bg="#1E293B", pady=15, padx=15)
status_card.grid(row=2, column=0, sticky="ew")

status_label = tk.Label(
    status_card, 
    text="System Ready • Waiting for input...", 
    font=("Segoe UI", 10, "bold"), 
    fg="#38BDF8", 
    bg="#1E293B", 
    wraplength=450
)
status_label.pack()

# Footer
footer_lbl = tk.Label(root, text="Final Year Project • Secure Iris Biometric System", font=("Segoe UI", 8), fg="#4B5563", bg="#0B0F19")
footer_lbl.grid(row=2, column=0, sticky="s", pady=10)

root.mainloop()