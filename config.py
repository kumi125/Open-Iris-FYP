import os

# -----------------------------
# DIRECTORY PATHS
# -----------------------------
USER_DB_DIR = os.path.join("data", "users")
TEMP_LOGIN_DIR = os.path.join("data", "temp_login")

# Ensure required directories exist
os.makedirs(USER_DB_DIR, exist_ok=True)
os.makedirs(TEMP_LOGIN_DIR, exist_ok=True)

# -----------------------------
# BIOMETRIC & SECURITY THRESHOLDS
# -----------------------------
# Feature Distance Match Target (Webcam Calibrated)
MATCH_THRESHOLD = 0.28  

# Anti-Spoofing / FFT High-Frequency Limit
SPOOF_THRESHOLD = 155.0 

# Sample Requirements
REGISTRATION_SAMPLES = 3
MIN_CONSENSUS_MATCHES = 2