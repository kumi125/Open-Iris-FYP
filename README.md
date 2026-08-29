# 👁️ Open-Iris: AES-256 Encrypted Biometric Access Control

Open-Iris is a biometric authentication and identity verification system. It combines real-time eye localization, Hough circle segmentation, 2D FFT anti-spoofing, and AES-256 template encryption at rest.

---

## 🌟 Key Features

* **Biometric Feature Extraction:** Captures, preprocesses, and isolates pupil/iris regions using OpenCV and Hough Transform logic.
* **AES-256 Biometric Vault:** Encrypts saved iris templates (`.enc`) at rest to prevent template theft or replay attacks.
* **Dual-Layer Anti-Spoofing:**
  * **2D FFT Moiré Analysis:** Filters frequency spectra to detect high-frequency artifacts caused by screen presentation attacks.
  * **Liveness Detection Engine:** Ensures proper eye closure/opening state before proceeding to feature comparison.
* **Majority Consensus Matching:** Evaluates candidates against 3 stored templates using distance thresholds to maximize accuracy and minimize False Acceptance Rates (FAR).
* **Modern Adaptive GUI:** Fully resizable Tkinter dashboard with dynamic status monitor cards and visual state feedback.

 ---

## 📁 System Architecture

```text
Open-Iris-FYP/
├── config.py              # Central system parameters & match thresholds
├── gui_app.py             # Main GUI application & orchestration logic
├── test_camera.py         # Frame capture module
├── preprocess_iris.py     # Eye region localization & preprocessing
├── segment_iris.py        # Hough Circles boundary extraction
├── main.py                # Feature comparison & distance calculation
├── secure_vault.py        # AES-256 template encryption / decryption engine
├── liveness_detector.py   # Anti-spoofing & liveness verification module
├── data/                  # Biometric database & processing pipelines
│   ├── captured/          # Raw frame captures
│   ├── processed/         # Preprocessed grayscale crops
│   ├── segmented/         # Isolated iris boundary backups
│   ├── temp_login/        # In-memory authentication artifacts
│   └── users/             # AES-encrypted user templates (.enc)
└── requirements.txt       # Project dependencies
```
  

 ## 🚀 Getting Started
 ### Prerequisites
  Python 3.9+  
  Active Webcam  
  OpenCV Installed  

 ### Installation
  Clone the repository: git clone [https://github.com/kumi125/Open-Iris-FYP.git](https://github.com/kumi125/Open-Iris-FYP.git)  
  cd Open-Iris-FYP  

  ### Install dependencies:
   pip install -r requirements.txt  
   
  ### Launch the application:
   python gui_app.py


## 🔒 Security & Anti-Spoofing Calibration
 * Matching Threshold: Default match boundary set to 0.28 (Calibrated for standard webcams).
 * FFT High-Frequency Threshold: Screen detection score capped at 155.0 to filter out display moiré patterns.
 * Authentication Rule: Access requires $\ge 2/3$ positive template matches or a single high-confidence match ($< 0.22$).


## 📜 License
 Developed as a Final Year Project for Computer Science & Cyber Security.