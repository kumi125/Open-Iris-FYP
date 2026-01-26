# 🟢 Open-Iris-FYP

### Stage 1 – Camera & Image Capture ✅

**Project Status:** Stage 1 complete | Stage 2 ⬜  

---

## 📌 Project Summary
Open-Iris-FYP is a Final Year Project that captures **iris images using a webcam** to test hardware and OpenCV feasibility. Stage 1 proves that the system works and is ready for further iris recognition development.

---

## ⚡ Key Features (Stage 1)
- Open webcam and live video stream  
- Capture iris images  
- Save captured images to disk  
- Hardware & OpenCV integration verified  

---

## 🏃‍♂️ Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run Stage 1 script
python test_camera.py

## Stage 2 – Dataset Preparation & Preprocessing ✅

### Objective
The goal of Stage 2 is to prepare a standardized iris image dataset suitable for feature extraction and recognition.  
Raw iris images often vary in size, lighting, and noise, so preprocessing is required to normalize them.

---

### Dataset
- Public iris dataset (e.g., MMU / public iris dataset)
- Dataset is stored locally and **excluded from GitHub using `.gitignore`**
- Folder structure:


---

### Preprocessing Steps Implemented
The following preprocessing operations are applied to each iris image:

1. Image loading from dataset directory  
2. Conversion to grayscale  
3. Image resizing to a fixed resolution (224 × 224)  
4. Noise reduction using Gaussian blur  
5. Saving the processed images to a separate directory  

This ensures all images have consistent format and quality for later stages.

---

### Script
Preprocessing is implemented in:
preprocess_iris.py


📂 **Folder Structure**
Open-Iris-FYP/
├── test_camera.py
data/
├── raw/ # Original iris images
├── processed/ # Preprocessed iris images
├── README.md
├── .gitignore
├── iris-sample/
