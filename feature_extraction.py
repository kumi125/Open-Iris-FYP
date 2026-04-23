import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load segmented iris image (grayscale)
image_path = "data/segmented/iris_segmented.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("[ERROR] Segmented iris image not found!")
    exit()

# -----------------------------
# Feature Extraction (Histogram)
# -----------------------------
hist = cv2.calcHist([image], [0], None, [256], [0, 256])

# Normalize histogram (important for comparison later)
hist = cv2.normalize(hist, hist).flatten()

# -----------------------------
# Print Info
# -----------------------------
print("[INFO] Feature vector size:", hist.shape)

print("\n[INFO] First 10 feature values:")
print(hist[:10])

# -----------------------------
# Show Image
# -----------------------------
cv2.imshow("Segmented Iris", image)

# -----------------------------
# Plot Histogram
# -----------------------------
plt.plot(hist)
plt.title("Iris Histogram Features")
plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Frequency")
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()