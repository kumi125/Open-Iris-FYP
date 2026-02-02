import cv2
import os
import numpy as np

INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/segmented"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for img_name in os.listdir(INPUT_DIR):
    img_path = os.path.join(INPUT_DIR, img_name)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    img_blur = cv2.GaussianBlur(img, (7, 7), 1.5)

    circles = cv2.HoughCircles(
        img_blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=100,
        param2=30,
        minRadius=40,
        maxRadius=110
    )

    mask = np.zeros_like(img)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        cv2.circle(mask, (x, y), r, 255, -1)

    segmented = cv2.bitwise_and(img, img, mask=mask)

    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), segmented)

print("Iris segmentation completed.")
