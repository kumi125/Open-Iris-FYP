import os
import cv2

# Paths
RAW_DATASET_PATH = "data/raw"
PROCESSED_DATASET_PATH = "data/processed"

# Create processed folder if it doesn't exist
os.makedirs(PROCESSED_DATASET_PATH, exist_ok=True)

# Image size for standardization
IMAGE_SIZE = (224, 224)

processed_count = 0

for root, dirs, files in os.walk(RAW_DATASET_PATH):
    for file in files:
        if file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            raw_image_path = os.path.join(root, file)

            # Read image
            image = cv2.imread(raw_image_path)

            if image is None:
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Resize image
            resized = cv2.resize(gray, IMAGE_SIZE)

            # Noise reduction
            blurred = cv2.GaussianBlur(resized, (5, 5), 0)

            # Save processed image
            save_path = os.path.join(
                PROCESSED_DATASET_PATH,
                f"processed_{processed_count}.png"
            )

            cv2.imwrite(save_path, blurred)
            processed_count += 1

print(f"[INFO] Preprocessing completed. Total images processed: {processed_count}")
