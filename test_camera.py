import cv2
import os

# Create folder for captured images
capture_folder = "data/captured"
os.makedirs(capture_folder, exist_ok=True)

cap = cv2.VideoCapture(0)

print("Press 's' to save image")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Iris Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # 🔥 Generate unique filename (no overwrite)
        existing_files = [f for f in os.listdir(capture_folder) if f.endswith(".jpg")]
        count = len(existing_files)

        filename = f"captured_{count}.jpg"
        save_path = os.path.join(capture_folder, filename)

        cv2.imwrite(save_path, frame)
        print(f"[INFO] Image saved as {save_path}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()