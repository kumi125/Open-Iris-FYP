import cv2
import os
import glob

def preprocess_latest():

    capture_folder = "data/captured"
    processed_folder = "data/processed"
    os.makedirs(processed_folder, exist_ok=True)

    # Latest captured image
    list_of_files = glob.glob(os.path.join(capture_folder, "*.jpg"))

    if len(list_of_files) == 0:
        print("[ERROR] No captured images found!")
        return None

    latest_file = max(list_of_files, key=os.path.getctime)
    image = cv2.imread(latest_file)

    if image is None:
        print("[ERROR] Failed to load image!")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improved eye detection
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=8,
        minSize=(80, 80)
    )

    if len(eyes) == 0:
        print("[ERROR] No eyes detected!")
        return None

    # Sort eyes by size (pick largest)
    eyes = sorted(eyes, key=lambda x: x[2]*x[3], reverse=True)
    (x, y, w, h) = eyes[0]

    eye_roi = gray[y:y+h, x:x+w]

    # Resize + blur
    resized = cv2.resize(eye_roi, (224, 224))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)

    # Save (no overwrite)
    existing_files = [f for f in os.listdir(processed_folder) if f.endswith(".png")]
    count = len(existing_files)

    filename = f"processed_{count}.png"
    processed_path = os.path.join(processed_folder, filename)

    cv2.imwrite(processed_path, blurred)

    print(f"[INFO] Preprocessed image saved as: {processed_path}")

    # Display (optional)
    if __name__ == "__main__":
        cv2.imshow("Detected Eye Region", eye_roi)
        cv2.imshow("Preprocessed Eye", blurred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return processed_path