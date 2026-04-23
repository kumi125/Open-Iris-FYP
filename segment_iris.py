import cv2
import os
import numpy as np
import glob

def segment_iris(save_folder):

    os.makedirs(save_folder, exist_ok=True)

    # Load latest processed image
    list_of_files = glob.glob("data/processed/*.png")

    if len(list_of_files) == 0:
        print("[ERROR] No processed images found!")
        return None

    latest_file = max(list_of_files, key=os.path.getctime)

    image = cv2.imread(latest_file, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("[ERROR] Cannot load image!")
        return None

    print(f"[INFO] Using processed image: {latest_file}")

    equalized = cv2.equalizeHist(image)

    circles = cv2.HoughCircles(
        equalized,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=100,
        param2=20,
        minRadius=30,
        maxRadius=90
    )

    if circles is None:
        print("[INFO] No iris detected")
        return None

    circles = np.round(circles[0, :]).astype("int")

    # 🔴 IMPORTANT: only take BEST circle
    x, y, r = circles[0]

    iris = image[y-r:y+r, x-r:x+r]

    if iris.size == 0:
        print("[ERROR] Empty iris crop")
        return None

    iris = cv2.resize(iris, (120, 120))

    # save file
    existing_files = [f for f in os.listdir(save_folder) if f.endswith(".jpg")]
    count = len(existing_files)

    filename = f"iris_segmented_{count}.jpg"
    save_path = os.path.join(save_folder, filename)

    cv2.imwrite(save_path, iris)

    print(f"[INFO] Iris saved: {save_path}")

    if __name__ == "__main__":
        cv2.imshow("Segmented Iris", iris)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return save_path