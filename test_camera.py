import os
import re
import cv2


def capture_frame(save_path="data/captured"):
    """Opens Index 0 (Built-in Laptop Camera) with default backend and sets capture resolution,

    while resizing the display window so it fits on screen.
    """
    os.makedirs(save_path, exist_ok=True)

    # 1. Switch index back to 0 (Built-in Laptop Webcam)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # 2. Set Resolution (1280x720 is optimal for built-in webcams, but attempts 1080p first)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Check if Index 0 opened validly
    ret, test_frame = cap.read()
    if (
        not cap.isOpened()
        or not ret
        or test_frame is None
        or test_frame.size == 0
    ):
        print(
            "[WARNING] Index 0 DSHOW failed. Trying default backend on Index 0..."
        )
        cap.release()
        cap = cv2.VideoCapture(0)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"[INFO] Connected to Built-in Laptop Camera (Index 0) at {actual_w}x{actual_h}!"
    )
    print("[INFO] Camera started. Look into the camera.")
    print("       Press 's' to capture/save image.")
    print("       Press 'q' to quit.")

    captured_file = None
    window_name = "Webcam Eye Capture - Open-Iris"

    # Enable Resizable Window & Set Window Display Size
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        window_name, 960, 540
    )  # Fits cleanly on any laptop display

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            continue

        height, width, _ = frame.shape

        # Define ROI box centered in raw frame space (shifted slightly upwards for eyes)
        box_w = int(width * 0.35)
        box_h = int(height * 0.30)

        center_x = width // 2
        center_y = int(height * 0.35)  # Shifted up to align with eye level

        x1 = max(0, center_x - (box_w // 2))
        y1 = max(0, center_y - (box_h // 2))
        x2 = min(width - 1, center_x + (box_w // 2))
        y2 = min(height - 1, center_y + (box_h // 2))

        display_frame = frame.copy()
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Shift text higher above top boundary
        label_y = max(35, y1 - 25)
        cv2.putText(
            display_frame,
            "Align Eye Inside Box",
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            existing_files = [
                f
                for f in os.listdir(save_path)
                if f.startswith("captured_") and f.endswith(".jpg")
            ]
            indices = [
                int(re.findall(r"\d+", f)[0])
                for f in existing_files
                if re.findall(r"\d+", f)
            ]

            next_index = max(indices) + 1 if indices else 1
            filename = f"captured_{next_index}.jpg"
            full_path = os.path.join(save_path, filename)

            # Saves the full unscaled high-res frame
            cv2.imwrite(full_path, frame)
            print(f"[INFO] Image saved as {full_path}")
            captured_file = full_path
            break

        elif key == ord("q"):
            print("[INFO] Capture canceled by user.")
            break

    cap.release()
    cv2.destroyWindow(window_name)
    cv2.waitKey(1)
    return captured_file


if __name__ == "__main__":
    print("[INFO] Running standalone camera test...")
    capture_frame()