import cv2

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
        cv2.imwrite("iris_sample.jpg", frame)
        print("Image saved as iris_sample.jpg")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
