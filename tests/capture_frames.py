
import cv2
import time
import os

def capture_frames():
    save_dir = "../captures"
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open camera.")
        return

    print("✅ Camera opened. Press SPACE to capture the first frame.")
    captured = 0
    while captured < 2:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break
        cv2.imshow("Live Feed", frame)

        key = cv2.waitKey(1)
        if key == 32:  # SPACE key
            filename = os.path.join(save_dir, f"frame{captured + 1}.jpg")
            cv2.imwrite(filename, frame)
            print(f"📸 Captured {filename}")
            captured += 1
            if captured < 2:
                print("➡️ Move the camera slightly and press SPACE again.")

        elif key == 27:  # ESC to quit
            print("❌ Capture cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_frames()
