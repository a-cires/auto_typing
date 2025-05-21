
import cv2
import os
from pathlib import Path

# Define the project root and target directory
ROOT_DIR = Path(__file__).resolve().parent.parent
CALIB_DIR = ROOT_DIR / 'calibration_images'
CALIB_DIR.mkdir(parents=True, exist_ok=True)

def get_next_filename():
    existing = sorted([f.name for f in CALIB_DIR.glob('calib*.jpg')])
    index = 1
    if existing:
        last = existing[-1]
        index = int(''.join(filter(str.isdigit, last))) + 1
    return CALIB_DIR / f"calib{index:03d}.jpg"

def capture_calibration_images():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open camera.")
        return

    print("📷 Camera open. Press SPACE to capture a calibration image. ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        cv2.imshow("Calibration Capture", frame)
        key = cv2.waitKey(1)

        if key == 32:  # SPACE
            filename = get_next_filename()
            cv2.imwrite(str(filename), frame)
            print(f"✅ Saved: {filename.name}")

        elif key == 27:  # ESC
            print("🛑 Capture ended.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_calibration_images()
