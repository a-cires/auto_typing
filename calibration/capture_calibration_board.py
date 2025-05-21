
import cv2
import argparse
from pathlib import Path
from auto_typing.utils.config import load_config, ROOT_DIR

def get_next_filename(calib_dir):
    existing = sorted([f.name for f in calib_dir.glob('calib*.jpg')])
    index = 1
    if existing:
        last = existing[-1]
        digits = ''.join(filter(str.isdigit, last))
        index = int(digits) + 1 if digits else 1
    return calib_dir / f"calib{index:03d}.jpg"

def capture_calibration_images(config):
    calib_dir = ROOT_DIR / config['calibration']['calibration_image_dir']
    calib_dir.mkdir(parents=True, exist_ok=True)

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
            filename = get_next_filename(calib_dir)
            cv2.imwrite(str(filename), frame)
            print(f"✅ Saved: {filename.name}")

        elif key == 27:  # ESC
            print("🛑 Capture ended.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description='Capture calibration images using YAML config.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    args = parser.parse_args()

    config = load_config(args.config)
    capture_calibration_images(config)
