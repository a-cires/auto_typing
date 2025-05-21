
import cv2
import os
from pathlib import Path
from auto_typing.utils.config import load_config, ROOT_DIR
import argparse

def get_next_filename(capture_dir, base='frame', ext='jpg'):
    existing = sorted([f.name for f in capture_dir.glob(f'{base}*.{ext}')])
    index = 1
    if existing:
        last = existing[-1]
        digits = ''.join(filter(str.isdigit, last))
        index = int(digits) + 1 if digits else 1
    return capture_dir / f"{base}{index:03d}.{ext}"

def capture_frames(config):
    capture_dir = ROOT_DIR / config['paths']['capture_dir']
    capture_dir.mkdir(parents=True, exist_ok=True)

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
            filename = get_next_filename(capture_dir)
            cv2.imwrite(str(filename), frame)
            print(f"📸 Captured {filename.name}")
            captured += 1
            if captured < 2:
                print("➡️ Move the camera slightly and press SPACE again.")

        elif key == 27:  # ESC to quit
            print("❌ Capture cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Capture stereo or sequence frames.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    args = parser.parse_args()

    config = load_config(args.config)
    capture_frames(config)
