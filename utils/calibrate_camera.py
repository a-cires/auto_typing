
import cv2
import numpy as np
import glob
import yaml
import argparse
from pathlib import Path

# Define project root dynamically
ROOT_DIR = Path(__file__).resolve().parent.parent

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def calibrate_camera(config):
    pattern_size = tuple(config['calibration']['pattern_size'])
    square_size = config['calibration']['square_size']
    image_dir = ROOT_DIR / config['calibration']['calibration_image_dir']
    output_file = ROOT_DIR / config['calibration']['output_file']

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Scale by actual square size in meters

    objpoints = []
    imgpoints = []

    images = glob.glob(str(image_dir / '*.jpg'))

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)
            cv2.imshow('Checkerboard Detection', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    if not objpoints:
        print("❌ No valid checkerboard detections found.")
        return

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret:
        print("✅ Calibration successful.")
        print("Camera matrix:")
        print(mtx)
        np.savez(output_file, camera_matrix=mtx, dist_coeffs=dist)
        print(f"🔽 Saved to '{output_file}'")
    else:
        print("❌ Calibration failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Camera calibration with YAML config.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path

    config = load_config(config_path)
    calibrate_camera(config)
