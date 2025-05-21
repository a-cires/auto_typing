import cv2
import numpy as np
import glob
import argparse
from pathlib import Path
from auto_typing.utils.config import load_config, ROOT_DIR

def calibrate_camera(config):
    pattern_size = tuple(config['calibration']['pattern_size'])
    square_size = config['calibration']['square_size']
    image_dir = ROOT_DIR / config['calibration']['calibration_image_dir']
    output_file = ROOT_DIR / config['calibration']['output_file']

    # Prepare object points
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

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
        print(f"🔽 Calibration saved to: {output_file}")
    else:
        print("❌ Calibration failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Camera calibration with YAML config.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    args = parser.parse_args()

    config = load_config(args.config)
    calibrate_camera(config)
