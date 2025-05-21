
import cv2
import numpy as np
import argparse
import yaml
from pathlib import Path
from phase1_keyboard_localization import Phase1KeyboardLocalization, load_config

# Define project root
ROOT_DIR = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 1 Test Script')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    parser.add_argument('--image1', type=str, default='captures/frame1.jpg', help='Path to first image')
    parser.add_argument('--image2', type=str, default='captures/frame2.jpg', help='Path to second image')
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path
    config = load_config(config_path)

    image1_path = ROOT_DIR / args.image1
    image2_path = ROOT_DIR / args.image2

    img1 = cv2.imread(str(image1_path), cv2.IMREAD_COLOR)
    img2 = cv2.imread(str(image2_path), cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        raise ValueError(f"Check image paths: {image1_path} and {image2_path} not found.")

    localizer = Phase1KeyboardLocalization(config)

    # Step 1: Detect features
    kp1, des1 = localizer.detect_features(img1)
    kp2, des2 = localizer.detect_features(img2)

    # Step 2: Match features
    matches = localizer.match_features(des1, des2)

    # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Output
    print(f"🔍 Detected {len(kp1)} and {len(kp2)} keypoints in each image.")
    print(f"✅ Good matches found: {len(matches)}")

    if config['debug']['show_matches']:
        cv2.imshow("Matches", img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
