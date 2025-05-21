
import cv2
import numpy as np
import argparse
from pathlib import Path
from auto_typing.phase1.localizer import Phase1KeyboardLocalization
from auto_typing.utils.config import load_config, ROOT_DIR
from datetime import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 1 Optical Flow Depth Estimation Test Script')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    parser.add_argument('--image1', type=str, default='captures/frame001.jpg', help='Path to first image')
    parser.add_argument('--image2', type=str, default='captures/frame002.jpg', help='Path to second image')
    parser.add_argument('--translation', type=float, default=0.01, help='Translation between frames in meters')
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = ROOT_DIR / config['paths']['results_dir']
    results_dir.mkdir(parents=True, exist_ok=True)

    image1_path = ROOT_DIR / args.image1
    image2_path = ROOT_DIR / args.image2

    img1 = cv2.imread(str(image1_path), cv2.IMREAD_COLOR)
    img2 = cv2.imread(str(image2_path), cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        raise ValueError(f"Check image paths: {image1_path} and {image2_path} not found.")

    localizer = Phase1KeyboardLocalization(config)

    # Step 1: Estimate depth from optical flow
    pts, depths = localizer.estimate_depth_from_flow(img1, img2, args.translation)

    # Step 2: Visualize tracked points
    img_vis = img1.copy()
    for (x, y), z in zip(pts, depths):
        cv2.circle(img_vis, (int(x), int(y)), 3, (0, 255, 0), -1)
        cv2.putText(img_vis, f"{z:.2f}m", (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    print(f"✅ Tracked {len(pts)} valid points with estimated depth.")

    # Save and optionally display
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = results_dir / f"flow_depth_{timestamp}.jpg"
    cv2.imwrite(str(result_path), img_vis)
    print(f"💾 Saved result to: {result_path}")

    if config['debug']['show_matches']:
        cv2.imshow("Depth from Flow", img_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
