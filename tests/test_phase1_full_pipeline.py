
import cv2
import numpy as np
import argparse
from pathlib import Path
from auto_typing.utils.config import load_config, ROOT_DIR
from auto_typing.phase1.localizer import Phase1KeyboardLocalization
from datetime import datetime

def run_phase1_pipeline(config_path, img1_path, img2_path, translation_m):
    config = load_config(config_path)
    results_dir = ROOT_DIR / config['paths']['results_dir']
    results_dir.mkdir(parents=True, exist_ok=True)

    img1 = cv2.imread(str(img1_path), cv2.IMREAD_COLOR)
    img2 = cv2.imread(str(img2_path), cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Missing input image(s): {img1_path}, {img2_path}")

    localizer = Phase1KeyboardLocalization(config)

    # Step 1: Estimate depth from optical flow
    pts_2d, depths = localizer.estimate_depth_from_flow(img1, img2, translation_m)
    if pts_2d.size == 0:
        print("❌ No valid optical flow correspondences.")
        return

    # Step 2: Build a sparse depth map
    depth_map = np.zeros(img1.shape[:2], dtype=np.float32)
    for (x, y), z in zip(pts_2d, depths):
        x, y = int(round(x)), int(round(y))
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            depth_map[y, x] = z

    # Step 3: Detect text regions
    text_boxes = localizer.detect_text_regions(img1)

    # Step 4: Compute 3D points from text regions
    points_3d = localizer.compute_3d_points_from_text(text_boxes, depth_map)
    if points_3d.size == 0:
        print("❌ No confident text points with valid depth.")
        return

    # Step 5: Fit plane to the 3D points
    model = localizer.fit_plane_svm(points_3d)

    # Step 6: Compute pose
    normal, translation = localizer.compute_keyboard_pose(model)
    print(f"✅ Keyboard normal vector: {normal}")
    print(f"📍 Translation (stub): {translation}")

    # Optional result visualization
    img_out = img1.copy()
    for pt in points_3d:
        u = int(pt[0] * localizer.camera_matrix[0, 0] / pt[2] + localizer.camera_matrix[0, 2])
        v = int(pt[1] * localizer.camera_matrix[1, 1] / pt[2] + localizer.camera_matrix[1, 2])
        if 0 <= u < img_out.shape[1] and 0 <= v < img_out.shape[0]:
            cv2.circle(img_out, (u, v), 3, (255, 0, 0), -1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = results_dir / f"phase1_output_{ts}.jpg"
    cv2.imwrite(str(result_path), img_out)
    print(f"🖼 Saved result to {result_path}")

    if config['debug']['show_matches']:
        cv2.imshow("Phase 1 Output", img_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 1 Full Pipeline Test')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--image1', type=str, default='captures/frame003.jpg')
    parser.add_argument('--image2', type=str, default='captures/frame004.jpg')
    parser.add_argument('--translation', type=float, default=0.01, help='Translation in meters')
    args = parser.parse_args()

    run_phase1_pipeline(Path(args.config), ROOT_DIR / args.image1, ROOT_DIR / args.image2, args.translation)
