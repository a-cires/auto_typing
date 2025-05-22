import sys
import cv2
import numpy as np
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from auto_typing.phase1.localizer import Phase1KeyboardLocalization
from auto_typing.utils.config import load_config, ROOT_DIR
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Camera intrinsics loaded from file
camera_matrix = np.array([
    [928.0659487443656, 0.0, 663.3936029971047],
    [0.0, 928.8286633219413, 346.4757880013423],
    [0.0, 0.0, 1.0]
])
dist_coeffs = np.array([[0.06904180960211706, -0.10002270113639813, 0.002143990781784916, 0.00044861984450294016, 0.01847125218557232]])

key_positions = {
    'z': (55, 32.5, 30.665), 'x': (74, 32.5, 30.665), 'c': (92, 32.5, 30.665), 'v': (110, 32.5, 30.665),
    'b': (131, 32.5, 30.665), 'n': (150, 32.5, 30.665), 'm': (169, 32.5, 30.665), 'a': (45, 51.5, 31.115),
    's': (64, 51.5, 31.115), 'd': (84, 51.5, 31.115), 'f': (102, 51.5, 31.115), 'g': (121, 51.5, 31.115),
    'h': (141, 51.5, 31.115), 'j': (159, 51.5, 31.115), 'k': (178, 51.5, 31.115), 'l': (197, 51.5, 31.115),
    'r': (98, 70.5, 33.345), 't': (117, 70.5, 33.345), 'y': (136, 70.5, 33.345), 'u': (154, 70.5, 33.345),
    'i': (173, 70.5, 33.345), 'o': (193, 70.5, 33.345), 'p': (262, 70.5, 33.345)
}

def preprocess_keyboard_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
    _, thresh = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)
    clean_kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, clean_kernel)
    return cleaned

def extract_key_contours(thresh):
    height = thresh.shape[0]
    mask = np.zeros_like(thresh)
    mask[int(height * 0.25):, :] = 255
    masked = cv2.bitwise_and(thresh, mask)
    contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    key_contours = [cnt for cnt in contours if 100 < cv2.contourArea(cnt) < 10000]
    return key_contours

def recognize_keys(image, key_contours, tolerance=5):
    results = []
    for cnt in key_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x = max(0, x - tolerance)
        y = max(0, y - tolerance)
        w += 2 * tolerance
        h += 2 * tolerance
        key_roi = image[y:y+h, x:x+w]
        if key_roi.shape[0] < 20 or key_roi.shape[1] < 20:
            key_roi = cv2.resize(key_roi, (0, 0), fx=2, fy=2)
        char = pytesseract.image_to_string(key_roi, config='--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789').strip()
        if char and not char.isdigit() and char.upper() != 'P':
            results.append(((x, y, w, h), char))
    return results


def compute_pixel_to_mm_scale(detections, key_positions):
    pixel_xs, pixel_ys, real_xs, real_ys = [], [], [], []
    for (x, y, w, h), char in detections:
        key = char.lower()
        if key in key_positions:
            x_mm, y_mm, _ = key_positions[key]
            pixel_xs.append(x + w // 2)
            pixel_ys.append(y + h // 2)
            real_xs.append(x_mm)
            real_ys.append(y_mm)
    dxs, dx_mm, dys, dy_mm = [], [], [], []
    for i in range(len(pixel_xs)):
        for j in range(i + 1, len(pixel_xs)):
            dx_real = abs(real_xs[i] - real_xs[j])
            dy_real = abs(real_ys[i] - real_ys[j])
            if dx_real > 0:
                dxs.append(abs(pixel_xs[i] - pixel_xs[j]))
                dx_mm.append(dx_real)
            if dy_real > 0:
                dys.append(abs(pixel_ys[i] - pixel_ys[j]))
                dy_mm.append(dy_real)
    scale_x = np.mean(np.array(dxs) / np.array(dx_mm)) if dx_mm else float('nan')
    scale_y = np.mean(np.array(dys) / np.array(dy_mm)) if dy_mm else float('nan')
    return scale_x, scale_y

def estimate_missing_key_positions(image, detections, key_positions, scale_x, scale_y):
    height, width = image.shape[:2]
    pixel_map = {char.lower(): (x + w // 2, y + h // 2) for (x, y, w, h), char in detections}
    real_map = {k: (pos[0], pos[1]) for k, pos in key_positions.items()}
    estimated = {}
    for key in set(key_positions) - set(pixel_map):
        if key not in real_map:
            continue
        x_mm_target, y_mm_target = real_map[key]
        px_total, py_total = 0, 0
        count = 0
        for detected_key, (px, py) in pixel_map.items():
            if detected_key not in real_map:
                continue
            x_mm_ref, y_mm_ref = real_map[detected_key]
            dx_mm = x_mm_target - x_mm_ref
            dy_mm = -(y_mm_target - y_mm_ref)  # Invert Y to match image coords (top-left origin)
            est_x = px + dx_mm * scale_x
            est_y = py + dy_mm * scale_y
            px_total += est_x
            py_total += est_y
            count += 1
        if count > 0:
            avg_x = int(px_total / count)
            avg_y = int(py_total / count)
            estimated[key] = (avg_x, avg_y)
    for k, (x, y) in estimated.items():
        cv2.circle(image, (x, y), 6, (255, 0, 0), 2)
        cv2.putText(image, k.upper(), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image

def main1():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--image', type=str, default='captures/frame003.jpg')
    args = parser.parse_args()

    config = load_config(args.config)
    image = cv2.imread(str(ROOT_DIR / args.image))
    if image is None:
        raise ValueError("Image not found.")

    # Undistort the image before any processing
    image = cv2.undistort(image, camera_matrix, dist_coeffs)

    image = cv2.rotate(image, cv2.ROTATE_180)
    image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = preprocess_keyboard_image(image)
    contours = extract_key_contours(thresh)
    detections = recognize_keys(gray, contours)

    scale_x, scale_y = compute_pixel_to_mm_scale(detections, key_positions)
    print(f"Scale X: {scale_x:.2f} px/mm, Scale Y: {scale_y:.2f} px/mm")

    vis = image.copy()
    for (x, y, w, h), char in detections:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis, char, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    vis = estimate_missing_key_positions(vis, detections, key_positions, scale_x, scale_y)

    print(f"Detected {len(detections)} keys.")
    cv2.imshow("Detected + Estimated Keys", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main1()
