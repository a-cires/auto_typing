
import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
from pathlib import Path
from datetime import datetime
from auto_typing.utils.config import ROOT_DIR

class Phase1KeyboardLocalization:
    def __init__(self, config):
        self.config = config
        self.verbose = config.get('debug', {}).get('verbose_logging', False)
        self.log_dir = ROOT_DIR / config['paths']['log_dir']
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'phase1_log_{timestamp}.txt'
        self.log(f"Initialized Phase1KeyboardLocalization at {timestamp}")

        if config.get('calibration', {}).get('load_from_file', False):
            calib_file = ROOT_DIR / config['calibration']['output_file']
            if not calib_file.exists():
                raise FileNotFoundError(f"Camera intrinsics file not found at: {calib_file}")
            data = np.load(calib_file)
            self.camera_matrix = data['camera_matrix']
            self.log("Loaded camera matrix from file.")
        else:
            f = config['depth']['focal_length']
            self.camera_matrix = np.array([[f, 0, 320],
                                           [0, f, 240],
                                           [0, 0, 1]])
            self.log("Using fallback camera matrix from config.")

    def log(self, message):
        if self.verbose:
            with open(self.log_file, 'a') as f:
                f.write(f"{datetime.now().isoformat()} - {message}\n")

    def detect_features(self, image):
        self.log("Detecting good features to track (Shi-Tomasi)...")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=1000,
            qualityLevel=0.01,
            minDistance=10
        )
        self.log(f"Detected {len(features) if features is not None else 0} trackable points.")
        return features

    def estimate_depth_from_flow(self, img1, img2, translation_m):
        self.log("Estimating depth from optical flow...")
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        pts1 = self.detect_features(img1)
        if pts1 is None:
            self.log("No features to track in image 1.")
            return np.array([]), np.array([])

        pts2, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, pts1, None)

        focal_length = self.camera_matrix[0, 0]
        depths = []
        pts1_filtered = []

        for (p1, p2, s) in zip(pts1, pts2, status):
            if s[0] == 1:
                dx = p2[0][0] - p1[0][0]
                if abs(dx) > 1e-3:
                    Z = (focal_length * translation_m) / dx
                    depths.append(Z)
                    pts1_filtered.append(p1[0])

        self.log(f"Tracked {len(depths)} points with valid depth estimates.")
        return np.array(pts1_filtered), np.array(depths)

    def compute_disparity_map(self, imgL, imgR):
        self.log("Computing disparity map...")
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(imgL, imgR)
        return disparity

    def estimate_depth_map(self, disparity):
        self.log("Estimating depth map from disparity...")
        focal_length = self.config['depth']['focal_length']
        baseline = self.config['depth']['baseline']
        depth_map = (focal_length * baseline) / (disparity + 1e-6)
        return depth_map

    def detect_text_regions(self, image):
        self.log("Detecting keyboard key regions using morphology and OCR...")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
        _, thresh = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)
        clean_kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, clean_kernel)

        # Extract key contours
        height = cleaned.shape[0]
        mask = np.zeros_like(cleaned)
        mask[int(height * 0.25):, :] = 255
        masked = cv2.bitwise_and(cleaned, mask)
        contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        key_contours = [cnt for cnt in contours if 100 < cv2.contourArea(cnt) < 10000]

        # Recognize characters using Tesseract
        results = []
        for cnt in key_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            key_roi = gray[max(y-5,0):y+h+5, max(x-5,0):x+w+5]
            if key_roi.shape[0] < 20 or key_roi.shape[1] < 20:
                key_roi = cv2.resize(key_roi, (0, 0), fx=2, fy=2)
            char = pytesseract.image_to_string(key_roi, config='--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789').strip()
            if char and not char.isdigit() and char.upper() != 'P':
                results.append(((x, y, w, h), char))

        # Format output in Tesseract-style dict
        text_boxes = {'left': [], 'top': [], 'width': [], 'height': [], 'conf': [], 'text': []}
        for (x, y, w, h), char in results:
            text_boxes['left'].append(x)
            text_boxes['top'].append(y)
            text_boxes['width'].append(w)
            text_boxes['height'].append(h)
            text_boxes['conf'].append(95)
            text_boxes['text'].append(char)

        return text_boxes

    def compute_3d_points_from_text(self, text_boxes, depth_map):
        self.log("Computing 3D points from text regions...")
        points_3d = []
        for i in range(len(text_boxes['text'])):
            if int(text_boxes['conf'][i]) > self.config['text']['min_confidence']:
                x, y, w, h = (text_boxes['left'][i], text_boxes['top'][i],
                              text_boxes['width'][i], text_boxes['height'][i])
                cx, cy = x + w // 2, y + h // 2
                z = depth_map[cy, cx]
                X = (cx - self.camera_matrix[0, 2]) * z / self.camera_matrix[0, 0]
                Y = (cy - self.camera_matrix[1, 2]) * z / self.camera_matrix[1, 1]
                points_3d.append([X, Y, z])
        self.log(f"Computed {len(points_3d)} 3D points.")
        return np.array(points_3d)

    def fit_plane_svm(self, points_3d):
        self.log("Fitting plane using RANSAC...")
        if points_3d.shape[0] < 3:
            raise ValueError(f"Need at least 3 points to fit a plane, got {points_3d.shape[0]}")
        from sklearn.linear_model import RANSACRegressor
        X = points_3d[:, :2]
        y = points_3d[:, 2]
        model = RANSACRegressor().fit(X, y)
        return model

    def compute_keyboard_pose(self, plane_model):
        self.log("Computing keyboard plane normal...")
        coef = plane_model.estimator_.coef_
        normal = np.array([-coef[0], -coef[1], 1.0])
        normal = normal / np.linalg.norm(normal)
        translation = np.array([0, 0, 0])  # To be updated
        self.log(f"Keyboard normal: {normal.tolist()}")
        return normal, translation
