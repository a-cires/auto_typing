
import cv2
import numpy as np
import pytesseract
from pathlib import Path
from auto_typing.utils.config import ROOT_DIR

class Phase1KeyboardLocalization:
    def __init__(self, config):
        self.config = config

        if config.get('calibration', {}).get('load_from_file', False):
            calib_file = ROOT_DIR / config['calibration']['output_file']
            if not calib_file.exists():
                raise FileNotFoundError(f"Camera intrinsics file not found at: {calib_file}")
            data = np.load(calib_file)
            self.camera_matrix = data['camera_matrix']
        else:
            f = config['depth']['focal_length']
            self.camera_matrix = np.array([[f, 0, 320],
                                           [0, f, 240],
                                           [0, 0, 1]])

    def detect_features(self, image):
        orb = cv2.ORB_create(
            nfeatures=self.config['orb']['n_features'],
            scaleFactor=self.config['orb']['scale_factor'],
            nlevels=self.config['orb']['n_levels']
        )
        keypoints, descriptors = orb.detectAndCompute(image, None)
        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        index_params = self.config['flann']['index_params']
        search_params = self.config['flann']['search_params']
        ratio_thresh = self.config['flann']['ratio_test_threshold']

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)
        good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
        return good_matches

    def compute_disparity_map(self, imgL, imgR):
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(imgL, imgR)
        return disparity

    def estimate_depth_map(self, disparity):
        focal_length = self.config['depth']['focal_length']
        baseline = self.config['depth']['baseline']
        depth_map = (focal_length * baseline) / (disparity + 1e-6)
        return depth_map

    def detect_text_regions(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        return text_data

    def compute_3d_points_from_text(self, text_boxes, depth_map):
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
        return np.array(points_3d)

    def fit_plane_svm(self, points_3d):
        from sklearn.linear_model import RANSACRegressor
        X = points_3d[:, :2]
        y = points_3d[:, 2]
        model = RANSACRegressor().fit(X, y)
        return model

    def compute_keyboard_pose(self, plane_model):
        coef = plane_model.estimator_.coef_
        normal = np.array([-coef[0], -coef[1], 1.0])
        normal = normal / np.linalg.norm(normal)
        translation = np.array([0, 0, 0])  # To be updated
        return normal, translation
