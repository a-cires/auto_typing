
import cv2
import numpy as np

class Phase1KeyboardLocalization:
    def __init__(self, camera_matrix=None):
        self.camera_matrix = camera_matrix  # Optional: for future projection/depth work

    def detect_features(self, image):
        # A.1: Keypoints and ORB Features
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)
        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        # A.3: Lowe's ratio test with FLANN based kd-tree
        index_params = dict(algorithm=6,  # FLANN_INDEX_LSH for ORB
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(desc1, desc2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        return good_matches

    def compute_disparity_map(self, imgL, imgR):
        # B.1: Placeholder - stereo matcher setup for disparity
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(imgL, imgR)
        return disparity

    def estimate_depth_map(self, disparity):
        # B.2: Convert disparity to depth (placeholder formula)
        # Requires known baseline and focal length
        focal_length = 1.0  # placeholder
        baseline = 1.0      # placeholder
        depth_map = (focal_length * baseline) / (disparity + 1e-6)
        return depth_map

    def detect_text_regions(self, image):
        # C.1: Text Detection using OpenCV's EAST or pytesseract (placeholder)
        # Will use pytesseract for simplicity
        import pytesseract
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        return text_data

    def compute_3d_points_from_text(self, text_boxes, depth_map):
        # C.2: Placeholder to map 2D text positions to 3D
        # Requires camera intrinsics and depth map
        points_3d = []
        for i in range(len(text_boxes['text'])):
            if int(text_boxes['conf'][i]) > 60:
                x, y, w, h = (text_boxes['left'][i], text_boxes['top'][i],
                              text_boxes['width'][i], text_boxes['height'][i])
                cx, cy = x + w // 2, y + h // 2
                z = depth_map[cy, cx]
                X = (cx - self.camera_matrix[0, 2]) * z / self.camera_matrix[0, 0]
                Y = (cy - self.camera_matrix[1, 2]) * z / self.camera_matrix[1, 1]
                points_3d.append([X, Y, z])
        return np.array(points_3d)

    def fit_plane_svm(self, points_3d):
        # D.1: Fit a plane using least squares as placeholder for SVM
        from sklearn.linear_model import RANSACRegressor
        X = points_3d[:, :2]
        y = points_3d[:, 2]
        model = RANSACRegressor().fit(X, y)
        return model

    def compute_keyboard_pose(self, plane_model):
        # D.2: Compute normal vector and camera-to-keyboard transformation
        # Placeholder: return normal and dummy transformation
        coef = plane_model.estimator_.coef_
        normal = np.array([-coef[0], -coef[1], 1.0])
        normal = normal / np.linalg.norm(normal)
        translation = np.array([0, 0, 0])  # To be updated
        return normal, translation
