
import cv2
import numpy as np
from phase1_keyboard_localization import Phase1KeyboardLocalization

# Placeholder intrinsics
K = np.array([[600, 0, 320],
              [0, 600, 240],
              [0,   0,   1]])

# Load images
img1 = cv2.imread('frame1.jpg', cv2.IMREAD_COLOR)  # Replace with actual image
img2 = cv2.imread('frame2.jpg', cv2.IMREAD_COLOR)  # Replace with actual image

if img1 is None or img2 is None:
    raise ValueError("Check image paths: frame1.jpg and frame2.jpg not found.")

# Initialize algorithm
localizer = Phase1KeyboardLocalization(camera_matrix=K)

# Step 1: Detect features
kp1, des1 = localizer.detect_features(img1)
kp2, des2 = localizer.detect_features(img2)

# Step 2: Match features
matches = localizer.match_features(des1, des2)

# Draw matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Output
print(f"Detected {len(kp1)} and {len(kp2)} keypoints in each image.")
print(f"Good matches found: {len(matches)}")

# Show result
cv2.imshow("Matches", img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
