Imports:
import numpy as np

Setup:
translation = CameraTranslation()

Running:
X, Y, Z = translation.image_to_camera_coords(pixel_x, pixel_y, pixel_depth, K)