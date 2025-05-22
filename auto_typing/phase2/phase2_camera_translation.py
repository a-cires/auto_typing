import numpy as np

class CameraTranslation:
    def __init__(self):
        return
    
    def image_to_camera_coords(u, v, depth, K):
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        Xc = (u - cx) * depth / fx
        Yc = (v - cy) * depth / fy
        Zc = depth

        return Xc, Yc, Zc