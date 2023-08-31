import numpy as np


class Ray:
    def __init__(self, *params):
        self.v = None
        self.origin = None
        if len(params) == 5:
            self.init_from_camera(params[0], params[1], params[2], params[3], params[4])
        if len(params) == 2:
            self.init_directly(params[0], params[1])

    def init_from_camera(self, camera, i, j, rx, ry):
        p = camera.screen_center + (j - rx // 2) * camera.ratio * camera.right - (
                    i - ry // 2) * camera.ratio * camera.up_vector
        self.v = p - camera.position
        self.v = self.v / np.linalg.norm(self.v)
        self.origin = camera.position

    def init_directly(self, origin, v):
        self.origin = origin
        self.v = v / np.linalg.norm(v)
