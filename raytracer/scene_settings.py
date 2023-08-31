import numpy as np


class SceneSettings:
    def __init__(self, background_color, root_number_shadow_rays, max_recursions):
        self.background_color = np.array(background_color, dtype="float")
        self.root_number_shadow_rays = int(root_number_shadow_rays)
        self.max_recursions = max_recursions
