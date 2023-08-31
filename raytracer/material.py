import numpy as np


class Material:
    def __init__(self, diffuse_color, specular_color, reflection_color, shininess, transparency):
        self.diffuse_color = np.array(diffuse_color, dtype='float')
        self.specular_color = np.array(specular_color, dtype='float')
        self.reflection_color = np.array(reflection_color, dtype='float')
        self.shininess = shininess
        self.transparency = transparency
