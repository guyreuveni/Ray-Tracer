from abc import ABC, abstractmethod
from ray import Ray

class Surface(ABC):
    def __init__(self, material_index):
        self.material_index = material_index

    def get_material(self, materials):
        return materials[self.material_index - 1]

    @abstractmethod
    def get_intersection_with_ray(self, ray):
        pass
    @abstractmethod
    def get_intersection_with_rays(self, rays):
        pass
    @abstractmethod
    def get_normal(self, point):
        pass

    def get_reflected_ray(self, ray, point):
        surf_normal = self.get_normal(point)
        cos_a = ray.v @ surf_normal
        reflection_dir = ray.v - 2 * cos_a * surf_normal
        return Ray(point, reflection_dir)


