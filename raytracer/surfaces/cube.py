from surfaces.surface import Surface
import numpy as np
from intersection import Intersection
from surfaces.infinite_plane import InfinitePlane

EPSILON = 0.000001


class Cube(Surface):
    def __init__(self, position, scale, material_index):
        super(Cube, self).__init__(material_index)
        self.position = np.array(position, dtype="float")
        self.scale = scale
        self.min = self.position - self.scale / 2
        self.max = self.position + self.scale / 2
        self.planes = [
            InfinitePlane(np.array([1, 0, 0], dtype="float"), self.position[0] + self.scale / 2, self.material_index),
            InfinitePlane(np.array([-1, 0, 0], dtype="float"), -(self.position[0] - self.scale / 2),
                          self.material_index),
            InfinitePlane(np.array([0, 1, 0], dtype="float"), self.position[1] + self.scale / 2, self.material_index),
            InfinitePlane(np.array([0, -1, 0], dtype="float"), -(self.position[1] - self.scale / 2),
                          self.material_index),
            InfinitePlane(np.array([0, 0, 1], dtype="float"), self.position[2] + self.scale / 2, self.material_index),
            InfinitePlane(np.array([0, 0, -1], dtype="float"), -(self.position[2] - self.scale / 2),
                          self.material_index)]

    def get_intersection_with_ray(self, ray):

        with np.errstate(divide='ignore'):  # division by zero is ok.
            t_min = (self.min - ray.origin) / ray.v
            t_max = (self.max - ray.origin) / ray.v
            t_enter = np.max(np.minimum(t_min, t_max))
            t_exit = np.min(np.maximum(t_min, t_max))
            if t_enter > t_exit:
                return None
            return Intersection(self, ray, t_enter)

    def get_intersection_with_rays(self, rays):
        rays_v = np.array([ray.v for ray in rays])
        rays_origin = np.array([ray.origin for ray in rays])
        with np.errstate(divide='ignore'):  # division by zero is ok.
            t_min = (self.min - rays_origin) / rays_v
            t_max = (self.max - rays_origin) / rays_v
            t_enter = np.max(np.minimum(t_min, t_max), axis=1)
            t_exit = np.min(np.maximum(t_min, t_max), axis=1)
            return [Intersection(self, rays[i], t_enter[i]) if t_exit[i] >= t_enter[i] else None for i in
                    range(len(rays))]

    def in_cube(self, point):
        for i in range(3):
            if not (self.position[i] - self.scale / 2 <= point[i] <= self.position[i] + self.scale / 2):
                return False
        return True

    def get_normal(self, point):
        for plane in self.planes:
            if abs(np.dot(plane.normal, point) - plane.offset) < EPSILON:
                return plane.normal
