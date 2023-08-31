import numpy as np
from surfaces.surface import Surface
from intersection import Intersection


class Sphere(Surface):
    def __init__(self, position, radius, material_index):
        super(Sphere, self).__init__(material_index)
        self.position = np.array(position, dtype="float")
        self.radius = radius

    def get_intersection_with_ray(self, ray):
        ray_to_center = ray.origin - self.position
        a = 1
        b = 2 * (ray.v @ ray_to_center)
        c = (ray_to_center @ ray_to_center) - self.radius ** 2
        disc = b ** 2 - 4 * a * c
        if disc <= 0:
            return None
        disc_sqrt = np.sqrt(disc)
        t1 = (-b + disc_sqrt) / 2 * a
        t2 = (-b - disc_sqrt) / 2 * a
        if t1 < 0 and t2 < 0:
            return None
        t1 = t1 if t1 >= 0 else float('inf')
        t2 = t2 if t2 >= 0 else float('inf')
        t = min(t1, t2)
        return Intersection(self, ray, t)

    @staticmethod
    def t_comp(t1, t2):
        if t1 < 0:
            if t2 < 0:
                return None
            return t2
        if t2 < 0:
            return t1
        return min(t1, t2)

    def get_intersection_with_rays(self, rays):
        if len(rays) == 1:
            return [self.get_intersection_with_ray(rays[0])]
        ray_origins = [ray.origin for ray in rays]
        ray_directions = [ray.v for ray in rays]
        rays_to_center = ray_origins - self.position
        b_values = 2 * np.einsum('ij, ij -> i', ray_directions, rays_to_center)
        c_values = np.einsum('ij, ij -> i', rays_to_center, rays_to_center) - self.radius ** 2
        disc_values = b_values ** 2 - 4 * c_values
        illegals = disc_values < 0
        disc_values[illegals] = 0
        disc_sqrt_values = disc_values ** 0.5
        t1_values = (-b_values + disc_sqrt_values) / 2
        t2_values = (-b_values - disc_sqrt_values) / 2
        t = [Sphere.t_comp(t1_values[i], t2_values[i]) for i in range(len(t1_values))]
        return [Intersection(self, rays[i], t[i]) if (not illegals[i]) and t[i] is not None else None for i in range(len(rays))]

    def get_normal(self, point):
        N = point - self.position
        return N / np.linalg.norm(N)
