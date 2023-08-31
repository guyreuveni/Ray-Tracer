from surfaces.surface import Surface
import numpy as np
from intersection import Intersection

EPSILON = 0.000001


class InfinitePlane(Surface):
    def __init__(self, normal, offset, material_index):
        super(InfinitePlane, self).__init__(material_index)
        self.normal = np.array(normal, dtype="float")
        self.normal /= np.linalg.norm(self.normal)
        self.offset = offset

    def get_intersection_with_ray(self, ray):
        dot_prod = self.normal @ ray.v
        if abs(dot_prod) < EPSILON:
            # ray is parallel to the plane
            return None
        t = ((self.offset * self.normal - ray.origin) @ self.normal) / dot_prod

        if t < 0:
            # plane is behind the ray origin
            return None

        return Intersection(self, ray, t)

    def get_intersection_with_rays(self, rays):
        if len(rays) == 1:
            return [self.get_intersection_with_ray(rays[0])]
        with np.errstate(divide='ignore'):  # division by zero is ok.
            rays_v = np.array([ray.v for ray in rays])
            rays_origin = np.array([ray.origin for ray in rays])

            dot_prods = np.einsum('ij,j->i', rays_v, self.normal)
            t_values = (np.einsum('ij,j->i', (self.offset * self.normal - rays_origin), self.normal)) / dot_prods

            boolean_array = np.logical_or(abs(dot_prods) < EPSILON, t_values < 0)
            return [Intersection(self, rays[i], t_values[i]) if not boolean_array[i] else None for i in
                    range(len(rays))]

    def get_intersection_with_rays_(self, rays):
        return [self.get_intersection_with_ray(ray) for ray in rays]

    def get_normal(self, point):
        return self.normal
