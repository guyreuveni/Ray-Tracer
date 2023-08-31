class Intersection:
    def __init__(self, surface, ray, t):
        self.ray = ray
        self.surface = surface
        self.hit_point = ray.origin + t * ray.v
        self.t = t

