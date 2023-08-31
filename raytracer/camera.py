import numpy as np


class Camera:
    def __init__(self, position, look_at, up_vector, screen_distance, screen_width):
        self.position = np.array(position, dtype="float")
        self.look_at = np.array(look_at, dtype="float")
        self.up_vector = np.array(up_vector, dtype="float")
        self.screen_distance = screen_distance
        self.screen_width = screen_width
        self.direction = (self.look_at - self.position)
        self.direction = self.direction / np.linalg.norm(self.direction)
        self.right = np.cross(self.direction, self.up_vector)
        self.right = self.right / np.linalg.norm(self.right)
        self.up_vector = np.cross(self.right, self.direction)
        self.up_vector = self.up_vector / np.linalg.norm(self.up_vector)
        self.screen_center = self.position + self.screen_distance * self.direction
        self.ratio = -1

    def calc_and_set_ratio(self, result_img_width):
        self.ratio = self.screen_width / result_img_width
