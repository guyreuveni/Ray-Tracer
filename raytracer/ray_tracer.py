import argparse
from PIL import Image
from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from ray import Ray
import numpy as np
import time
EPSILON = 0.000001
camera: Camera
scene_settings: SceneSettings
surfaces = []
materials = []
lights = []
width = -1
height = -1
bonus = False
bonus_type = -1


def parse_scene_file(file_path):
    surfaces = []
    lights = []
    materials = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                materials.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                surfaces.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                surfaces.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                surfaces.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                lights.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, surfaces, materials, lights


def find_all_ray_intersections_sorted(ray):
    global surfaces
    intersections = []
    for surface in surfaces:
        intersection = surface.get_intersection_with_ray(ray)
        if intersection is None or intersection.t <= EPSILON:
            continue
        intersections.append(intersection)
    intersections.sort(key=lambda _intersection: _intersection.t)
    return intersections


def find_closest_rays_intersections_batch(rays):
    global surfaces
    min_t_values = np.array([float('inf') for _ in rays])
    closest_intersections = np.array([None for _ in rays])
    for surface in surfaces:
        intersections = np.array(surface.get_intersection_with_rays(rays))
        intersections_t = [intersection.t if intersection is not None else float('inf') for intersection in
                           intersections]
        new_values = intersections_t < min_t_values
        min_t_values = np.minimum(min_t_values, intersections_t)
        closest_intersections[new_values] = intersections[new_values]
    return closest_intersections


def find_closest_ray_intersections(ray):
    global surfaces
    min_t = float('inf')
    closest_intersection = None

    for surface in surfaces:
        intersection = surface.get_intersection_with_ray(ray)
        if intersection is None:
            continue
        if min_t > intersection.t:
            min_t = intersection.t
            closest_intersection = intersection
    return closest_intersection


def get_light_intensity_batch(light, intersection):
    global scene_settings
    global bonus
    plane_normal = intersection.hit_point - light.position
    plane_normal /= np.linalg.norm(plane_normal)
    transform_matrix = xy_to_general_plane(plane_normal, light.position)

    length_unit = light.radius / scene_settings.root_number_shadow_rays
    x_values_vec, y_values_vec = np.meshgrid(range(scene_settings.root_number_shadow_rays),
                                             range(
                                                 scene_settings.root_number_shadow_rays))  # build grid starting points
    x_values_vec = x_values_vec.reshape(-1)
    y_values_vec = y_values_vec.reshape(-1)
    base_xy = np.array([x_values_vec * length_unit - light.radius / 2,  # center grid around 0,0
                        y_values_vec * length_unit - light.radius / 2,
                        np.zeros_like(x_values_vec),
                        np.zeros_like(x_values_vec)])
    offset = np.array(  # select random point inside each grid cell
        [np.random.uniform(0, length_unit, y_values_vec.shape),
         np.random.uniform(0, length_unit, x_values_vec.shape),
         np.zeros_like(x_values_vec),
         np.ones_like(x_values_vec)])  # for translation matrix
    rectangle_points = base_xy + offset

    light_points = (transform_matrix @ rectangle_points)[:3].T

    rays = [Ray(point, intersection.hit_point - point) for point in light_points]
    if not bonus:
        return calc_light_intensity_regular(light, intersection, rays), light.color
    return calc_light_intensity_bonus(light, intersection, rays)


def calc_light_intensity_regular(light, intersection, rays):
    light_hits = find_closest_rays_intersections_batch(rays)
    c = sum(1 for light_hit in light_hits if light_hit is not None and
            np.linalg.norm(intersection.hit_point - light_hit.hit_point) < EPSILON)
    light_intensity = (1 - light.shadow_intensity) + light.shadow_intensity * (
            c / (scene_settings.root_number_shadow_rays ** 2))
    return light_intensity


def calc_light_intensity_bonus(light, intersection, rays):
    light_hits = [find_all_ray_intersections_sorted(ray) for ray in rays]
    light_color = np.array([0, 0, 0], dtype='float')
    c = 0
    for light_intersections in light_hits:
        ray_contribution = 1
        curr_ray_color = np.array(light.color)

        for light_intersection in light_intersections:

            if np.linalg.norm(intersection.hit_point - light_intersection.hit_point) < EPSILON:
                c += ray_contribution
                light_color += curr_ray_color
                break
            curr_ray_color *= light_intersection.surface.get_material(materials).diffuse_color
            ray_contribution *= light_intersection.surface.get_material(materials).transparency
            if ray_contribution == 0:
                break
    light_intensity = (1 - light.shadow_intensity) + light.shadow_intensity * (
            c / (scene_settings.root_number_shadow_rays ** 2))
    if bonus_type == 2:
        return light_intensity, light.color * 0.9 + 0.1 * (light_color / (scene_settings.root_number_shadow_rays ** 2))
    return light_intensity, light.color


def get_light_intensity(light, intersection):  # before batch optimizations
    global scene_settings

    plane_normal = intersection.hit_point - light.position
    plane_normal /= np.linalg.norm(plane_normal)
    transform_matrix = xy_to_general_plane(plane_normal, light.position)
    rectangle_points = []
    length_unit = light.radius / scene_settings.root_number_shadow_rays
    for x in range(scene_settings.root_number_shadow_rays):
        for y in range(scene_settings.root_number_shadow_rays):
            offset = np.array([np.random.uniform(0, length_unit),
                               np.random.uniform(0, length_unit),
                               0,
                               1], dtype="float")
            base_xy = np.array([x * length_unit - light.radius / 2,
                                y * length_unit - light.radius / 2,
                                0,
                                0], dtype="float")
            rectangle_points.append(base_xy + offset)
    rectangle_points = np.array(rectangle_points).T
    light_points = (transform_matrix @ rectangle_points).T
    light_points = light_points[:, :-1]

    rays = [Ray(point, intersection.hit_point - point) for point in light_points]
    light_hits = [find_closest_ray_intersections(ray) for ray in rays]
    c = 0
    for light_hit in light_hits:
        if np.linalg.norm(intersection.hit_point - light_hit.hit_point) < EPSILON:
            c += 1
    light_intensity = (1 - light.shadow_intensity) + light.shadow_intensity * \
                      (c / (scene_settings.root_number_shadow_rays ** 2))

    return light_intensity


def get_reflection_color(intersection, reflection_rec_level):
    global scene_settings
    if reflection_rec_level >= scene_settings.max_recursions:
        return scene_settings.background_color
    reflection_ray = intersection.surface.get_reflected_ray(intersection.ray, intersection.hit_point)
    intersections = find_all_ray_intersections_sorted(reflection_ray)
    reflection_color = get_ray_color(intersections, reflection_rec_level + 1)
    return reflection_color


def get_ray_color(intersections, reflection_rec_level=0):
    global materials
    global scene_settings
    if intersections is None or len(intersections) == 0:
        return scene_settings.background_color
    for i in range(len(intersections)):
        if intersections[i].surface.get_material(materials).transparency == 0:
            intersections = intersections[0:i + 1]
            break

    i = len(intersections) - 1
    bg_color = scene_settings.background_color
    color = None
    while i >= 0:
        transparency = intersections[i].surface.get_material(materials).transparency
        d_s_color = get_diffuse_and_specular_color(intersections[i])
        reflection_color = get_reflection_color(intersections[i], reflection_rec_level) * \
                           intersections[i].surface.get_material(
                               materials).reflection_color

        color = ((1 - transparency) * d_s_color + transparency * bg_color) \
                + reflection_color
        bg_color = color
        i -= 1
    return color


def get_diffuse_and_specular_color(intersection):
    global scene_settings
    global materials
    global lights
    global EPSILON
    if intersection is None:
        return scene_settings.background_color
    dif_sum = np.array([0, 0, 0], dtype='float')
    spec_sum = np.array([0, 0, 0], dtype='float')
    for light in lights:
        N = intersection.surface.get_normal(intersection.hit_point)
        L = light.position - intersection.hit_point
        L /= np.linalg.norm(L)
        N_L_dot = N @ L
        if N_L_dot <= 0:
            continue
        light_intensity, light_color = get_light_intensity_batch(light, intersection)
        V = intersection.ray.origin - intersection.hit_point
        V /= np.linalg.norm(V)
        light_ray = Ray(light.position, -L)
        R = intersection.surface.get_reflected_ray(light_ray, intersection.hit_point).v
        dif_sum += light_color * light_intensity * N_L_dot
        spec_sum += light_color * light_intensity * light.specular_intensity * \
                    (np.power(np.dot(R, V), intersection.surface.get_material(materials).shininess))
    diffuse_color = dif_sum * intersection.surface.get_material(materials).diffuse_color
    specular_color = spec_sum * intersection.surface.get_material(materials).specular_color

    color = diffuse_color + specular_color
    return color


def construct_image(res):
    global width
    global height
    global camera
    camera.calc_and_set_ratio(width)
    for i in range(height):
        for j in range(width):
            ray = Ray(camera, i, j, width, height)
            intersections = find_all_ray_intersections_sorted(ray)
            color = get_ray_color(intersections)
            res[i, j] = color
    res[res > 1] = 1
    res[res < 0] = 0
    return res * 255


def save_image(image_array, img_name):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save(f"{img_name}.png")


def xy_to_general_plane(plane_normal, plane_point):
    z_axis = np.array([0, 0, 1], dtype="float")
    if np.linalg.norm(z_axis - abs(plane_normal)) < EPSILON:  # only translate
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = plane_point
        return translation_matrix

    rotation_axis = np.cross(z_axis, plane_normal)
    cos_theta = np.dot(z_axis, plane_normal)
    sin_theta = np.linalg.norm(rotation_axis) / np.linalg.norm(plane_normal)
    rotation_axis /= np.linalg.norm(rotation_axis)

    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = (cos_theta * np.eye(3)) + \
                              (sin_theta * np.array([
                                  [0, -rotation_axis[2], rotation_axis[1]],
                                  [rotation_axis[2], 0, -rotation_axis[0]],
                                  [-rotation_axis[1], rotation_axis[0], 0]
                              ], dtype="float") +
                               (1 - cos_theta) * np.outer(rotation_axis, rotation_axis)
                               )

    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = plane_point

    transformation_matrix = np.matmul(translation_matrix, rotation_matrix)

    return transformation_matrix


def set_global_variables(camera_, scene_settings_, surfaces_, materials_, lights_, width_, height_, bonus_):
    global camera
    global scene_settings
    global surfaces
    global materials
    global lights
    global width
    global height
    global bonus
    global bonus_type
    bonus = False if bonus_ == -1 else True
    bonus_type = bonus_
    camera = camera_
    scene_settings = scene_settings_
    surfaces = surfaces_
    materials = materials_
    lights = lights_
    width = width_
    height = height_


def main():
    start = time.time()
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, default='./scenes/Room.txt', help='Path to the scene file')
    parser.add_argument('output_image', type=str, default='./output/try1.png', help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    parser.add_argument('--bonus', type=int, default=-1,
                        help='Bonus (-1 for  no bonus, 1 for bonus without coloring and 2 for bonus with coloring) ')

    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, surfaces, materials, lights = parse_scene_file(args.scene_file)
    set_global_variables(camera, scene_settings, surfaces, materials, lights, args.width, args.height, args.bonus)
    res = np.zeros((height, width, 3), dtype='float')
    image_array = construct_image(res)
    # Save the output image
    save_image(image_array, args.output_image)
    print(time.time()-start)

if __name__ == '__main__':
    main()
