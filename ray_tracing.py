import argparse
import sys

from PIL import Image
import random
import numpy as np

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere


mtl, sph, pln, box, lgt = [], [], [], [], []

def parse_scene_file(file_path):
    objects = []
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
                camera = Camera(np.array(params[:3]), np.array(params[3:6]), np.array(params[6:9]), params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(np.array(params[:3]), params[3], params[4])
            elif obj_type == "mtl":
                material = Material(np.array(params[:3]), np.array(params[3:6]), np.array(params[6:9]), params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(np.array(params[:3]), params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(np.array(params[:3]), params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(np.array(params[:3]), params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(np.array(params[:3]), np.array(params[3:6]), params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def save_image(image_array , path):
    image = Image.fromarray(np.uint8(image_array))
    # Save the image to a file
    image.save(path)


def norm(vector):
    norm = np.linalg.norm(vector)
    if norm < sys.float_info.epsilon:
        return vector
    else:
        return vector / np.linalg.norm(vector)

def quadric_eq(a, b, c):
    eps = sys.float_info.epsilon
    discriminant = b ** 2 - (4 * a * c)
    if discriminant < eps:  # no solution
        return -1, -1, False
    discriminant = discriminant ** 0.5
    solution_1 = (-b - discriminant) / (2 * a)
    solution_2 = (-b + discriminant) / (2 * a)
    return solution_1, solution_2, True

def sphere_intersection(pass_shape, pass_index, ray_origin, direction_vec, min_t,obj_shape,obj_index):
    for i in range(len(sph)):
        if pass_shape == "sph" and pass_index == i:
            continue
        sphere= sph[i]
        a = np.dot(direction_vec, direction_vec)
        b = 2 * np.dot(direction_vec, ray_origin - sphere.position)
        c = np.dot(ray_origin - sphere.position, ray_origin - sphere.position) - sphere.radius ** 2
        eps = sys.float_info.epsilon
        t1, t2, discriminant = quadric_eq(a, b, c)
        if not discriminant:  # if discriminant < 0 no intersection
            continue
        t=min(t1,t2)
        if 0 < t < min_t:
            min_t = t
            obj_shape = "sph"
            obj_index = i

    return min_t, obj_shape,obj_index

def cube_intersection(pass_shape, pass_index, e, v, min_t, obj_shape, obj_index):
    for i, bx in enumerate(box):
        if pass_shape == "box" and pass_index == i:
            continue

        min_ext = bx.position - bx.scale / 2.0
        max_ext = bx.position + bx.scale / 2.0

        t_min, t_max = calculate_intersection_t(e, v, min_ext, max_ext)

        if is_intersection_valid(t_min, t_max, min_t):
            min_t = t_min
            obj_shape = "box"
            obj_index = i

    return min_t, obj_shape, obj_index

def calculate_intersection_t(e, v, min_ext, max_ext):
    t_min = (min_ext - e) / v
    t_max = (max_ext - e) / v
    t_min, t_max = np.minimum(t_min, t_max), np.maximum(t_min, t_max)
    return t_min.max(), t_max.min()

def is_intersection_valid(t_min, t_max, min_t):
    return (t_min <= t_max).all() and (0 < t_min < min_t)

def plane_intersection(pass_shape, pass_index, e, v, min_t,obj_shape,obj_index):
    for i in range(len(pln)):
        if pass_shape == "pln" and pass_index == i:
            continue
        plane= pln[i]
        denom = np.dot(plane.normal, v)
        if abs(denom) < sys.float_info.epsilon:
            t=sys.float_info.max
        else:
            t = (-1)*((np.dot(plane.normal, e) - plane.offset) / denom)
        if 0 < t < min_t:
            min_t=t
            obj_shape="pln"
            obj_index=i
    return min_t, obj_shape, obj_index

def FindIntersection(p,e,pass_shape,pass_index):
    min_t = sys.float_info.max
    obj_index = -1
    obj_shape = ""
    v = norm(p - e)

    min_t, obj_shape, obj_index = sphere_intersection(pass_shape, pass_index, e, v, min_t,obj_shape,obj_index)
    min_t, obj_shape,obj_index = cube_intersection(pass_shape, pass_index, e, v, min_t,obj_shape,obj_index)
    min_t, obj_shape,obj_index = plane_intersection(pass_shape, pass_index, e, v, min_t,obj_shape,obj_index)

    inter = e+min_t*v
    return inter, obj_shape, obj_index

def calc_color(bullseye, normal_vector, obj_diffuse, primitive_specular_color, shininess, camera, scene_settings):
    inter = bullseye[0]
    specular_color = np.zeros(3)
    diffuse_color = np.zeros(3)

    for l in lgt:
        obj2light = norm(l.position - inter)
        specular_intensity = l.specular_intensity

        if l.shadow_intensity == 0:
            ray_percent = 0
        else:
            ray_percent = calc_shadow(l.position, bullseye, l.radius, obj2light, scene_settings)

        light_intensity = (1 - l.shadow_intensity) + (l.shadow_intensity * ray_percent)

        projection = np.dot(normal_vector, obj2light)
        projection = max(0, projection)

        temp_diffuse = obj_diffuse * l.color * projection
        diffuse_color += temp_diffuse * light_intensity

        R = norm(((np.dot(2 * norm(obj2light), normal_vector)) * normal_vector) - norm(obj2light))
        RV = np.dot(R, norm(camera.position - inter))
        RV = max(0, RV) ** shininess

        temp_specular = primitive_specular_color * l.color * (specular_intensity * RV)
        specular_color += temp_specular * light_intensity

    return specular_color + diffuse_color


def calc_shadow(light_pos,bullseye,light_width,obj2light, scene_settings):
    shadow_rays = 0
    count = 0
    inter, obj_shape, obj_index = bullseye[0], bullseye[1], bullseye[2]
    original_inter=np.copy(inter)
    normal = (light_pos-inter)
    vx = norm(np.cross(normal, normal + np.array([0, 0, 1])))
    vy = norm(np.cross(normal, vx))
    p0 = light_pos - (vx * light_width / 2) - (vy * light_width / 2)
    p = np.copy(p0)
    vx = vx * (light_width / int(scene_settings.root_number_shadow_rays))
    vy = vy * (light_width / int(scene_settings.root_number_shadow_rays))
    i = 0
    while i < int(scene_settings.root_number_shadow_rays):
        j = 0
        while j < int(scene_settings.root_number_shadow_rays):
            x = random.random()
            y = random.random()
            p += (vx * x) + (vy * y)
            vector_to_light = p - inter

            new_inter, new_obj_shape, new_obj_index = FindIntersection(vector_to_light+p, inter, obj_shape, obj_index)
            if new_obj_index == -1:
                count += 1
            else:
                t=np.linalg.norm(inter-new_inter)
                if t > np.linalg.norm(p-inter):
                    count += 1

                while new_obj_index != -1 :
                    if new_obj_shape == "sph":
                        trans=mtl[sph[new_obj_index].material_index - 1].transparency
                    elif new_obj_shape == "box":
                        trans=mtl[box[new_obj_index].material_index - 1].transparency
                    else:
                        trans=mtl[pln[new_obj_index].material_index - 1].transparency

                    if trans == 0:
                        break
                    else:
                        count += trans
                        new_inter += vector_to_light * t
                        shadow_rays += 1
                        inter = np.copy(new_inter)
                        new_inter, new_obj_shape, new_obj_index=FindIntersection(vector_to_light+new_inter, new_inter,new_obj_shape,new_obj_index)

                    if new_obj_index == -1:
                        count+=1    
                    else:
                        t = np.linalg.norm(inter - new_inter)
                        if t > np.linalg.norm(p - original_inter):
                            count += 1
            p += vx * (1-x)
            p -= vy * y
            j += 1

        p -= (vx * int(scene_settings.root_number_shadow_rays))
        p += vy
        i += 1
    return count / (pow(int(scene_settings.root_number_shadow_rays), 2) + shadow_rays)


def find_normal(obj_shape, obj_index, inter):
    intersected_obj = []

    if (obj_shape == "sph"):
        intersected_obj = sph[obj_index]
        sph_center = intersected_obj.position
        normal_vector = norm(inter - sph_center)

    elif (obj_shape == "box"):
        eps = 1 * 10**-8
        normal_vector = np.zeros(3)
        intersected_obj = box[obj_index]
        if (intersected_obj.position[0]-0.5*intersected_obj.scale + eps >= inter[0]):
            normal_vector=np.array([-1.0,0.0,0.0])
        elif (intersected_obj.position[0]+0.5*intersected_obj.scale - eps<= inter[0]):
            normal_vector=np.array([1.0,0.0,0.0])
        elif (intersected_obj.position[1]-0.5*intersected_obj.scale + eps>= inter[1]):
            normal_vector=np.array([0.0,-1.0,0.0])
        elif (intersected_obj.position[1]+0.5*intersected_obj.scale - eps<= inter[1]):
            normal_vector=np.array([0.0,1.0,0.0])
        elif (intersected_obj.position[2]-0.5*intersected_obj.scale + eps>= inter[2]):
            normal_vector=np.array([0.0,0.0,-1.0])
        elif (intersected_obj.position[2]+0.5*intersected_obj.scale - eps<= inter[2]):
            normal_vector=np.array([0.0,0.0,1.0])
        else:
            print("error")


    elif (obj_shape == "pln"):
        intersected_obj = pln[obj_index]
        normal_vector = intersected_obj.normal
    else:
        print("error in itersection between ray and object")

    return intersected_obj, normal_vector

def GetColor(bullseye,max_rec, start_position_of_ray , camera , scene_settings):
    inter = bullseye[0]
    transparency_color=np.zeros(3)
    reflection_color=np.zeros(3)
    obj_shape=bullseye[1]
    obj_index=bullseye[2]
    background_color=scene_settings.background_color
    if(obj_index==-1):
        return background_color
    intersected_obj, normal_vector = find_normal(obj_shape, obj_index, inter)
    obj_diffuse = mtl[intersected_obj.material_index -1 ].diffuse_color
    primitive_specular_color= mtl[intersected_obj.material_index -1].specular_color
    diffuse_plus_specular=calc_color(bullseye,normal_vector,obj_diffuse,primitive_specular_color,mtl[intersected_obj.material_index -1].shininess, camera , scene_settings)
    if mtl[intersected_obj.material_index-1].transparency != 0 and max_rec > 0:
        next_inter = FindIntersection(inter-start_position_of_ray, inter, obj_shape, obj_index)
        transparency_color += GetColor(next_inter,max_rec-1,inter , camera , scene_settings)

    if max_rec > 0:
        cur_vector=norm(inter-start_position_of_ray)
        reflected_vector=norm((( np.dot(2*cur_vector,normal_vector) )*normal_vector) - cur_vector)
        next_inter=FindIntersection(inter,inter+reflected_vector, obj_shape,obj_index)
        next_reflection_color=GetColor(next_inter,max_rec-1,inter , camera , scene_settings)
        final_reflection_color = next_reflection_color * mtl[intersected_obj.material_index - 1 ].reflection_color
        reflection_color = reflection_color + final_reflection_color
    return (transparency_color*mtl[intersected_obj.material_index-1].transparency) + (diffuse_plus_specular*(1-mtl[intersected_obj.material_index-1].transparency)) + reflection_color


def ray_trace(vx, vy, p0, height, width, e, camera, scene_settings):
    image = np.zeros((height, width, 3), dtype=np.float32)
    cam_eye = camera.position
    for i in range(height):
        p = np.ndarray.copy(p0)
        print(i)
        for j in range(width):
            bullseye = FindIntersection(np.ndarray.copy(p), np.ndarray.copy(e), None, -1)
            x, y, z = GetColor(bullseye, int(scene_settings.max_recursions) ,cam_eye , camera , scene_settings)
            x, y, z = min(x * 255, 255), min(y * 255, 255), min(z * 255, 255)
            image[height - i - 1][j] = np.array([x, y, z])
            
            p += vx*(camera.screen_width/width)
        p0 += vy * ((camera.screen_width*float(height)/float(width))/height)

    return image

def parse_objects(objects):
    for obj in objects:
        if isinstance(obj, Sphere):
            sph.append(obj)
        elif isinstance(obj, InfinitePlane):
            pln.append(obj)
        elif isinstance(obj, Cube):
            box.append(obj)
        elif isinstance(obj, Light):
            lgt.append(obj)
        elif isinstance(obj, Material):
            mtl.append(obj)
        else:
            print("WTF?")
    


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)
    parse_objects(objects)
    # TODO: Implement the ray tracer

    # Dummy result
    image_array = np.zeros((500, 500))

    height = int(args.height)
    width = int(args.width)
    vx = np.zeros(3)
    vy = np.zeros(3)
    vz = norm(camera.look_at - camera.position)
    sx = (-1) * vz[1]
    cx = (1 - sx ** 2) ** 0.5
    sy = (-1) * (vz[0]) / cx
    cy = vz[2] / cx
    vx[0], vx[1], vx[2] = cy, 0, sy
    vy[0], vy[1], vy[2] = ((-1) * sx * sy), cx, sx*cy
    vx = norm(vx)
    vy = norm(vy)


    p = camera.position + vz * camera.screen_distance
    sc_height = camera.screen_width * float(height) / float(width)

    p0 = p - (float(camera.screen_width / 2) * vx) - (float(sc_height / 2) * vy)

    image_array = ray_trace(vx, vy, np.ndarray.copy(p0), height, width, camera.position, camera, scene_settings)

    # Save the output image
    save_image(image_array,args.output_image)


if __name__ == '__main__':
    main()
