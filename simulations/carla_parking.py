import sys

import cv2
import numpy as np
import time

sys.path.append('E:/UADY/CARLA/CARLA_Latest/WindowsNoEditor/PythonAPI/carla')

import carla


class SimulationParking:

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

    def load_world(self, world_name):
        self.client.set_timeout(15.0)
        self.client.load_world(world_name)

    def init_vehicle(self, model, location):
        bp_vehicle = self.blueprint_library.filter(f"*{model}*")

        location_transform = carla.Transform(
            carla.Location(x=location['x'], y=location['y'], z=location['z']),
            carla.Rotation(yaw=location['yaw'])
        )
        actor_vehicle = self.world.try_spawn_actor(bp_vehicle[0], location_transform)
        return actor_vehicle

    def init_camera(self, vehicle, location, type, image):
        width, height, _ = image.shape

        bp_camera = self.blueprint_library.find(f'sensor.camera.{type}')
        bp_camera.set_attribute('image_size_x', str(width))
        bp_camera.set_attribute('image_size_y', str(height))
        bp_camera.set_attribute('sensor_tick', '0.1')  # Frecuencia de actualización de la
        bp_camera.set_attribute("fov", f"90")

        location_transform = carla.Transform(
            carla.Location(x=location['x'], y=location['y'], z=location['z'])
        )

        sensor_camera = self.world.spawn_actor(bp_camera, location_transform, attach_to=vehicle)

        return sensor_camera

    def add_camera_listen_capture_images(self, sensor_camera, image_path, tags_callback=None):
        sensor_camera.listen(
            lambda data:
            self.capture_image(
                data,
                image_path,
                tags_callback(sensor_camera) if tags_callback else None
            )
        )

    def aplly_control(self, vehicle, throttle, steer, brake):
        vehicle.apply_control(
            carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake
            )
        )

    def capture_image(self, image, image_path, tags):
        image_array = np.frombuffer(image.raw_data, dtype=np.uint8)
        image_array = np.reshape(image_array, (image.height, image.width, 4))
        image_bgr = image_array[:, :, :3]  # Convertir de formato RGBA a formato BGR para OpenCV
        cv2.imwrite(f'{image_path}{tags}_.jpg', image_bgr)

    def process_img(self, image_from, image_to):
        height, width, _ = image_to.shape
        # image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image_from.raw_data)
        i = i.reshape((height, width, 4))[:, :, :3].astype(
            np.uint8)  # this is to ignore the 4th Alpha channel - up to 3
        image_to = i

    def init_spectator(self):
        new_location = carla.Location(x=18.633804, y=-19.904999, z=7.794066)
        new_rotation = carla.Rotation(pitch=-29.543756, yaw=-117.807945, roll=0.000025)
        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(new_location, new_rotation))

    def calculate_distance_actor_to_location(self, actor, location):
        return actor.get_transform().location.distance(location)

    def calculate_angle_actor_to_location(self, actor, location):
        # Obtener la posición actual del actor (auto)
        actor_location = actor.get_transform().location

        # Calcular el vector de distancia del actor al punto
        vector_distance = np.array([location.x - actor_location.x,
                                    location.y - actor_location.y,
                                    location.z - actor_location.z])

        # Obtener la orientación hacia adelante del actor (auto)
        forward_vector = actor.get_transform().get_forward_vector()
        forward_vector = np.array([forward_vector.x, forward_vector.y, forward_vector.z])

        # Normalizar ambos vectores
        vector_distance_norm = vector_distance / np.linalg.norm(vector_distance)
        forward_vector_norm = forward_vector / np.linalg.norm(forward_vector)

        # Calcular el ángulo entre ambos vectores utilizando el producto punto
        dot_product = np.dot(forward_vector_norm, vector_distance_norm)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Asegurar que esté en el rango [-1, 1]

        # Convertir el ángulo de radianes a grados
        angle_degrees = np.degrees(angle)

        return angle_degrees

    def calculate_distance(self, camera_location, target_location):
        return camera_location.distance(target_location)

    def get_location_by_coordinates(self, x, y, z):
        return carla.Location(x=x, y=y, z=z)
