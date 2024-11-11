import sys

import cv2
import numpy as np
import time
import math
import pygame

# sys.path.append('E:/UADY/CARLA/CARLA_Latest/WindowsNoEditor/PythonAPI/carla')

import carla


class SimulationParking:

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

    def load_world(self, world_name):
        current_world = self.client.get_world().get_map().name

        if current_world == world_name:
            return

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

    def move_vehicle_to(self, vehicle, destination):

        destination_Transform = carla.Transform(
            carla.Location(x=destination['x'], y=destination['y'], z=destination['z'])
        )
        destination_location = destination_Transform.location

        # Inicializar el control del vehículo
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.steer = 0.0
        control.brake = 0.0
        # Máximo ángulo permitido para el giro (en radianes)
        max_steer_angle = math.radians(30)

        # Bucle de control para moverse al destino
        while True:
            distance = self.calculate_distance_actor_to_location(vehicle, destination_location)
            print(f"distance: {distance}")

            angle = self.calculate_angle_actor_to_location(vehicle, destination_location)
            # conversion from degrees to -1 to +1 input for apply control function
            # Ajusta el ángulo al rango [-180, 180]
            alfa = ((angle + 180) % 360) - 180
            # Normaliza el ángulo al rango [-1, 1]
            alfa = alfa / 180
            control.steer = alfa / 75

            # Ajustar la aceleración
            if distance > 1.0:  # Si la distancia al destino es mayor que 1 metro
                control.throttle = 0.15
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = 1.0
                vehicle.apply_control(control)
                print("Llegó al destino")
                break

            # Aplicar el control al vehículo
            vehicle.apply_control(control)
            self.world.tick()
            time.sleep(0.05)

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

    def enable_manual_control(self, vehicle):
        """
        Controla un vehículo en CARLA utilizando el teclado.
        W - Acelerar
        S - Frenar
        A - Girar a la izquierda
        D - Girar a la derecha
        R - Revesa On/Off
        Escape - Salir del control manual
        """
        pygame.init()
        screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Manual Vehicle Control")

        control = carla.VehicleControl()
        control.throttle = 0.0
        control.steer = 0.0
        control.brake = 0.0
        control.hand_brake = False
        control.reverse = False

        clock = pygame.time.Clock()

        print("Controles: W (acelerar), S (frenar), A (izquierda), D (derecha), Espacio (freno de mano)")

        try:
            while True:
                # Manejar eventos de salida
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return

                # Obtener el estado de todas las teclas
                keys = pygame.key.get_pressed()

                # Control del vehículo
                if keys[pygame.K_w]:
                    control.throttle = 0.6  # Acelerar
                    control.brake = 0.0
                else:
                    control.throttle = 0.0

                if keys[pygame.K_s]:
                    control.brake = 0.8  # Frenar
                    control.throttle = 0.0
                else:
                    control.brake = 0.0

                if keys[pygame.K_a]:
                    control.steer = max(-1.0, control.steer - 0.1)  # Girar a la izquierda
                elif keys[pygame.K_d]:
                    control.steer = min(1.0, control.steer + 0.1)  # Girar a la derecha
                else:
                    control.steer = 0.0  # No girar

                if keys[pygame.K_r]:
                    control.reverse = not control.reverse # Reversa On/Off

                if keys[pygame.K_SPACE]:
                    control.hand_brake = True
                else:
                    control.hand_brake = False

                if keys[pygame.K_ESCAPE]:
                    pygame.quit()
                    print("\nControl manual terminado")
                    return
                # Aplicar el control al vehículo
                vehicle.apply_control(control)

                # Establecer FPS para el bucle de control
                clock.tick(30)
                self.world.tick()

        except KeyboardInterrupt:
            print("\nControl manual terminado")
        finally:
            pygame.quit()
