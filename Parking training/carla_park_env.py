import cv2
import numpy as np
import math
import time
import gym
from gym import spaces
import carla


class CarlaParkingEnv(gym.Env):
    camera_height = 600
    camera_width = 800
    SHOW_CAM = True

    initial_location = carla.Transform(
        carla.Location(x=20, y=-30, z=0.3),
        carla.Rotation(yaw=180)
    )

    corners_points = [
        carla.Location(x=6, y=-28.5, z=0),
        carla.Location(x=6, y=-31.5, z=0),
        carla.Location(x=11.5, y=-28.5, z=0),
        carla.Location(x=11.5, y=-31.5, z=0)
    ]
    # thresholds distances to consider the episode done
    aceptable_distance = 5  #  meters from camera to corners
    unacceptable_distance = 15  #  meters from camera to corners
    unacceptable_duration = 10  #  seconds

    def __init__(self):
        super(CarlaParkingEnv, self).__init__()

        # actors
        self.blueprint_library = None
        self.world = None
        self.client = None
        self.sensor_collision = None
        self.sensor_camera = None
        self.vehicle = None
        self.front_camera = None

        # training
        self.step_counter = None

        self.steering_lock_start = None
        self.steering_lock = None
        self.episode_start = None

        # self.previous_distances = None
        # self.steps_stuck = None
        self.collision_hist = []

        # Define action and observation space
        # now we use discrete actions
        # First discrete variable with 9 possible actions for steering with middle being straight
        # Second discrete variable with 6 possible actions for throttle/braking
        self.action_space = spaces.MultiDiscrete([9, 6])

        # Define observation space (distances to the corners)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.init_simulator(rendering_mode=True)

    def reset(self):

        self.collision_hist = []
        self.cleanup()
        bp_tesla = self.world.get_blueprint_library().filter('*model3*')

        while self.vehicle is None or not self.vehicle.is_alive:
            #
            cv2.destroyAllWindows()
            self.vehicle = self.world.try_spawn_actor(bp_tesla[0], self.initial_location)
            time.sleep(1)

        location_mirror = carla.Location(x=0.5, y=0.0, z=1.3)
        transform_mirror = carla.Transform(location_mirror)

        # camera = self.init_camera_vehicle(vehicle_tesla)
        bp_camera = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp_camera.set_attribute('image_size_x', str(self.camera_width))
        bp_camera.set_attribute('image_size_y', str(self.camera_height))
        bp_camera.set_attribute('sensor_tick', '0.1')  # Frecuencia de actualización de la
        bp_camera.set_attribute("fov", f"90")

        self.sensor_camera = self.world.spawn_actor(bp_camera, transform_mirror, attach_to=self.vehicle)
        self.sensor_camera.listen(lambda data: self.process_img(data))

        time.sleep(3)
        if self.SHOW_CAM:
            cv2.namedWindow('Camera', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Camera', self.front_camera)
            cv2.waitKey(1)

        bp_col_sensor = self.blueprint_library.find("sensor.other.collision")
        self.sensor_collision = self.world.spawn_actor(bp_col_sensor, transform_mirror, attach_to=self.vehicle)
        self.sensor_collision.listen(lambda event: self.collision_data(event))

        self.vehicle.apply_control(carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0))

        self.episode_start = time.time()
        self.steering_lock = False
        self.steering_lock_start = None  # this is to count time in steering lock and start penalising for long time in steering lock
        self.step_counter = 0

        distances = self.get_observation()
        return distances

    def step(self, action):

        self.step_counter += 1

        # map steering actions
        steer = action[0]
        throttle = action[1]
        steer = get_stering(steer)
        throttle, brake = get_throttle_and_brake(throttle)

        # apply control actions
        self.vehicle.apply_control(
            carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake
            )
        )

        # calculate distances from camera to corners
        distances = self.get_observation()

        # print information in the camera
        if self.SHOW_CAM and self.front_camera is not None:
            self.print_distances_info_in_camera(distances, steer, throttle)
            cv2.resizeWindow('Camera', self.camera_width, self.camera_height)
            cv2.imshow('Camera', self.front_camera)
            cv2.waitKey(1)

        # Calcular recompensa y verificar si el episodio ha terminado
        reward, done = self.get_reward(distances, throttle)

        if self.step_counter % 20 == 0:
            print('steer input from model:', steer, ', throttle: ', throttle)
            print('reward:', reward)

        return distances, reward, done, {}

    def get_reward(self, distances, throttle):
        done = False
        reward = 0

        total_distance = np.sum(distances)
        distance_deviation = np.std(distances)
        episode_duration = time.time() - self.episode_start

        # Criterio de finalización (el agente está muy cerca del objetivo)
        if distance_deviation <= 0.2 and total_distance <= self.aceptable_distance * 4:
            print("Recompensa por completar el objetivo")
            done = True
            reward += 100  # Gran recompensa por completar el objetivo
            self.cleanup()
            return reward, done

        # Criterio de finalización (el agente se aleja demasiado del objetivo)
        if total_distance >= self.unacceptable_distance * 4:
            print("Penalización por alejarse demasiado del objetivo: ", total_distance)
            done = True
            reward -= 10
            self.cleanup()
            return reward, done

        # Criterio de finalización (el episodio dura demasiado tiempo)
        if episode_duration > self.unacceptable_duration:
            print("Penalización por duración del episodio")
            done = True
            reward -= 10
            self.cleanup()
            return reward, done

        # recompensa por acortar distancias
        distance_reward = punish_logistic(total_distance, m=self.aceptable_distance * 4, p=1)
        reward += distance_reward
        print("Recompensa por acortar distancias: + ", distance_reward)

        # Penalizar si se para y las distancias son grandes
        if throttle < 0.2 and total_distance > self.unacceptable_distance * 2:
            distance_penalty = punish_logistic(total_distance, m=self.unacceptable_distance * 4, p=1)
            reward -= distance_penalty
            print("Penalización por parar muy lejos: - ", distance_penalty)

        # Penalización por duración del episodio (rapidez)
        time_penalty = punish_logistic(episode_duration, m=self.unacceptable_duration, p=1)
        reward -= time_penalty
        print("Penalización por duración del episodio: - ", time_penalty)

        # # Calcula una recompensa inversa a la distancia total
        # # Cuanto más pequeña sea la distancia, mayor será la recompensa
        # distance_reward = self.unacceptable_distance * 2 / (total_distance + 1e-5)
        # reward += distance_reward

        # # Penalizar si hay mucha diferencia entre las distancias
        # reward -= distance_deviation * 10

        # # punish for collision
        # if len(self.collision_hist) != 0:
        #     print("Penalización por colisión")
        #     done = True
        #     reward = reward - 1000
        #     self.cleanup()
        #     return reward, done
        #


        # Penalización si se ha quedado "parado" durante demasiados pasos
        # if self.steps_stuck > 700 and total_distance > 8:  # Si se queda más de 6 pasos sin cambiar distancias
        #     reward -= 10 * self.steps_stuck  # Penaliza más cuanto más tiempo esté "parado"
        #     print("Penalización por estar parado:", self.steps_stuck)
        # if self.steps_stuck > 1000:
        #     print("Fin del episodio por estar parado")
        #     done = True
        #     reward = reward - 300
        #     self.cleanup()
        #     return reward, done

        # # Recompensa adicional por ir a una velocidad razonable (ni muy lento ni demasiado rápido)
        # if kmh > 0:
        #     speed_reward = min(kmh, 30) / 30  # Premia velocidades moderadas (hasta 30 km/h)
        #     reward += speed_reward

        return reward, done

    def get_observation(self):
        camera_transform = self.sensor_camera.get_transform()
        camera_location = camera_transform.location
        return np.array([calculate_distance(camera_location, point) for point in self.corners_points])

    def cleanup(self):
        for sensor in self.world.get_actors().filter('*sensor*'):
            sensor.destroy()
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        cv2.destroyAllWindows()

    def init_simulator(self, rendering_mode=False):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.client.load_world('Town05')
        self.world = self.client.get_world()

        if not rendering_mode:
            settings = self.world.get_settings()
            settings.no_rendering_mode = False
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1
            self.world.apply_settings(settings)
        else:
            # spectator
            new_location = carla.Location(x=18.633804, y=-19.904999, z=7.794066)
            new_rotation = carla.Rotation(pitch=-29.543756, yaw=-117.807945, roll=0.000025)
            spectator = self.world.get_spectator()
            spectator.set_transform(carla.Transform(new_location, new_rotation))

        self.blueprint_library = self.world.get_blueprint_library()

    def process_img(self, image):
        # image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i = i.reshape((self.camera_height, self.camera_width, 4))[:, :, :3].astype(
            np.uint8)  # this is to ignore the 4th Alpha channel - up to 3
        self.front_camera = i

    def collision_data(self, event):
        self.collision_hist.append(event)

    def print_distances_info_in_camera(self, distances, steer, throttle):
        # print distance1
        texts_to_print=[
            f"Distances: {distances[0]:.2f}, {distances[1]:.2f}, {distances[2]:.2f}, {distances[3]:.2f}",
            f"Total distance: {np.sum(distances):.2f}m",
            f"Std deviation: {np.std(distances):.2f}m",
            f"Steer: {steer:.2f}",
            f"Throttle: {throttle:.2f}",
            f"Step: {self.step_counter:.2f}",
            f"Time: {time.time() - self.episode_start:.2f}s"
        ]
        for i, text in enumerate(texts_to_print):
            cv2.putText(
                self.front_camera,
                text,
                (10, 30 + 30 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


def calculate_distance(camera_location, target_location):
    return camera_location.distance(target_location)

def calculate_distanceV2(vehicle_location, target_location):
    dx = vehicle_location.x - target_location.x
    dy = vehicle_location.y - target_location.y
    dz = vehicle_location.z - target_location.z
    return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

def get_stering(steer):
    # switch case
    switcher = {
        0: - 0.9,
        1: -0.25,
        2: -0.1,
        3: -0.05,
        4: 0.0,
        5: 0.05,
        6: 0.1,
        7: 0.25,
        8: 0.9
    }
    return switcher.get(steer, 0.0)

def get_throttle_and_brake(throttle):
    # switch case
    switcher = {
        0: (0.0, 1.0),
        1: (0.0, 0.5),
        2: (0.3, 0.0),
        3: (0.7, 0.0),
        4: (1.0, 0.0),
    }
    return switcher.get(throttle, (0.0, 0.0))

# v es el valor que se quiere penalizar
# k controla la inclinación de la curva (qué tan rápido crece la penalización)
# m controla la penalización maxima
# p controla la magnitud de la penalización
def punish_logistic(v, k=0.1, m=1, p=1):
    logistic = 1 / (1 + np.exp(-k * (v - m)))
    return logistic * p


