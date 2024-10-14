import cv2
import numpy as np
import math
import time
import gym
from gym import spaces
import carla
from tensorflow.python.ops.summary_ops_v2 import get_step

camera_height = 240
camera_width = 320
N_CHANNELS = 3
FIXED_DELTA_SECONDS = 0.2

check_parking_point = carla.Transform(
    carla.Location(x=20, y=-30, z=0.3),
    carla.Rotation(yaw=180)
)
SHOW_PREVIEW = True


class CarlaParkingEnv(gym.Env):
    SHOW_CAM = SHOW_PREVIEW

    def __init__(self):
        super(CarlaParkingEnv, self).__init__()

        # Define action and observation space
        # now we use discrete actions

        self.action_space = spaces.MultiDiscrete([9, 4])
        # First discrete variable with 9 possible actions for steering with middle being straight
        # Second discrete variable with 4 possible actions for throttle/braking

        # Example for using image as input normalised to 0..1 (channel-first; channel-last also works):
        self.observation_space = (
            spaces.Box(low=0.0,
                       high=1.0,
                       shape=(camera_height, camera_width, N_CHANNELS),
                       dtype=np.float32
                       )
        )

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(15.0)
        self.client.load_world('Town05')

        self.world = self.client.get_world()


        self.settings = self.world.get_settings()
        self.settings.no_rendering_mode = True
        # self.settings.synchronous_mode = False
        # self.settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(self.settings)
        self.blueprint_library = self.world.get_blueprint_library()

        self.front_camera = None
        self.vehicle = None

    def reset(self):

        self.collision_hist = []
        self.actor_list = []
        self.vehicle_tesla = self.world.get_blueprint_library().filter('*model3*')

        self.initial_location = check_parking_point

        while self.vehicle is None or not self.vehicle.is_alive:
            # self.cleanup()
            cv2.destroyAllWindows()
            self.vehicle = self.world.try_spawn_actor(self.vehicle_tesla[0], check_parking_point)
            time.sleep(1)

        # vehicle_tesla.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        self.actor_list.append(self.vehicle)

        # camera = self.init_camera_vehicle(vehicle_tesla)
        self.camera = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.camera.set_attribute('image_size_x', str(camera_width))
        self.camera.set_attribute('image_size_y', str(camera_height))
        self.camera.set_attribute('sensor_tick', '0.1')  # Frecuencia de actualización de la
        self.camera.set_attribute("fov", f"90")


        camera_location_mirror = carla.Location(x=0.5, y=0.0, z=1.3)
        camera_transform_mirror = carla.Transform(camera_location_mirror)

        self.sensor = self.world.spawn_actor(self.camera, camera_transform_mirror, attach_to=self.vehicle)

        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(2)

        if self.SHOW_CAM:
            cv2.namedWindow('Camera', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Camera', self.front_camera)
            cv2.waitKey(1)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, camera_transform_mirror, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.steering_lock = False
        self.steering_lock_start = None  # this is to count time in steering lock and start penalising for long time in steering lock
        self.step_counter = 0

        return self.front_camera / 255.0

    def step(self, action):

        self.step_counter += 1
        steer = action[0]
        throttle = action[1]
        # map steering actions
        steer = self.get_stering(steer)
        throttle, brake = self.get_throttle_and_brake(throttle)
        self.vehicle.apply_control(
            carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake
            )
        )
        # print steer and throttle every 50 steps
        if self.step_counter % 50 == 0:
            print('steer input from model:', steer, ', throttle: ', throttle)

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        # distance_travelled = self.initial_location.distance(self.vehicle.get_location())

        # storing camera to return at the end in case the clean-up function destroys it
        cam = self.front_camera
        # showing image
        if self.SHOW_CAM:
            cv2.imshow('Camera', cam)
            cv2.waitKey(1)

        reward, done = self.get_step_reward(kmh, 0.0)
        # print('Reward:', reward)
        # print('Done:', done)
        return self.front_camera / 255.0, reward, done, {}

    def cleanup(self):
        for sensor in self.world.get_actors().filter('*sensor*'):
            sensor.destroy()
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        cv2.destroyAllWindows()

    def get_step_reward(self, kmh, reward):
        done = False
        # punish for collision
        if len(self.collision_hist) != 0:
            done = True
            reward = reward - 300
            self.cleanup()
        # reward for acceleration
        switcher = {
            kmh < 5: -3,
            kmh < 10: -1
        }
        reward = reward - switcher.get(kmh, 0.0)

        return reward, done

    def get_stering(self, steer):
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

    def get_throttle_and_brake(self, throttle):
        # switch case
        switcher = {
            0: (0.0, 1.0),
            1: (0.3, 0.0),
            2: (0.7, 0.5),
        }
        return switcher.get(throttle, (1.0, 0.0))

    def init_camera_vehicle(self, vehicle: carla.Actor):

        # Obtener la cámara frontal del vehículo
        camera_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_blueprint.set_attribute('image_size_x', str(camera_width))
        camera_blueprint.set_attribute('image_size_y', str(camera_height))
        camera_blueprint.set_attribute('sensor_tick', '0.1')  # Frecuencia de actualización de la

        # Configurar y spawnear la cámara frontal del espejo (mirror)
        camera_location_mirror = carla.Location(x=0.5, y=0.0,
                                                z=1.3)  # Ajustar la ubicación de la cámara según sea necesario
        camera_transform_mirror = carla.Transform(camera_location_mirror)
        camera_mirror = self.world.spawn_actor(camera_blueprint, camera_transform_mirror, attach_to=vehicle)
        camera_mirror.listen(lambda data: self.process_img(data))

        return camera_mirror

    def process_img(self, image):
        # image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i = i.reshape((camera_height, camera_width, 4))[:, :, :3]  # this is to ignore the 4th Alpha channel - up to 3
        self.front_camera = i

    def collision_data(self, event):
        self.collision_hist.append(event)
