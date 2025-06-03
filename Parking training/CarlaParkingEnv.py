import time

import cv2
import gym
import numpy as np
from gym import spaces
from shapely.geometry import Polygon, Point
import carla


class CarlaParkingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        rendering_mode = True
        # steer ∈ [-1, 1]
        # throttle ∈ [-1, 1], negativo es reversa, 0 es freno

        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]),
                                       high=np.array([1.0, 1.0]),
                                       dtype=np.float32)

        # Observación similar a ParkingEnv
        obs_low = np.array([-100.0, -100.0, -np.pi, 0.0, -10.0, -10.0], dtype=np.float32)
        obs_high = np.array([100.0, 100.0, np.pi, 1.0, 10.0, 10.0], dtype=np.float32)
        goal_low = np.array([-100.0, -100.0, -np.pi], dtype=np.float32)
        goal_high = np.array([100.0, 100.0, np.pi], dtype=np.float32)

        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
            "achieved_goal": spaces.Box(low=goal_low, high=goal_high, dtype=np.float32),
            "desired_goal": spaces.Box(low=goal_low, high=goal_high, dtype=np.float32),
        })

        # Inicializa CARLA
        self.simulator = CarlaInterface(rendering_mode)
        self.sensors = SensorSuite(self.simulator)

        corner_coords = np.array([[corner.x, corner.y] for corner in self.simulator.goal_corners])
        self.goal_position = np.mean(corner_coords, axis=0)

        p1 = self.simulator.goal_corners[2]
        p2 = self.simulator.goal_corners[3]
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        self.goal_orientation = np.arctan2(dy, dx)
        self.max_steps = 500
        self.step_count = 0

    def reset(self):
        self.simulator.reset()
        self.step_count = 0
        obs = self._get_observation()
        achieved_goal = obs[:3]  # x, y, orientation actuales
        desired_goal = np.array([self.goal_position[0], self.goal_position[1], self.goal_orientation])
        return {
            "observation": obs,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal
        }

    def close(self):
        self.simulator.cleanup()
        cv2.destroyAllWindows()

    def step(self, action):
        self.simulator.apply_control(action)
        self.simulator.tick()

        obs = self._get_observation()
        achieved_goal = obs[:3]
        desired_goal = np.array([self.goal_position[0], self.goal_position[1], self.goal_orientation])
        reward = self.compute_reward(achieved_goal, desired_goal, info={})
        done = self._is_done(obs)

        self.step_count += 1
        return {
            "observation": obs,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal
        }, reward, done, {}

    def _get_observation(self):
        # Datos del vehículo desde el simulador
        ego_pose = self.sensors.get_ego_pose()
        ego_velocity = self.sensors.get_velocity()
        rel_position = self.goal_position - ego_pose[:2]
        rel_orientation = self.goal_orientation - ego_pose[2]

        return np.array([
            rel_position[0], rel_position[1], rel_orientation,
            ego_pose[3],  # steering
            ego_velocity[0], ego_velocity[1]
        ], dtype=np.float32)

    def _compute_reward(self, obs):
        x, y, orientation, steer, vx, vy = obs
        distance = np.linalg.norm([x, y])
        angle_diff = np.abs(orientation)
        speed_penalty = 0.1 * np.linalg.norm([vx, vy]) if distance < 2.0 else 0.0

        reward = -0.5 * distance - 0.2 * angle_diff - speed_penalty
        return reward

    def compute_reward(self, achieved_goal, desired_goal, info):
        if self.simulator.is_within_parking_zone():
            return 10.0  # gran recompensa por estacionar correctamente

        position_diff = achieved_goal[:2] - desired_goal[:2]
        orientation_diff = achieved_goal[2] - desired_goal[2]

        distance = np.linalg.norm(position_diff)
        angle_diff = np.abs(orientation_diff)

        reward = -0.5 * distance - 0.2 * angle_diff
        return reward

    def _is_done(self, obs):
        if self.simulator.is_within_parking_zone():
            return True
        if self.step_count >= self.max_steps:
            return True
        if self.simulator.has_collided():
            return True
        return False


class SensorSuite:
    def __init__(self, simulator):
        self.sim = simulator

    def get_ego_pose(self):
        """Devuelve (x, y, yaw, steering)"""
        transform = self.sim.get_vehicle_transform()
        x = transform.location.x
        y = transform.location.y
        yaw = np.radians(transform.rotation.yaw)
        steering = self.sim.get_steering_angle()
        return np.array([x, y, yaw, steering])

    def get_velocity(self):
        """Devuelve (vx, vy) en m/s"""
        velocity = self.sim.get_vehicle_velocity()
        return np.array([velocity.x, velocity.y])


class CarlaInterface:
    def __init__(self, rendering_mode=False):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.client.load_world('Town05')
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        self.camera_width = 1920
        self.camera_height = 1080
        self.sensor_camera = None
        self.SHOW_CAM = False
        self.vehicle = None
        self.front_camera = None
        self.collision_sensor = None
        self.collision_flag = False

        self.spawn_point = carla.Transform(
            carla.Location(x=20, y=-30, z=0.3),
            carla.Rotation(yaw=180)
        )
        self.goal_corners = [
            carla.Location(x=6, y=-28.5, z=0),
            carla.Location(x=6, y=-31.5, z=0),
            carla.Location(x=11.5, y=-28.5, z=0),
            carla.Location(x=11.5, y=-31.5, z=0)
        ]
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

    def reset(self):

        self.cleanup()
        bp_tesla = self.world.get_blueprint_library().filter('*model3*')
        # Reintentar hasta que se pueda hacer spawn sin colisión
        while self.vehicle is None or not self.vehicle.is_alive:
            cv2.destroyAllWindows()
            self.vehicle = self.world.try_spawn_actor(bp_tesla[0], self.spawn_point)
            time.sleep(1)

        # Coloca cámara
        location_mirror = carla.Location(x=0.5, y=0.0, z=1.3)
        transform_mirror = carla.Transform(location_mirror)

        bp_camera = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp_camera.set_attribute('image_size_x', str(self.camera_width))
        bp_camera.set_attribute('image_size_y', str(self.camera_height))
        bp_camera.set_attribute('sensor_tick', '0.1')
        bp_camera.set_attribute("fov", f"90")

        self.sensor_camera = self.world.spawn_actor(bp_camera, transform_mirror, attach_to=self.vehicle)
        self.sensor_camera.listen(lambda data: self.process_img(data))

        time.sleep(3)
        if self.SHOW_CAM:
            cv2.namedWindow('Camera', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Camera', self.front_camera)
            cv2.waitKey(1)

        self._setup_collision_sensor()

    def cleanup(self):
        for sensor in self.world.get_actors().filter('*sensor*'):
            sensor.destroy()
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        cv2.destroyAllWindows()

    def apply_control(self, action):
        steer, throttle_input = action
        control = carla.VehicleControl()
        # Dirección
        control.steer = float(np.clip(steer, -1.0, 1.0))

        # Aceleración y freno
        if throttle_input > 0:
            control.throttle = float(np.clip(throttle_input, 0.0, 1.0))
            control.brake = 0.0
            control.reverse = False
        elif throttle_input < 0:
            control.throttle = float(np.clip(-throttle_input, 0.0, 1.0))
            control.brake = 0.0
            control.reverse = True
        else:
            control.throttle = 0.0
            control.brake = 1.0
            control.reverse = False

        self.vehicle.apply_control(control)

        # print information in the camera
        if self.SHOW_CAM and self.front_camera is not None:
            # self.print_distances_info_in_camera(distances, steer, throttle)
            cv2.resizeWindow('Camera', self.camera_width, self.camera_height)
            cv2.imshow('Camera', self.front_camera)
            cv2.waitKey(1)

    def tick(self):
        self.world.tick()

    def get_vehicle_transform(self):
        return self.vehicle.get_transform()

    def get_steering_angle(self):
        return self.vehicle.get_control().steer

    def get_vehicle_velocity(self):
        return self.vehicle.get_velocity()

    def has_collided(self):
        return getattr(self, 'collision_flag', False)

    def _setup_collision_sensor(self):
        # if self.collision_sensor:
        #     self.collision_sensor.destroy()

        location_mirror = carla.Location(x=0.5, y=0.0, z=1.3)
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(location_mirror),
            attach_to=self.vehicle
        )
        self.collision_flag = False

        def _on_collision(event):
            self.collision_flag = True

        self.collision_sensor.listen(_on_collision)

    def _destroy_vehicle(self):
        if self.collision_sensor:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()
            self.collision_sensor = None
        if self.vehicle:
            self.vehicle.destroy()
            self.vehicle = None

    def is_within_parking_zone(self):
        location = self.vehicle.get_location()
        min_x = min(p.x for p in self.goal_corners)
        max_x = max(p.x for p in self.goal_corners)
        min_y = min(p.y for p in self.goal_corners)
        max_y = max(p.y for p in self.goal_corners)
        return min_x <= location.x <= max_x and min_y <= location.y <= max_y

    def process_img(self, image):
        # image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i = i.reshape((self.camera_height, self.camera_width, 4))[:, :, :3].astype(
            np.uint8)  # this is to ignore the 4th Alpha channel - up to 3
        self.front_camera = i
