from gymnasium import Wrapper
import numpy as np
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, LineType
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.objects import Landmark, Obstacle

class CarlaInitRoadWrapper(Wrapper):
    """Wrapper para construir el RoadNetwork con geometría de CARLA."""
    def __init__(self, env, carla_client, carla_vehicle):
        super().__init__(env)
        self.carla_client = carla_client
        self.carla_vehicle = carla_vehicle

    def get_parking_lanes_from_carla2(self):
        """
        Placeholder que simula las coordenadas de los cajones obtenidas de CARLA.
        Debes reemplazar esto con tu código real que obtenga geometría del mapa en CARLA.
        """
        return [
            {"entry": [-10, -18], "exit": [-10, -10], "width": 5.0},
            {"entry": [-5, -18], "exit": [-5, -10], "width": 5.0},
            {"entry": [0, -18], "exit": [0, -10], "width": 5.0},
            {"entry": [5, -18], "exit": [5, -10], "width": 5.0},
            {"entry": [10, -18], "exit": [10, -10], "width": 5.0},
        ]

    def get_parking_lanes_from_carla(self):
        """
        Extrae una sola línea central del cajón de estacionamiento desde las coordenadas de sus esquinas.
        """
        # Usamos las esquinas para obtener la línea central del cajón
        p1 = self.carla_client.get_location_by_coordinates(6, -30, 0)
        p2 = self.carla_client.get_location_by_coordinates(11.5, -30, 0)

        entry = [p1.x, p1.y]
        exit = [p2.x, p2.y]
        width = 3  # Ancho aproximado del cajón

        return [{
            "entry": entry,
            "exit": exit,
            "width": width
        }]

    def build_custom_road_network(self):
        net = RoadNetwork()
        lanes = self.get_parking_lanes_from_carla()
        road_center_x = 0
        road_center_y = 0

        for i, lane in enumerate(lanes):
            net.add_lane(
                "a", "b",
                StraightLane(
                    lane["entry"],
                    lane["exit"],
                    width=lane.get("width", 5.0),
                    line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
                )
            )
            road_center_x += (lane["entry"][0] + lane["exit"][0]) / 2
            road_center_y += (lane["entry"][1] + lane["exit"][1]) / 2

        road_center_x /= len(lanes)
        road_center_y /= len(lanes)

        road = Road(
            network=net,
            np_random=self.env.unwrapped.np_random,
            record_history=self.env.unwrapped.config.get("show_trajectories", False),
        )

        # 🔰 Añadir paredes en torno al centro del layout
        width, height = 60, 40
        for y in [-height / 2, height / 2]:
            obstacle = Obstacle(road, [road_center_x, road_center_y + y])
            obstacle.LENGTH, obstacle.WIDTH = (width, 1)
            obstacle.diagonal = np.sqrt(obstacle.LENGTH ** 2 + obstacle.WIDTH ** 2)
            road.objects.append(obstacle)
        for x in [-width / 2, width / 2]:
            obstacle = Obstacle(road, [road_center_x + x, road_center_y], heading=np.pi / 2)
            obstacle.LENGTH, obstacle.WIDTH = (height, 1)
            obstacle.diagonal = np.sqrt(obstacle.LENGTH ** 2 + obstacle.WIDTH ** 2)
            road.objects.append(obstacle)

        return road

    def _create_custom_vehicles2(self):
        """
        Versión personalizada de _create_vehicles:
        - Posición fija del vehículo.
        - Orientación aleatoria.
        - Goal (Landmark) en un cajón específico.
        """
        # 1. Crear el vehículo en posición fija (ej: x=0, y=0) y orientación aleatoria
        x0, y0 = 0.0, 0.0  # Posición inicial fija
        heading = 2 * np.pi * self.env.unwrapped.np_random.uniform()  # Orientación aleatoria en radianes
        #orientacion fija = 0.0  # Puedes fijar una orientación específica si lo deseas
        heading = -np.pi / 2

        vehicle = self.env.unwrapped.action_type.vehicle_class(
            self.env.unwrapped.road,
            [x0, y0],
            heading,
            0  # Velocidad inicial
        )
        vehicle.color = VehicleGraphics.EGO_COLOR
        self.env.unwrapped.road.vehicles.append(vehicle)
        self.env.unwrapped.controlled_vehicles = [vehicle]

        # 2. Crear el Landmark (goal) igual que en el original pero controlado por nosotros
        self._create_custom_goal(vehicle)

    def _create_custom_vehicles(self):
        """
        Crea un solo vehículo en la posición exacta donde se colocó en CARLA.
        """
        initial_loc = self.carla_vehicle.get_transform()
        x0 = initial_loc.location.x
        y0 = initial_loc.location.y
        yaw = initial_loc.rotation.yaw
        heading = np.deg2rad(yaw)

        vehicle = self.env.unwrapped.action_type.vehicle_class(
            self.env.unwrapped.road,
            [x0, y0],
            heading,
            0
        )
        vehicle.color = VehicleGraphics.EGO_COLOR
        self.env.unwrapped.road.vehicles.append(vehicle)
        self.env.unwrapped.controlled_vehicles = [vehicle]

        self._create_custom_goal(vehicle)

    def _create_custom_goal(self, vehicle):
        """
        Crea un Landmark (goal) en el centro de un cajón específico.
        - Usa la misma lógica que ParkingEnv pero con selección manual del cajón.
        """
        # Obtener todas las lanes de estacionamiento (cajones disponibles)
        lanes_dict = self.env.unwrapped.road.network.lanes_dict()
        lane_ids = list(lanes_dict.keys())

        if not lane_ids:
            raise ValueError("No hay cajones de estacionamiento definidos en el RoadNetwork.")

        # Elegir un cajón específico (ej: el primero, o puedes modificarlo)
        selected_lane_id = lane_ids[0]  # Primer cajón
        lane = lanes_dict[selected_lane_id]

        # Crear el Landmark en el centro del cajón (igual que ParkingEnv)
        vehicle.goal = Landmark(
            self.env.unwrapped.road,
            lane.position(lane.length / 2, 0),  # Posición central
            heading=lane.heading  # Orientación del cajón
        )
        self.env.unwrapped.road.objects.append(vehicle.goal)
        # print(f"Goal creado en posición: {vehicle.goal.position}, heading: {np.rad2deg(vehicle.goal.heading)}°")

    def reset(self, **kwargs):
        # Interceptar y reemplazar el road network antes del reset de ParkingEnv
        obs, info = self.env.reset(**kwargs)
        self.env.unwrapped.road = self.build_custom_road_network()
        self._create_custom_vehicles()

        # print("Post-reset, lanes del Road actual:")
        # for k, lane in self.env.unwrapped.road.network.lanes_dict().items():
        #     print(f"{k}: start={lane.start}, end={lane.end}")

        # print("obs shapes:")
        # print("  observation:", obs["observation"])
        # print("  achieved_goal:", obs["achieved_goal"])
        # print("  desired_goal:", obs["desired_goal"])
        # print("Duración máxima del episodio:", self.env.unwrapped.config["duration"])

        return obs, info

class CarlaObservationWrapper(Wrapper):
    """Wrapper para sobrescribir las observaciones con las de CARLA."""
    def __init__(self, env, carla_client, carla_vehicle):
        super().__init__(env)
        self.carla_client = carla_client
        self.carla_vehicle = carla_vehicle

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.sync_vehicle_state_from_carla()
        obs = self.override_obs_with_carla(obs)
        return obs, reward, done, truncated, info

    def sync_vehicle_state_from_carla(self):
        carla_vehicle = self.carla_vehicle
        transform = carla_vehicle.get_transform()
        velocity = carla_vehicle.get_velocity()

        x = transform.location.x
        y = transform.location.y
        yaw = np.deg2rad(transform.rotation.yaw)
        vx = velocity.x
        vy = velocity.y

        ego_vehicle = self.env.unwrapped.controlled_vehicles[0]
        ego_vehicle.position = np.array([x, y])
        ego_vehicle.heading = yaw
        ego_vehicle.speed = np.linalg.norm([vx, vy])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self.override_obs_with_carla(obs)
        return obs, info

    def override_obs_with_carla(self, obs):
        vehicle = self.carla_vehicle
        loc = vehicle.get_transform().location
        rot = vehicle.get_transform().rotation
        vel = vehicle.get_velocity()

        observation = np.array([
            loc.x, loc.y,
            vel.x, vel.y,
            np.cos(np.deg2rad(rot.yaw)),
            np.sin(np.deg2rad(rot.yaw))
        ])

        obs["observation"] = observation
        obs["achieved_goal"] = observation.copy()

        return obs

class CarlaActionWrapper(Wrapper):
    """Wrapper para enviar la acción al vehículo en CARLA."""
    def __init__(self, env, carla_client, carla_vehicle):
        super().__init__(env)
        self.carla_client = carla_client
        self.carla_vehicle = carla_vehicle

    def step(self, action):
        # Aquí se convertirá y aplicará la acción a CARLA
        self.apply_action_to_carla(action)
        return self.env.step(action)

    def apply_action_to_carla(self, action):
        """
        Convierte la acción [throttle, steering] de SAC en un control para CARLA.
        """
        throttle = action[0]
        steering = action[1]

        self.carla_client.aplly_control(
            self.carla_vehicle,
            throttle_input=float(throttle),
            steer=float(steering),
        )
        # print( f"Aplicando control a CARLA: Aceleración={throttle}, Dirección={steering}")
        self.carla_client.world.tick()