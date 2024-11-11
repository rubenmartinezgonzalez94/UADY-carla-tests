import sys

import cv2
import numpy as np

sys.path.append('../')
from carla_parking import SimulationParking

location_mirror = {
    'x': 0.5,
    'y': 0.0,
    'z': 1.3
}
routes = [
    [ {
        'x': 20,
        'y': -30,
        'z': 0.5,
        'yaw': 180
    },
    {
        'x': 60,
        'y': -28.5,
        'z': 0,
        'yaw': 0
    }
    ]]

simulation = SimulationParking()
print('simulation initialized')
simulation.load_world('Town05')
print('world loaded')
vehicle = simulation.init_vehicle('model3', routes[0][0])
print('vehicle initialized')
simulation.init_spectator()
print('spectator initialized')

camera = simulation.init_camera(
    vehicle, location_mirror, 'rgb', np.zeros((1920, 1080, 3))
)
print('camera initialized')
simulation.move_vehicle_to(vehicle, routes[0][1])
print('vehicle moved')

# simulation.add_camera_listen_capture_images(
#     camera,
#     './parking_sequence/',
#     lambda sensor: f'camera_{sensor.id}'
# )
# vehicle.set_autopilot(True)
# cv2.waitKey(3000)
# vehicle.destroy()
# camera.destroy()
