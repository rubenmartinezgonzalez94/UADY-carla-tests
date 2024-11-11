import sys
import time

import cv2
import numpy as np

sys.path.append('../')
from carla_parking import SimulationParking

def tags_callback(sensor):

    created = time.time()
    tags = f"s=camera_front_mirror_"  # Etiqueta inicial
    # Añadir la marca de tiempo
    tags += f"t={created}"

    return tags

initial_location = {
    'x': 20,
    'y': -30,
    'z': 0.5,
    'yaw': 180
}

location_mirror = {
    'x': 0.5,
    'y': 0.0,
    'z': 1.3
}

simulation = SimulationParking()
print('simulation initialized')
simulation.load_world('Town05')
print('world loaded')
vehicle = simulation.init_vehicle('model3', initial_location)
print('vehicle initialized')
simulation.init_spectator()
print('spectator initialized')
camera = simulation.init_camera(
    vehicle, location_mirror, 'rgb', np.zeros((1920, 1080, 3))
)
print('camera initialized')
simulation.add_camera_listen_capture_images(
    camera,
    './manual_sequence/',
    tags_callback
)

simulation.enable_manual_control(vehicle)

vehicle.destroy()
camera.destroy()