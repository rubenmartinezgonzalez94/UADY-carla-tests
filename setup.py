from setuptools import setup
import sys
import subprocess

if sys.version_info < (3, 8):
    sys.exit("This project requires Python 3.8+ by carla library")

def install_carla_wheel():
    wheel_path = "../carla/dist/carla-0.9.15-cp38-cp38-linux_x86_64.whl"
    subprocess.run([sys.executable, "-m", "pip", "install", wheel_path])

install_carla_wheel()
setup(
    name="CarlaParkingProcessing",
    version="1.0.0",
    install_requires=[
        'tensorflow[and-cuda]',
        'gymnasium',
        'pygame',
        'stable_baselines3',
        'highway_env',
        'opencv-python',
        'numpy',
        'carla @ file://../carla/dist/carla-0.9.15-cp38-cp38-linux_x86_64.whl',
    ],
)