import gymnasium as gym
from stable_baselines3 import PPO,HerReplayBuffer, SAC
import highway_env
from highway_carla_wrappers import *
from carla_parking import SimulationParking

def main(simulation):

   env = gym.make("parking-v0", render_mode="human")

   env = CarlaInitRoadWrapper(env, carla_client=simulation, carla_vehicle=vehicle)
   env = CarlaObservationWrapper(env, carla_client=simulation, carla_vehicle=vehicle)
   env = CarlaActionWrapper(env, carla_client=simulation, carla_vehicle=vehicle)

   # SAC hyperparams:
   model = SAC(
       "MultiInputPolicy",
       env,
       replay_buffer_class=HerReplayBuffer,
       replay_buffer_kwargs=dict(
           n_sampled_goal=4,
           goal_selection_strategy="future",
       ),
       verbose=1,
       learning_starts=int(1e4),
       buffer_size=int(1e6),
       learning_rate=1e-3,
       gamma=0.95,
       batch_size=256,
       policy_kwargs=dict(net_arch=[256, 256, 256]),
   )

   model.learn(int(1e5))
   model.save('her_sac_highway_carla')

def evaluate(simulation):
   env = gym.make("parking-v0", render_mode="human")
   env = CarlaInitRoadWrapper(env, carla_client=simulation, carla_vehicle=vehicle)
   # env = CarlaActionWrapper(env, carla_client=simulation, carla_vehicle=vehicle)
   model = SAC.load('her_sac_highway_carla', env=env)

   obs, _ = env.reset()

   # Evaluate the agent
   episode_reward = 0
   for _ in range(1000):
       action, _ = model.predict(obs, deterministic=True)
       obs, reward, terminated, truncated, info = env.step(action)
       done = truncated or terminated
       episode_reward += reward
       if done or info.get("is_success", False):
           print("Reward:", episode_reward, "Success?", info.get("is_success", False))
           episode_reward = 0.0
           obs, _ = env.reset()


if __name__ == "__main__":
        simulation = SimulationParking()
        simulation.load_world("Town05")
        initial_location = {
            "x": 20,
            "y": -30,
            "z": 0.3,
            "yaw": 180,
        }
        vehicle = simulation.init_vehicle("model3", initial_location)
        goal_corners = [
            simulation.get_location_by_coordinates(6, -28.5, 0),
            simulation.get_location_by_coordinates(6, -31.5, 0),
            simulation.get_location_by_coordinates(11.5, -28.5, 0),
            simulation.get_location_by_coordinates(11.5, -31.5, 0)
        ]
        simulation.init_spectator()

        main(simulation)
        evaluate(simulation)

