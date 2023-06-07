import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper

env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")
env = FullyObsWrapper(env)

observation, info = env.reset(seed=42)

obs_dim = env.observation_space['image'].shape
print(obs_dim)
action_dim = env.action_space.n
print(action_dim)
total_reward, length = 0, 0

for _ in range(100):
   action = env.action_space.sample()
   observation, reward, terminated, truncated, info = env.step(action)
   total_reward += reward
   length += 1
   state = observation['image']
   if terminated or truncated:
      print(total_reward, length)
      total_reward, length = 0, 0
      observation, info = env.reset()
env.close()