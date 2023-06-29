import gymnasium as gym
from minigrid.manual_control import ManualControl
from customenvs import SimpleBoxesEnv, MazeBoxesEnv

env = SimpleBoxesEnv(render_mode="human")
env.reset()

manual_control = ManualControl(env, seed=42)
manual_control.start()

# simulate 10 steps printing the reward and observation at each step
for _ in range(10):
    action = env.action_space.sample()
    print(f"action: {action}")
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"reward: {reward}")
    if terminated or truncated:
        break

# enable manual control for testing
#manual_control = ManualControl(env, seed=42)
#manual_control.start()