import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
import time

def get_state_tensor(state, cnn=True):
    if cnn:
        image = torch.tensor(state['image'], dtype=torch.float32)
        image_size = image.shape[-2]
        direction = torch.from_numpy(state['direction'])
        image = image.permute(0,3,1,2)
        direction_channel = direction.reshape((direction.shape[0], 1, 1, 1)).expand((-1, 1, image_size, image_size))
        state_tensor = torch.cat((image, direction_channel), dim=1)
        return state_tensor
    else:
        return torch.tensor(state['image'], dtype=torch.float32)

def plot_logs(timesteps, rewards, episode_lengths, step, smooth=True, title="ppo", save_path="ppo.png"):
    # plot both rewards and episode lengths in same figure, but different scales
    alpha_non_smoothed, n_smooth = 1, 5
    rewards, rewards_std = np.array(rewards).T
    episode_lengths, episode_lengths_std = np.array(episode_lengths).T
    smooth = smooth and len(rewards) > n_smooth
    if smooth:
        conv_smooth = np.ones((n_smooth,))/n_smooth
        smoothed_rewards = np.convolve(rewards, conv_smooth, mode='valid')
        smoothed_episode_lengths = np.convolve(episode_lengths, conv_smooth, mode='valid')
        alpha_non_smoothed = 0.2
    fig, ax1 = plt.subplots()
    ax1.plot(timesteps, rewards, 'b-', alpha=alpha_non_smoothed)
    #ax1.fill_between(timesteps, rewards-rewards_std, rewards+rewards_std, color='b', alpha=0.2)
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Average reward', color='b')
    ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    ax2.plot(timesteps, episode_lengths, 'r-', alpha=alpha_non_smoothed)
    #ax2.fill_between(timesteps, episode_lengths-episode_lengths_std, episode_lengths+episode_lengths_std, color='r', alpha=0.2)
    ax2.set_ylabel('Average episode length', color='r')
    ax2.tick_params('y', colors='r')
    if smooth:
        ax1.plot(timesteps[n_smooth-1:], smoothed_rewards, 'b-')
        ax2.plot(timesteps[n_smooth-1:], smoothed_episode_lengths, 'r-')
    if len(timesteps) > 1:
        plt.xlim(0, timesteps[-1]*1.05)
    plt.title(title)
    plt.savefig(save_path, dpi=200)
    plt.close()
    

class TimeCostWrapper(gym.Wrapper):
    """
    Wrapper which adds a cost to actions and time.
    """
    
    def __init__(self, env, action_cost=0.01, time_cost=0.01, noops_actions=[6]):
        """A wrapper that adds a cost to actions and time.

        Args:
            env: The environment to apply the wrapper
            action_cost: The cost of an action
            time_cost: The cost of time
        """
        super().__init__(env)
        self.action_cost = action_cost
        self.time_cost = time_cost
        self.noops_actions = noops_actions

    def step(self, action):
        """Steps through the environment with `action`."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # add time cost
        reward -= self.time_cost

        # add action cost
        if action not in self.noops_actions:
            reward -= self.action_cost

        return obs, reward, terminated, truncated, info


class BoxesWrapper(gym.Wrapper):
    """
    Wrapper which simulates the boxes environment.
    In addition to the usual movement actions, the agent has two special actions: open, which opens a box if it is closed and the agent is at the location of the box and has no effect otherwise (when a closed box is opened it transitions first to a half-open state for one time step and then automatically to an open state at the next time step regardless of the action by the agent), and eat, which has no effect unless the agent is at a box location, the box at that location is half-open, and there happens to be food (prey) in that box, in which case the agent consumes that food.
    An open box closes with probability 0.1 at every time step.6 A closed box always contains food. The prey always escapes when the box is open. Thus to consume food, the agent has to find a closed box, open it, and eat immediately in the next time step when the box is half-open. When the agent consumes food it feels satiated for one time step. The agent is hungry at all other time steps. The agent-environment interaction is not divided into trials or episodes. The agent’s observation is 6 dimensional: the x and y coordinates of the agent’s location, the agent’s hunger-status, the open/half-open/closed status of both boxes, as well the presence/absence of food in the square where the agent is located. These environments are Markovian because the agent senses the status of both boxes regardless of location and because closed boxes always contain food; hence each immediate observation is a state
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.blue_box_status = 0 # 0: closed, 1: half-open, 2: open
        self.red_box_status = 0 # 0: closed, 1: half-open, 2: open

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        print(obs)
        time.sleep(1)

        # add boxes logic here
        

        return obs, reward, terminated, truncated, info