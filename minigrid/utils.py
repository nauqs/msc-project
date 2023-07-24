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
    
    def __init__(self, env, action_cost=0.01, time_cost=0.01, final_reward_penalty=False, noops_actions=[6]):
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
        self.final_penalty = final_reward_penalty

    def step(self, action):
        """Steps through the environment with `action`."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated and not self.final_penalty:
            reward = 1

        # add time cost
        reward -= self.time_cost

        # add action cost
        if action not in self.noops_actions:
            reward -= self.action_cost

        return obs, reward, terminated, truncated, info


class ContEnergyWrapper(gym.Wrapper):
    """
    Wrapper which converts episodic setttings into
     continuous with the energy budget framework 
     """
    
    def __init__(self, env, refuel_goal, initial_energy, time_bonus=0.1, goal_reward=0):

        super().__init__(env)
        self.refuel_goal = refuel_goal
        self.initial_energy = initial_energy
        self.time_bonus = time_bonus
        self.goal_reward = goal_reward
        self.initial_refuel_goal = refuel_goal

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.energy = self.initial_energy
        self.refuel_goal = self.initial_refuel_goal
        return obs
    
    def step(self, action):
        """
        When energy reaches 0, episode terminates
        When agent reaches goal, episode doesn't terminate, energy gets refilled
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.energy -= 1

        if terminated and reward > 0:
            self.energy += self.refuel_goal
            reward = self.goal_reward
            self.refuel_goal -= 1
            terminated = False
        elif self.energy <= 0:
            terminated = True
            reward = -self.initial_energy * self.time_bonus
        
        reward += self.time_bonus

        return obs, reward, terminated, truncated, info
