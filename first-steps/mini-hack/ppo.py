# https://github.com/Kaixhin/spinning-up-basic/blob/master/ppo.py

import gym
import minihack
from nle import nethack
import tqdm
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
torch.autograd.set_detect_anomaly(True)

import sys
sys.path.append('../')
from models import MiniHackActorNet, MiniHackCriticNet
from models import DiscreteActorNet, CriticNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### PPO CLIP HYPERPARAMS
HIDDEN_SIZE = 32
OPTIMIZER_LR = 1e-3
MAX_TIMESTEPS = 2000000
TIMESTEPS_PER_BATCH = 2400
DISCOUNT_FACTOR = 0.99
TRACE_DECAY = 0.97## LOOK AT THIS
PPO_CLIP = 0.2
PPO_EPOCHS = 60
VALUE_EPOCHS = 5
PRINT_EVERY_N_TIMESTEPS = 10 # set to MAX_TIMESTEPS+1 
PLOT = True
ENTROPY_BETA = 0.001

# Minihack hyperparams
ROOM_TYPE = "Random" #"", "Random", "Dark", "Monster", "Trap, "Ultimate"
ROOM_SIZE = "5x5" #"5x5", "15x15"
room_str = f'{ROOM_TYPE+"-" if ROOM_TYPE!="" else ""}{ROOM_SIZE}'
ENV_NAME = f'MiniHack-Room-{room_str}-v0'
ACTION_KEYS = tuple(nethack.CompassDirection)  # Restrict actions to movement only
OBS_KEYS = ("glyphs",)
# Define room minihack environment
env = gym.make(ENV_NAME,
               actions=ACTION_KEYS,
               observation_keys=OBS_KEYS
)

# Plotting logs
timesteps, rewards, episode_lengths = [], [], []
timestamp = round(time.time())//1000

def plot_logs(timesteps, rewards, episode_lengths, smooth=True):
    # plot both rewards and episode lengths in same figure, but different scales
    alpha_non_smoothed, n_smooth = 1, 5
    smooth = smooth and len(rewards) > n_smooth
    if smooth:
        conv_smooth = np.ones((n_smooth,))/n_smooth
        smoothed_rewards = np.convolve(rewards, conv_smooth, mode='valid')
        smoothed_episode_lengths = np.convolve(episode_lengths, conv_smooth, mode='valid')
        alpha_non_smoothed = 0.2
    fig, ax1 = plt.subplots()
    ax1.plot(timesteps, rewards, 'b-', alpha=alpha_non_smoothed)
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Average reward', color='b')
    ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    ax2.plot(timesteps, episode_lengths, 'r-', alpha=alpha_non_smoothed)
    ax2.set_ylabel('Average episode length', color='r')
    ax2.tick_params('y', colors='r')
    if smooth:
        ax1.plot(timesteps[n_smooth-1:], smoothed_rewards, 'b-')
        ax2.plot(timesteps[n_smooth-1:], smoothed_episode_lengths, 'r-')
    plt.xlim(0, step+1)
    plt.title(f'{ENV_NAME}\nobs: {", ".join(OBS_KEYS)}')
    plt.savefig(f'figs/ppo_train_{room_str}_{timestamp}.png', dpi=200)
    plt.close()

# Initialise environment state
state = env.reset()
#state_tensor = torch.cat([torch.tensor(state[key].flatten(), dtype=torch.float32) for key in state.keys()])
state_tensor = torch.cat([torch.tensor(state[key], dtype=torch.float32) for key in state.keys()])
total_reward, done = 0, False
trajectories = []
obs_dim = state_tensor.shape

print(f"\nObservation space: (flattened shape {obs_dim})")
#for keys in env.observation_space:
  #print(f"  {keys}: {env.observation_space[keys]}")
print("Action space:", env.action_space, "\n")

# Define models
#actor_net = DiscreteActorNet(21*79, env.action_space.n, hidden_dim=HIDDEN_SIZE)
#critic_net = CriticNet(21*79, HIDDEN_SIZE)
actor_net = MiniHackActorNet(cnn=True)
critic_net = MiniHackCriticNet(cnn=True)
actor_optimiser = torch.optim.Adam(actor_net.parameters(), lr=OPTIMIZER_LR)
critic_optimiser = torch.optim.Adam(critic_net.parameters(), lr=OPTIMIZER_LR)
episode_length, batch_count = 0, 0


for step in range(MAX_TIMESTEPS):

  # Collect set of trajectories trajectories by running current policy
  action, log_prob_action, _ = actor_net.get_action(state_tensor)
  value = critic_net(state_tensor)
  next_state, reward, done, _ = env.step(action.item())
  total_reward += reward
  trajectories.append({'state': state_tensor.unsqueeze(0), 
                      'action': action.unsqueeze(0), 
                      'reward': torch.tensor([reward]), 
                      'done': torch.tensor([done], dtype=torch.float32), 
                      'log_prob_action': log_prob_action.unsqueeze(0), 
                      'old_log_prob_action': log_prob_action.unsqueeze(0).detach(), 
                      'value': value.unsqueeze(0)})
  state = next_state
  #state_tensor = torch.cat([torch.tensor(state[key].flatten(), dtype=torch.float32) for key in state.keys()])
  state_tensor = torch.cat([torch.tensor(state[key], dtype=torch.float32) for key in state.keys()])
  episode_length += 1

  if done: 
    # print step, reward and length of last episode
    if (step+1)%PRINT_EVERY_N_TIMESTEPS==0: pass#print(f"\n Step: {step+1} | Reward: {total_reward:.2f} | Episode length: {episode_length}")
    state, total_reward = env.reset(), 0
    #state_tensor = torch.cat([torch.tensor(state[key].flatten(), dtype=torch.float32) for key in state.keys()])
    state_tensor = torch.cat([torch.tensor(state[key], dtype=torch.float32) for key in state.keys()])
    episode_length = 0
    
    if len(trajectories) >= TIMESTEPS_PER_BATCH:
      batch_count += 1
      episodes_done = float(sum([trajectory['done'].item() for trajectory in trajectories]))
      print(f"\nTimestep: {step+1}, batch {batch_count}")
      average_reward = sum([trajectory['reward'].item() for trajectory in trajectories])/episodes_done
      print(f"Average reward: {average_reward:.2f} | Average episode length: {float(len(trajectories))/episodes_done:.2f}")
      if PLOT:
        timesteps.append(step+1)
        rewards.append(average_reward)
        episode_lengths.append(float(len(trajectories))/episodes_done)
        plot_logs(timesteps, rewards, episode_lengths)

      # Compute rewards-to-go R and advantage estimates based on the current value function V
      with torch.no_grad():
        reward_to_go, advantage, next_value = torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.])  # No bootstrapping needed for next value here as only updated at end of an episode
        for episode_step in trajectories[::-1]:
          reward_to_go = episode_step['reward'] + DISCOUNT_FACTOR * reward_to_go
          episode_step['reward_to_go'] = reward_to_go
          TD_error = episode_step['reward'] + DISCOUNT_FACTOR * next_value - episode_step['value']
          advantage = TD_error +  DISCOUNT_FACTOR * TRACE_DECAY * advantage
          episode_step['advantage'] = advantage
          next_value = episode_step['value']
          if episode_step["done"]:
            reward_to_go, next_value, advantage = torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.])
      
      batch = {k: torch.cat([trajectory[k] for trajectory in trajectories], dim=0) for k in trajectories[0].keys()}
      batch['advantage'] = (batch['advantage'] - batch['advantage'].mean()) / (batch['advantage'].std() + 1e-8)
      trajectories = []

      # Update the policy by maximising the PPO-Clip objective
      entropy = actor_net.get_action(batch['state'], action=batch['action'])[2]
      for epoch in range(PPO_EPOCHS):
        print("Update actor epoch", epoch)
        ratio = (batch['log_prob_action'] - batch['old_log_prob_action']).exp()
        clipped_ratio = torch.clamp(ratio, min=1 - PPO_CLIP, max=1 + PPO_CLIP)
        adv = batch['advantage']
        policy_loss = -torch.min(ratio * adv, clipped_ratio * adv).mean() - ENTROPY_BETA * entropy.mean()
        assert adv.shape == ratio.shape == clipped_ratio.shape
        actor_optimiser.zero_grad()
        policy_loss.backward()
        actor_optimiser.step()
        _, batch['log_prob_action'], entropy = actor_net.get_action(batch['state'], action=batch['action'].detach())
        batch['log_prob_action'] = batch['log_prob_action'].unsqueeze(-1)
        entropy = entropy.unsqueeze(-1) 

      # Fit value function by regression on mean-squared error
      for epoch in range(VALUE_EPOCHS):
        print("Update critic epoch", epoch)
        value_loss = (batch['value'] - batch['reward_to_go']).pow(2).mean()
        critic_optimiser.zero_grad()
        value_loss.backward(retain_graph=True)
        critic_optimiser.step()
        batch['value'] = critic_net(batch['state'])

      # Save the networks
      actor_net.save(filename=f'trained-models/actor_{room_str}_{timestamp}.pt')
      critic_net.save(filename=f'trained-models/critic_{room_str}_{timestamp}.pt')