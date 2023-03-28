# https://github.com/Kaixhin/spinning-up-basic/blob/master/ppo.py

import gym
import minihack
from nle import nethack
import tqdm
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# import ActorNet from ../models.py
import sys
sys.path.append('../')
from models import DiscreteActorNet, CriticNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### PPO CLIP HYPERPARAMS
HIDDEN_SIZE = 32
OPTIMIZER_LR = 1e-3
MAX_TIMESTEPS = 200000
TIMESTEPS_PER_BATCH = 2400
DISCOUNT_FACTOR = 0.99
TRACE_DECAY = 0.97## LOOK AT THIS
PPO_CLIP = 0.2
PPO_EPOCHS = 60
VALUE_EPOCHS = 5
PRINT_EVERY_N_TIMESTEPS = 1 # set to MAX_TIMESTEPS+1 

# Restrict actions to movement and navigation
MOVE_ACTIONS = tuple(nethack.CompassDirection)

# Define room minihack environment
env = gym.make('MiniHack-Room-15x15-v0',
               actions=MOVE_ACTIONS,
               observation_keys=("glyphs", "chars", "colors")
)

# Initialise environment state
state = env.reset()
state_tensor = torch.cat([torch.tensor(state[key].flatten(), dtype=torch.float32) for key in state.keys()])
total_reward, done = 0, False
trajectories = []
obs_dim = state_tensor.shape[0]

print(f"\nObservation space: (flattened shape {obs_dim})")
for keys in env.observation_space:
  print(f"  {keys}: {env.observation_space[keys]}")
print("Action space:", env.action_space, "\n")

# Define models
actor_net = DiscreteActorNet(obs_dim, env.action_space.n, hidden_dim=HIDDEN_SIZE)
critic_net = CriticNet(obs_dim, HIDDEN_SIZE)
actor_optimiser = torch.optim.Adam(actor_net.parameters(), lr=OPTIMIZER_LR)
critic_optimiser = torch.optim.Adam(critic_net.parameters(), lr=OPTIMIZER_LR)
episode_length = 0

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
  state_tensor = torch.cat([torch.tensor(state[key].flatten(), dtype=torch.float32) for key in state.keys()])
  episode_length += 1

  if done: 
    # print step, reward and length of last episode
    if (step+1)%PRINT_EVERY_N_TIMESTEPS==0: print(f"\n Step: {step+1} | Reward: {total_reward:.2f} | Episode length: {episode_length}")
    state, total_reward = env.reset(), 0
    state_tensor = torch.cat([torch.tensor(state[key].flatten(), dtype=torch.float32) for key in state.keys()])
    episode_length = 0
    
    if len(trajectories) >= TIMESTEPS_PER_BATCH:
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
      for _ in range(PPO_EPOCHS):
        ratio = (batch['log_prob_action'] - batch['old_log_prob_action']).exp()
        clipped_ratio = torch.clamp(ratio, min=1 - PPO_CLIP, max=1 + PPO_CLIP)
        adv = batch['advantage']
        policy_loss = -torch.min(ratio * adv, clipped_ratio * adv).mean()
        actor_optimiser.zero_grad()
        policy_loss.backward()
        actor_optimiser.step()
        batch['log_prob_action'] = actor_net.get_action(batch['state'], action=batch['action'].detach())[1]

      # Fit value function by regression on mean-squared error
      for _ in range(VALUE_EPOCHS):
        value_loss = (batch['value'] - batch['reward_to_go']).pow(2).mean()
        critic_optimiser.zero_grad()
        value_loss.backward()
        critic_optimiser.step()
        batch['value'] = critic_net(batch['state'])

# Save the networks
actor_net.save()
critic_net.save()