import gymnasium as gym
import torch
import time
import os
from minigrid.wrappers import FullyObsWrapper

from utils import plot_logs
from ppo import PPO
from minigrid_trajectory import TrajectoryCollector

import sys
sys.path.append('../')
from models import MinigridActorNet, MinigridCriticNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### PPO CLIP HYPERPARAMS
OPTIMIZER_LR = 1e-3
MAX_TIMESTEPS = 2000000
TIMESTEPS_PER_BATCH = 2048
DISCOUNT_FACTOR = 0.99
TRACE_DECAY = 0.97## LOOK AT THIS
PPO_CLIP = 0.2
PPO_EPOCHS = 60
VALUE_EPOCHS = 5
PRINT_EVERY_N_TIMESTEPS = 10 # set to MAX_TIMESTEPS+1 
PLOT = True
ENTROPY_BETA = 0.001

# Env hyperparams
ENV_TYPE = "Empty"
ENV_SIZE = "5x5" #"5x5", "8x8"
ENV_STR = f'{ENV_TYPE}-{ENV_SIZE}'
ENV_NAME = f'MiniGrid-{ENV_STR}-v0'

# Define minigrid environment
env = gym.make(ENV_NAME)
env = FullyObsWrapper(env) # Make fully observable MDP

timestamp = round(time.time())//100

print("Device is", device)

# Initialize models
CONV_NETS = True
obs_dim = env.observation_space['image'].shape
obs_dim = (4, obs_dim[0], obs_dim[1])
action_dim = env.action_space.n
actor_net = MinigridActorNet(obs_dim=obs_dim, action_dim=action_dim, cnn=CONV_NETS, device=device)
critic_net = MinigridCriticNet(obs_dim=obs_dim, cnn=CONV_NETS, device=device)

# Initialize PPO class
agent = PPO(actor_net, critic_net, OPTIMIZER_LR, PPO_CLIP, PPO_EPOCHS, VALUE_EPOCHS, ENTROPY_BETA)

# Initialize Trajectory Collector
collector = TrajectoryCollector(env, agent, DISCOUNT_FACTOR, TRACE_DECAY)

os.makedirs(f'trained-models', exist_ok=True)
os.makedirs(f'figs', exist_ok=True)

for step in range(MAX_TIMESTEPS//TIMESTEPS_PER_BATCH):

    # Collect Trajectories
    batch, info = collector.collect_trajectories(TIMESTEPS_PER_BATCH)

    print("Batch collected")

    # Plot and print
    if PLOT: 
        plot_logs(info["timestep_history"], info["reward_history"], info["length_history"], step,
                       title=f'{ENV_NAME}',
                       save_path=f'figs/ppo_{ENV_STR}_{timestamp}.png')
    print("Timestep", info["timestep_history"][-1])
    # transform previous line to f string with .2f formatting
    print("Mean reward {:.2f} | Mean episode length {:.2f}".format(info["reward_history"][-1], info["length_history"][-1]))

    # Update actor and critic networks
    agent.update_actor(batch, save=True, save_path=f'trained-models/actor_{ENV_STR}_test.pt')
    agent.update_critic(batch, save=True, save_path=f'trained-models/critic_{ENV_STR}_test.pt')