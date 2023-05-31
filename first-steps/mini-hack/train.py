import gym
import minihack
from nle import nethack
import torch
import torch.nn as nn
import numpy as np
import time

from utils import plot_logs
from ppo import PPOAgent
from trajectory import TrajectoryCollector

import sys
sys.path.append('../')
from models import MiniHackActorNet, MiniHackCriticNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### PPO CLIP HYPERPARAMS
HIDDEN_SIZE = 32
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
ENTROPY_BETA = 0.01

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
timestamp = round(time.time())//100

# Initialize models
actor_net = MiniHackActorNet(cnn=True)
critic_net = MiniHackCriticNet(cnn=True)

# Initialize PPO Agent
agent = PPOAgent(actor_net, critic_net, OPTIMIZER_LR, PPO_CLIP, PPO_EPOCHS, VALUE_EPOCHS, ENTROPY_BETA)

# Initialize Trajectory Collector
collector = TrajectoryCollector(env, agent, DISCOUNT_FACTOR, TRACE_DECAY)

for step in range(MAX_TIMESTEPS):

    # Collect Trajectories
    batch, info = collector.collect_trajectories(TIMESTEPS_PER_BATCH)

    # Plot and print
    if PLOT: plot_logs(info["timestep_history"], info["reward_history"], info["length_history"], step,
                       title=f'{ENV_NAME}\nobs: {", ".join(OBS_KEYS)}',
                       save_path=f'figs/ppo_{room_str}_{timestamp}.png')
    print("Timestep", info["timestep_history"][-1])
    # transform previous line to f string with .2f formatting
    print("Mean reward {:.2f} | Mean episode length {:.2f}".format(info["reward_history"][-1], info["length_history"][-1]))

    # Update actor and critic networks
    agent.update_actor(batch, save=True, save_path=f'trained-models/actor_{room_str}_test.pt')
    agent.update_critic(batch, save=True, save_path=f'trained-models/critic_{room_str}_test.pt')