import gymnasium as gym
import torch
import time
import os
import argparse
from minigrid.wrappers import FullyObsWrapper

from minigrid_trajectory import TrajectoryCollector

import sys
sys.path.append('../')
from utils import plot_logs
from ppo import PPO
from models import MinigridActorNet, MinigridCriticNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parse arguments
parser = argparse.ArgumentParser(description='Train PPO on MiniGrid')
parser.add_argument('--env_type', type=str, default='Empty', help='Environment type')
parser.add_argument('--env_size', type=str, default='5x5', help='Environment size')
parser.add_argument('--conv_nets', type=bool, default=True, help='Use convolutional networks')
parser.add_argument('--max_timesteps', type=int, default=100000, help='Maximum number of timesteps')
args = parser.parse_args()

### PPO CLIP HYPERPARAMS
OPTIMIZER_LR = 1e-3
MAX_TIMESTEPS = args.max_timesteps
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
ENV_TYPE = args.env_type # Empty, Empty-Random
ENV_SIZE = args.env_size #"5x5", "6x6", "8x8", "16x16"
ENV_STR = f'{ENV_TYPE}-{ENV_SIZE}'
ENV_NAME = f'MiniGrid-{ENV_STR}-v0'

# Define minigrid environment
env = gym.make(ENV_NAME)
env = FullyObsWrapper(env) # Make fully observable MDP

timestamp = round(time.time())//100

print("Device is", device)

# Initialize models
CONV_NETS = args.conv_nets
obs_dim = env.observation_space['image'].shape
obs_dim = (4, obs_dim[0], obs_dim[1])
action_dim = 3#env.action_space.n
actor_net = MinigridActorNet(obs_dim=obs_dim, action_dim=action_dim, cnn=CONV_NETS, device=device)
critic_net = MinigridCriticNet(obs_dim=obs_dim, cnn=CONV_NETS, device=device)

# Initialize PPO class
agent = PPO(actor_net, critic_net, OPTIMIZER_LR, PPO_CLIP, PPO_EPOCHS, VALUE_EPOCHS, ENTROPY_BETA)

# Initialize Trajectory Collector
collector = TrajectoryCollector(env, agent, DISCOUNT_FACTOR, TRACE_DECAY)

os.makedirs(f'trained-models', exist_ok=True)
os.makedirs(f'figs', exist_ok=True)

actor_loss, critic_loss = [], []

for step in range(MAX_TIMESTEPS//TIMESTEPS_PER_BATCH):

    # Collect Trajectories
    batch, info = collector.collect_trajectories(TIMESTEPS_PER_BATCH)

    print("Timestep", info["timestep_history"][-1])
    print("Mean reward {:.2f} | Mean episode length {:.2f}".format(info["reward_history"][-1], info["length_history"][-1]))

    # Update actor and critic networks
    actor_loss.append(agent.update_actor(batch, save=True, save_path=f'trained-models/actor_{ENV_STR}_test.pt'))
    critic_loss.append(agent.update_critic(batch, save=True, save_path=f'trained-models/critic_{ENV_STR}_test.pt'))

    # Plot and print
    if PLOT: 
        plot_logs(info["timestep_history"], info["reward_history"], info["length_history"], 
                    actor_loss,  critic_loss, step,
                    smooth=True,
                    title=f'{ENV_NAME}',
                    save_path=f'figs/ppo_{ENV_STR}_{timestamp}.png')


