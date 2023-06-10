import gymnasium as gym
import torch
import time
import os
import argparse
import random
from datetime import datetime
from minigrid.wrappers import FullyObsWrapper
from distutils.util import strtobool

import numpy as np
import torch
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter

from models import MiniGridAgent
from storage import TrajectoryCollector
from ppo import PPO
from utils import plot_logs, get_state_tensor

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="",
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--plot", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to plot metrics and save")
    parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to print metrics and training logs")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")


    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default=f'MiniGrid-Empty-16x16-v0',
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = FullyObsWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

args = parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

timestamp = datetime.now().strftime("%m%d_%H%M%S")
if args.exp_name == "": run_name = timestamp
else: run_name = args.exp_name

num_updates = args.total_timesteps // args.batch_size

# Set up vectorised environments
print(args.env_id, args.seed, 0, args.capture_video, run_name) 
envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_id, args.seed+i, i, args.capture_video, run_name) for i in range(args.num_envs)]
)

# Get dimension of a single transformed observation
obs_dim = get_state_tensor(envs.reset()[0])[0].shape

# Define agent
agent = MiniGridAgent(obs_dim, envs.single_action_space.n, n_channels=4).to(device)

# Define storage and ppo objects
storage = TrajectoryCollector(envs, obs_dim, agent, args, device)
ppo = PPO(agent, args, device)

os.makedirs(f'trained-models', exist_ok=True)
os.makedirs(f'figs', exist_ok=True)

timestep_history, return_history, length_history = [], [], []

# Run training algorithm
for update in range(1, num_updates+1):

    # Collect trajectories
    batch, stats = storage.collect_trajectories()

    # TODO: add std for error margins
    timestep_history.append(stats['initial_timestep'])
    return_history.append(stats['episode_returns'].mean())
    length_history.append(stats['episode_lengths'].mean())

    # Update PPO agents (actor and critic)
    # TODO: return info (actor/critic loss, KL...)
    # TODO: lr annealing / schedule?
    ppo.update_ppo_agent(batch)

    if args.plot:
        plot_logs(timestep_history, return_history, length_history, update,
            smooth=True,
            title=f'{args.env_id}',
            save_path=f'figs/ppo_{args.env_id}_{run_name}.png')
