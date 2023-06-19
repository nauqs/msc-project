import gymnasium as gym
import torch
import time
import os
import argparse
import random
from datetime import datetime
from minigrid.wrappers import FullyObsWrapper
from distutils.util import strtobool
import wandb

import numpy as np
import torch
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter

from models import MiniGridAgent
from storage import TrajectoryCollector
from ppo import PPO
from utils import plot_logs, get_state_tensor, ActionCostWrapper

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="",
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--plot", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to plot metrics and save")
    parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to print metrics and training logs")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--wandb", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use wandb to log metrics")


    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default=f'MiniGrid-Empty-8x8-v0',
        help="the id of the environment")
    parser.add_argument("--fully-obs", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use the fully observable wrapper")
    parser.add_argument("--action-cost", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use the action cost wrapper")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=256,
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

def make_env(env_id, fully_obs, action_cost, seed, idx, capture_video, run_name):
    def thunk():
        if env_id == "MiniGrid-FourRooms-v0":
            env = gym.make(env_id, max_steps=1024)
        else:
            env = gym.make(env_id)
        if fully_obs: env = FullyObsWrapper(env)
        if action_cost: env = ActionCostWrapper(env)
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
print(args)
envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_id, 
              args.fully_obs, 
              args.action_cost,
              args.seed+i, 
              i, 
              args.capture_video, 
              run_name)
            for i in range(args.num_envs)]
)

# Get dimension of a single transformed observation
obs_dim = get_state_tensor(envs.reset()[0])[0].shape

# Define agent
agent = MiniGridAgent(obs_dim, envs.single_action_space.n, n_channels=4).to(device)

# Define storage and ppo objects
storage = TrajectoryCollector(envs, obs_dim, agent, args, device)
ppo = PPO(agent, args, device)

os.makedirs(f'trained-models/{args.env_id}', exist_ok=True)
os.makedirs(f'figs/{args.env_id}', exist_ok=True)

if args.wandb:
    env_type = args.env_id.split('-')[1]
    wandb.init(project="action-cost-experiments", 
               entity="nauqs",
               name=run_name, 
               config=args)
    wandb.config.update({"env_type": env_type})

timestep_history, return_history, length_history = [], [], []

# Run training algorithm
for update in range(1, num_updates+1):

    # Collect trajectories
    batch, stats = storage.collect_trajectories()

    # Update PPO agents (actor and critic)
    # TODO: return info (actor/critic loss, KL...)
    # TODO: lr annealing / schedule?
    ppo.update_ppo_agent(batch, save_path=f'trained-models/{args.env_id}/actor_{run_name}.pth')

    # Unifinished episodes
    if len(stats['episode_returns'])==0: 
        stats['episode_returns'] = np.array([0])
    if len(stats['episode_lengths'])==0:
        stats['episode_lengths'] = np.array([args.num_steps])
    if len(stats['episode_timesteps'])==0:
        stats['episode_timesteps'] = np.array([stats['initial_timestep']])
        
    # Print stats
    if args.verbose:
        print(f"Timestep: {stats['initial_timestep']}")
        if len(stats['episode_returns'])>0:
            # print stats with mean and std and 3 decimals
            print(f"Episodic return: {stats['episode_returns'].mean():.3f}±{stats['episode_returns'].std():.3f}")
            print(f"Episodic length: {stats['episode_lengths'].mean():.3f}±{stats['episode_lengths'].std():.3f}")

    # Plot stats
    if args.plot:

        timestep_history.append(stats['initial_timestep'])
        return_history.append((stats['episode_returns'].mean(), stats['episode_returns'].std()))
        length_history.append((stats['episode_lengths'].mean(), stats['episode_lengths'].std()))

        plot_logs(timestep_history, return_history, length_history, update,
            smooth=True,
            title=f'{args.env_id}',
            save_path=f'figs/{args.env_id}/ppo_{args.env_id}_{run_name}.png')
        
    # Log metrics to wandb
    if args.wandb:
        wandb.log({
            "average_return": stats['episode_returns'].mean(),
            "average_length": stats['episode_lengths'].mean(),
            "timestep": stats['initial_timestep'],
        })
        for i in range(len(stats['episode_returns'])):
            wandb.log({
                "episode_timestep": stats['episode_timesteps'][i],
                "episode_return": stats['episode_returns'][i],
                "episode_length": stats['episode_lengths'][i],
            })

