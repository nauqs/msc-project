import torch
import numpy as np
import time
from utils import get_state_tensor

MAX_PATIENCE = 1000

class TrajectoryCollector:
    def __init__(self, envs, obs_dim, agent, args, device):
        self.envs = envs
        self.agent = agent
        self.args = args
        self.device = device
        self.obs_dim = tuple(obs_dim)

        self.obs = torch.zeros((self.args.num_steps, self.args.num_envs) + self.obs_dim).to(device)
        self.actions = torch.zeros((self.args.num_steps, self.args.num_envs) + envs.single_action_space.shape).to(device)
        self.logprobs = torch.zeros((self.args.num_steps, self.args.num_envs)).to(device)
        self.rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(device)
        self.dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(device)
        self.values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(device)
        self.global_step = 0

    def collect_trajectories(self):
        
        stats = {'initial_timestep': self.global_step}
        episode_returns, episode_lengths = [], []
        state = self.envs.reset()[0]
        next_obs = get_state_tensor(state).to(self.device)
        next_done = torch.zeros(self.args.num_envs).to(self.device)

        for step in range(0, self.args.num_steps):
            self.global_step += 1 * self.args.num_envs
            self.obs[step] = next_obs
            self.dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob

            next_state, reward, truncated, terminated, info = self.envs.step(action.cpu().numpy())
            next_obs = get_state_tensor(next_state)
            done = truncated | terminated
            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            next_obs = next_obs.to(self.device)
            next_done = torch.Tensor(done).to(self.device)

            # info is a dict with final_info and final_observation for the envs which reached a terminal state
            # everything else is None in the others
            if 'final_info' in info:
                for env_final_info in info['final_info']:
                    if env_final_info is not None:
                        episode_returns.append(env_final_info['episode']['r'].item())
                        episode_lengths.append(env_final_info['episode']['l'].item())

        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.args.num_steps)):
                if t == self.args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.args.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values

        batch = {'obs': self.obs.reshape((-1,) + self.obs_dim),
                    'log_probs': self.logprobs.reshape(-1),
                    'actions': self.actions.reshape((-1,) + self.envs.single_action_space.shape),
                    'advantages': advantages.reshape(-1),
                    'returns': returns.reshape(-1),
                    'values': self.values.reshape(-1)}
        
        stats['episode_returns'] = np.array(episode_returns)
        stats['episode_lengths'] = np.array(episode_lengths)
        stats['final_timestep'] = self.global_step
        
        return batch, stats