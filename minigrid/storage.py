import torch
import numpy as np
import time

MAX_PATIENCE = 1000

def get_state_tensor(state, flat=True):

    if flat:
        return torch.tensor(state['image'], dtype=torch.float32)

    else:

        image = torch.tensor(state['image'], dtype=torch.float32)
        direction = state['direction']

        image = image.permute(2,0,1)
        direction_channel = torch.full_like(image[0], direction)
        state_tensor = torch.cat((image, direction_channel.unsqueeze(0)), dim=0)

        return state_tensor


class TrajectoryCollector:
    def __init__(self, envs, agent, args, device):
        self.envs = envs
        self.agent = agent
        self.args = args
        self.device = device

        # ALGO Logic: Storage setup
        self.obs = torch.zeros((self.args.num_steps, self.args.num_envs) + envs.single_observation_space['image'].shape).to(device)
        self.actions = torch.zeros((self.args.num_steps, self.args.num_envs) + envs.single_action_space.shape).to(device)
        self.logprobs = torch.zeros((self.args.num_steps, self.args.num_envs)).to(device)
        self.rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(device)
        self.dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(device)
        self.values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(device)
        self.global_step = 0

    def collect_trajectories(self):

        state = self.envs.reset()[0]
        next_obs = torch.Tensor(state['image']).to(self.device)
        next_done = torch.zeros(self.args.num_envs).to(self.device)

        for step in range(0, self.args.num_steps):
            self.global_step += 1 * self.args.num_envs
            self.obs[step] = next_obs
            self.dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(next_obs.flatten(start_dim=1))
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob

            # execute env and log data.
            next_obs, reward, truncated, terminated, info = self.envs.step(action.cpu().numpy())
            next_obs = next_obs['image']
            done = truncated | terminated
            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            next_obs = torch.Tensor(next_obs).to(self.device)
            next_done = torch.Tensor(done).to(self.device)

            for item in info:
                for item in info["final_info"]:
                    if item is not None:
                        if "episode" in item.keys():
                            #print(f"global_step={self.global_step}, episodic_return={item['episode']['r']}")
                            break

        with torch.no_grad():
            next_value = self.agent.get_value(next_obs.flatten(start_dim=1)).reshape(1, -1)
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

        batch = {'obs': self.obs.reshape((-1,) + self.envs.single_observation_space['image'].shape),
                    'log_probs': self.logprobs.reshape(-1),
                    'actions': self.actions.reshape((-1,) + self.envs.single_action_space.shape),
                    'advantages': advantages.reshape(-1),
                    'returns': returns.reshape(-1),
                    'values': self.values.reshape(-1)}
        
        print("return", returns.mean())
        
        return batch