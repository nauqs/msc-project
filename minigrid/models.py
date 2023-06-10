import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class BasicAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        print("input shape for agent:", np.array(envs.single_observation_space['image'].shape).prod())
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space['image'].shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space['image'].shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    
    def save(self, file_path="trained-models/actor.pth"):
        torch.save(self, file_path)


class ConvBase(nn.Module):
    def __init__(self, n_channels=4):
        super(ConvBase, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=x.dim()-3)
        return x


class MiniGridAgent(nn.Module):
    def __init__(self, obs_dim, action_dim, n_channels=4):
        super(MiniGridAgent, self).__init__()
        
        # Convolutional base
        self.conv = ConvBase(n_channels=n_channels)
        self.conv_output_size = self.conv(torch.randn(1, *obs_dim)).shape[-1]

        # Critic head
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.conv_output_size, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Actor head
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.conv_output_size, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )

    def get_value(self, x):
        x = self.conv(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = self.conv(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def save(self, file_path="trained-models/actor.pth"):
        torch.save(self, file_path)
