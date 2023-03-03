import torch
import torch.nn as nn

class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=(64, 64)):
        super(ActorNet, self).__init__()
        
        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim[0]),
            nn.Tanh(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.Tanh(),
            nn.Linear(hidden_dim[1], action_dim),
        )
        
        self.std_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], action_dim),
            nn.Softplus(),
        )

    def forward(self, state):
        state = state
        mean = self.mean_net(state)
        std = self.std_net(state)
        return mean, std

    def get_action(self, state, action=None):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
          action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist.entropy()
