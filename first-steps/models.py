import torch
import torch.nn as nn

_half_log_2pi_ = 0.5*torch.log(torch.tensor(2*torch.pi))

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
            nn.Tanh(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.Tanh(),
            nn.Linear(hidden_dim[1], action_dim)
        )

    def forward(self, state):
        state = state
        mu = self.mean_net(state)
        log_sigma = self.std_net(state)
        return mu, log_sigma

    def get_action(self, state, action=None):
        mean, log_std = self.forward(state)
        policy = torch.distributions.Normal(mean, log_std.exp())
        if action is None:
            action = policy.rsample()
        return action.detach(), policy.log_prob(action), policy.entropy()

    def get_action_rep(self, state, action=None):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        # Reparametrization trick
        z = torch.normal(mean=1.,std=1.,size=(1,))
        if action is None:
            action = torch.tanh(mean + std * z)
        log_prob = -0.5 * ((action - mean) / std)**2 #- _half_log_2pi_ - log_std
        entropy = 0.5 + _half_log_2pi_ + log_std
        return action.detach(), log_prob, entropy
