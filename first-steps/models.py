import torch
import torch.nn as nn
import torch.nn.functional as F

_half_log_2pi_ = 0.5*torch.log(torch.tensor(2*torch.pi))

class DiscreteActorNet(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DiscreteActorNet, self).__init__()
        
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        n = torch.tanh(self.l1(state))
        n = torch.tanh(self.l2(n))
        return n

    def get_action(self, state, action=None, softmax_dim=0):
        n = self.forward(state)
        prob = F.softmax(self.l3(n), dim=softmax_dim)
        dist = torch.distributions.Categorical(prob)
        if action is None:
            action = torch.multinomial(prob, 1)
            #action = dist.sample()
        else:
            action = action.squeeze()
        entropy = dist.entropy()
        log_prob_action = dist.log_prob(action)
        return action.detach(), log_prob_action, entropy

    def save(self, filename='actor.pth'):
        torch.save(self, filename)


class ContActorNet(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=32, initial_log_std=0.):
        super(ContActorNet, self).__init__()
        
        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        self.log_std = nn.Parameter(torch.tensor([initial_log_std]))

    def forward(self, state):
        mu = self.mean_net(state)
        return mu, self.log_std

    def get_action(self, state, action=None, exploitation=False):
        mean, log_std = self.forward(state)
        if exploitation: return mean.detach()
        policy = torch.distributions.Normal(mean, log_std.exp())
        if action is None:
            action = policy.rsample()
            #if abs(state[0])<0.01: print(mean, log_std.exp(), action)
        return action.detach(), policy.log_prob(action), policy.entropy()

    def save(self, filename='actor.pth'):
        torch.save(self.mean_net, filename)


class CriticNet(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super(CriticNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        return self.net(state)

    def save(self, filename='critic.pth'):
        torch.save(self.net, filename)

class ConvBase(nn.Module):
    def __init__(self):
        super(ConvBase, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1) 

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(0)  # add batch dimension
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)
        return x
    

class MiniHackCriticNet(nn.Module):
    def __init__(self, conv_output_dim=32, hidden_dim=64):
        super(MiniHackCriticNet, self).__init__()

        self.conv_base = ConvBase()
        self.l1 = nn.Linear(conv_output_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1) 

    def forward(self, state):
        n = self.conv_base(state)
        n = torch.tanh(self.l1(n))
        n = torch.tanh(self.l2(n))
        n = self.l3(n) 
        return n

    def save(self, filename='critic.pth'):
        torch.save(self, filename)


class MiniHackActorNet(nn.Module):
    def __init__(self, conv_output_dim=32, action_dim=8, hidden_dim=64):
        super(MiniHackActorNet, self).__init__()

        self.conv_base = ConvBase()
        self.l1 = nn.Linear(conv_output_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        n = self.conv_base(state)
        n = torch.tanh(self.l1(n))
        n = torch.tanh(self.l2(n))
        return n

    def get_action(self, state, action=None, softmax_dim=0):
        n = self.forward(state)
        prob = F.softmax(self.l3(n), dim=softmax_dim)
        dist = torch.distributions.Categorical(prob)
        if action is None:
            action = dist.sample()
        else:
            action = action.squeeze()
        entropy = dist.entropy()
        log_prob_action = dist.log_prob(action)
        return action.detach(), log_prob_action, entropy

    def save(self, filename='actor.pth'):
        torch.save(self, filename)
