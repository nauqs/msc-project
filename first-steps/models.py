import torch
import torch.nn as nn
import torch.nn.functional as F

_half_log_2pi_ = 0.5*torch.log(torch.tensor(2*torch.pi))

class DiscreteActorNet(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=64, device='cpu'):
        super(DiscreteActorNet, self).__init__()
        
        self.device = device

        self.l1 = nn.Linear(state_dim, hidden_dim).to(device)
        self.l2 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.l3 = nn.Linear(hidden_dim, action_dim).to(device)

    def forward(self, state):
        n = torch.tanh(self.l1(state))
        n = torch.tanh(self.l2(n))
        return n

    def get_action(self, state, action=None, softmax_dim=0):
        state = state.to(self.device)
        n = self.forward(state)
        prob = F.softmax(self.l3(n), dim=softmax_dim)
        dist = torch.distributions.Categorical(prob)
        if action is None:
            action = torch.multinomial(prob, 1)
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
        return action.detach(), policy.log_prob(action), policy.entropy()

    def save(self, filename='actor.pth'):
        torch.save(self.mean_net, filename)


class CriticNet(nn.Module):
    def __init__(self, state_dim, hidden_size, device='cpu'):
        super(CriticNet, self).__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        ).to(device)

    def forward(self, state):
        state = state.to(self.device)
        return self.net(state)

    def save(self, filename='critic.pth'):
        torch.save(self.net, filename)


class ConvBase(nn.Module):
    def __init__(self):
        super(ConvBase, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = x.unsqueeze(x.dim()-2)  # add channel dimension
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        return x

class MiniHackActorNet(nn.Module):

    def __init__(self, action_dim=8, cnn=False, device='cpu'):
        super(MiniHackActorNet, self).__init__()

        self.device = device
        self.cnn = cnn
        if self.cnn:
            self.conv_base = ConvBase()
            self.l1 = nn.Linear(64*2*9, 256)
        else:
            self.l1 = nn.Linear(21*79, 256)
        
        self.l2 = nn.Linear(256, 64)
        self.l3 = nn.Linear(64, action_dim)

    def forward(self, state):
        if self.cnn:
            state = self.conv_base(state).flatten(start_dim=state.dim()-2)
        else:
            state = state.flatten(start_dim=state.dim()-2)
        n = torch.tanh(self.l1(state))
        n = torch.tanh(self.l2(n))
        return n
    
    def get_action(self, state, action=None, softmax_dim=0):
        n = self.forward(state)
        prob = F.softmax(self.l3(n), dim=softmax_dim)
        dist = torch.distributions.Categorical(prob)
        if action is None:
            action = torch.multinomial(prob, 1)
        else:
            action = action.to(self.device)
            action = action.squeeze()
        entropy = dist.entropy()
        log_prob_action = dist.log_prob(action)
        return action.detach(), log_prob_action, entropy

    
    def save(self, filename='actor.pth'):
        torch.save(self, filename)



class MiniHackCriticNet(nn.Module):
    def __init__(self, cnn=False, device='cpu'):
        super(MiniHackCriticNet, self).__init__()

        self.device = device
        self.cnn = cnn
        if self.cnn:
            self.conv_base = ConvBase()
            self.l1 = nn.Linear(64*2*9, 256)
        else:
            self.l1 = nn.Linear(21*79, 256)
        
        self.l2 = nn.Linear(256, 64)
        self.l3 = nn.Linear(64, 1)

    def forward(self, state):
        if self.cnn:
            state = self.conv_base(state).flatten(start_dim=state.dim()-2)
        else:
            state = state.flatten(start_dim=state.dim()-2)
        n = torch.tanh(self.l1(state))
        n = torch.tanh(self.l2(n))
        n = self.l3(n)
        return n
    
    def save(self, filename='critic.pth'):
        torch.save(self, filename)

