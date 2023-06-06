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

    def get_action(self, state, action=None, softmax_dim=0):  # TODO: check softmax dim (possible bug!!)
        state = state.to(self.device)
        n = self.forward(state)
        prob = F.softmax(self.l3(n), dim=softmax_dim)  # TODO: check softmax dim (possible bug!!)
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
    
class ConvBaseMinigrid(nn.Module):
    def __init__(self, n_channels=4):
        super(ConvBaseMinigrid, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=x.dim()-3)
        return x
    

class MinigridActorNet(nn.Module):

    def __init__(self, obs_dim, action_dim, cnn=False, device='cpu'):
        super(MinigridActorNet, self).__init__()

        self.hs = [512, 64]
        self.device = device
        self.cnn = cnn
        self.obs_dim = torch.tensor(obs_dim)
        if self.cnn:
            self.conv_base = ConvBaseMinigrid(n_channels=4)
            self.cnn_output_size = self.conv_base(torch.randn(1, *obs_dim)).shape[-1]
            self.l1 = nn.Linear(self.cnn_output_size, self.hs[0])
            print(f"CNN net (cnn output size: {self.cnn_output_size})")
        else:
            self.l1 = nn.Linear(self.obs_dim.prod().item(), self.hs[0])
            print(f"Fully connected net (input size: {self.obs_dim})")
        
        self.l2 = nn.Linear(self.hs[0], self.hs[1])
        self.l3 = nn.Linear(self.hs[1], action_dim)

    def forward(self, state):
        if self.cnn:
            state = self.conv_base(state)
        else:
            state = state.flatten(start_dim=state.dim()-3)
        n = torch.tanh(self.l1(state))
        n = torch.tanh(self.l2(n))
        return n
    
    def get_action(self, state, action=None, softmax_dim=-1):
        n = self.forward(state)
        prob = F.softmax(self.l3(n), dim=-1)
        dist = torch.distributions.Categorical(prob)

        # TODO: debug (frequency of prints)
        print_prob = 0.000
        noise_print = torch.rand(1).item() < print_prob
        if noise_print and state.dim()==3: print(state[0,1:-1,1:-1], prob, state[3,0,0])

        if action is None:
            action = torch.multinomial(prob, 1)
            if noise_print: print("action", action)
        else:
            action = action.to(self.device)
            action = action.squeeze()
        entropy = dist.entropy()
        log_prob_action = dist.log_prob(action)
        return action.detach(), log_prob_action, entropy

    
    def save(self, filename='actor.pth'):
        torch.save(self, filename)


class MinigridCriticNet(nn.Module):
    def __init__(self, obs_dim, cnn=False, device='cpu'):
        super(MinigridCriticNet, self).__init__()

        self.hs = [512, 64]
        self.device = device
        self.cnn = cnn
        self.obs_dim = torch.tensor(obs_dim)
        
        if self.cnn:
            self.conv_base = ConvBaseMinigrid(n_channels=4)
            self.cnn_output_size = self.conv_base(torch.randn(1, *obs_dim)).shape[-1]
            self.l1 = nn.Linear(self.cnn_output_size, self.hs[0])
        else:
            self.l1 = nn.Linear(self.obs_dim.prod().item(), self.hs[0])
        
        self.l2 = nn.Linear(self.hs[0], self.hs[1])
        self.l3 = nn.Linear(self.hs[1], 1)

    def forward(self, state):
        if self.cnn:
            state = self.conv_base(state)
        else:
            state = state.flatten(start_dim=state.dim()-3)
        n = torch.tanh(self.l1(state))
        n = torch.tanh(self.l2(n))
        n = self.l3(n)
        return n
    
    def save(self, filename='critic.pth'):
        torch.save(self, filename)

class MiniHackActorNet(nn.Module):

    def __init__(self, action_dim=8, cnn=False, device='cpu'):
        super(MiniHackActorNet, self).__init__()

        self.device = device
        self.cnn = cnn
        if self.cnn:
            self.conv_base = ConvBase()
            self.l1 = nn.Linear(3040, 256)
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
        prob = F.softmax(self.l3(n), dim=softmax_dim) # TODO: check softmax dim (possible bug!!)
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
            self.l1 = nn.Linear(3040, 256)
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
