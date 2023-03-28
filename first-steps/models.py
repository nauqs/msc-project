import torch
import torch.nn as nn
import torch.nn.functional as F

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
        if action is None:
            action = torch.multinomial(prob, 1)
        # return action, log prob of action, entropy
        entropy = -torch.sum(prob*torch.log(prob), dim=softmax_dim)
        return action.detach(), torch.log(prob.gather(softmax_dim, action)), entropy

    def save(self, filename='actor.pth'):
        torch.save(self, filename)


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