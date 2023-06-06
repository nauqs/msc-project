# ppo.py
import torch
import torch.nn as nn

class PPO(nn.Module):
    def __init__(self, actor_net, critic_net, optimizer_lr, ppo_clip, ppo_epochs, value_epochs, entropy_beta):
        super(PPO, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actor_net = actor_net.to(self.device)
        self.critic_net = critic_net.to(self.device)
        self.ppo_clip = ppo_clip
        self.ppo_epochs = ppo_epochs
        self.value_epochs = value_epochs
        self.entropy_beta = entropy_beta
        self.actor_optimiser = torch.optim.Adam(self.actor_net.parameters(), lr=optimizer_lr)
        self.critic_optimiser = torch.optim.Adam(self.critic_net.parameters(), lr=optimizer_lr)

    def update_actor(self, batch, save=False, verbose=True, save_path="saved-models/actor.pth"):
        # Update the policy by maximising the PPO-Clip objective
        total_loss = 0
        entropy = self.actor_net.get_action(batch['state'], action=batch['action'])[2]
        for epoch in range(self.ppo_epochs):
            ratio = (batch['log_prob_action'].to(self.device) - batch['old_log_prob_action'].to(self.device)).exp()
            clipped_ratio = torch.clamp(ratio, min=1 - self.ppo_clip, max=1 + self.ppo_clip)
            adv = batch['advantage'].to(self.device)
            policy_loss = -torch.min(ratio * adv, clipped_ratio * adv).mean() - self.entropy_beta * entropy.mean()
            assert adv.shape == ratio.shape == clipped_ratio.shape
            self.actor_optimiser.zero_grad()
            policy_loss.backward()
            self.actor_optimiser.step()
            _, batch['log_prob_action'], entropy = self.actor_net.get_action(batch['state'], action=batch['action'].detach().to(self.device))
            batch['log_prob_action'] = batch['log_prob_action'].unsqueeze(-1)
            entropy = entropy.unsqueeze(-1)
            total_loss += policy_loss
        if save:
            self.actor_net.save(filename=save_path)
        if verbose:
            print(f"Actor loss {total_loss/self.ppo_epochs:.3f}")
        return total_loss.detach().item()  / self.ppo_epochs

    def update_critic(self, batch, save=False, verbose=True, save_path="saved-models/critic.pth"):
        # Fit value function by regression on mean-squared error
        total_loss = 0
        for epoch in range(self.value_epochs):
            value_loss = (batch['value'].to(self.device) - batch['reward_to_go'].to(self.device)).pow(2).mean()
            self.critic_optimiser.zero_grad()
            value_loss.backward(retain_graph=True)
            self.critic_optimiser.step()
            batch['value'] = self.critic_net(batch['state'])
            total_loss += value_loss
        if save:
            self.critic_net.save(filename=save_path)
        if verbose:
            print(f"Critic loss {total_loss/self.value_epochs:.3f}")
        return total_loss.detach().item() / self.value_epochs


