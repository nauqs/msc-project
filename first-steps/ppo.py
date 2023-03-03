import gym #gym===0.25.2
import tqdm
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from models import ActorNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### PPO CLIP HYPERPARAMS
NET_HIDDEN = (64, 64)
N_EPOCHS = 300
EPISODES_PER_BATCH = 16
DISCOUNT_FACTOR = 0.99
UPDATE_EPOCHS = 10
ENTROPY_COEFFICIENT = 0
PPO_CLIP = 0.2
MINI_BATCH_SIZE = 64
OPTIMIZER_LR = 3e-4
ACTION_INIT_STD = 0.05
STD_HALVING_TIMESTEPS = 100000

# Define environment
env = gym.make('Pendulum-v1')#, render_mode="human")

# Define models
actor_net = ActorNet(env.observation_space.shape[0], env.action_space.shape[0])
critic_net = nn.Sequential(
              nn.Linear(env.observation_space.shape[0], NET_HIDDEN[0]),
              nn.Tanh(),
              nn.Linear(NET_HIDDEN[0], NET_HIDDEN[1]),
              nn.Tanh(),
              nn.Linear(NET_HIDDEN[1], 1)
              )

def collect_trajectories(n_trajectories, env, gamma):

    global log, t
    states, actions, rewards, rewards_to_go, log_probs = [], [], [], [], []

    for episode in range(n_trajectories): 
        obs = env.reset()
        done = False
        episode_rewards = []

        while not done:
            states.append(obs)
            action, log_prob, _ = actor_net.get_action(torch.tensor(obs, dtype=torch.float32))
            obs, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            actions.append(action)
            episode_rewards.append(reward)
            t += 1

        rewards += episode_rewards
        if episode % 5 == 0: log.append((t, sum(episode_rewards)))

        episode_rtgs = []
        G = 0
        for r in episode_rewards[::-1]:
            G = r + gamma * G
            episode_rtgs.insert(0, G) # insert reversed
        rewards_to_go += episode_rtgs

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    log_probs = torch.tensor(log_probs, dtype=torch.float32)
    rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float32)

    return states, actions, log_probs, rewards_to_go, rewards

# TRAINING

t = 0
log = []

for k in range(N_EPOCHS):

    if k%1==0: print(f"Epoch {k} (timesteps so far {t})")

    ## Collect batch of trajectories D_k and rewards-to-go by running current policy
    trajectories = collect_trajectories(n_trajectories=EPISODES_PER_BATCH, 
                                      env=env, 
                                      gamma=DISCOUNT_FACTOR)
    states, actions, log_probs, rewards_to_go, rewards = trajectories

    ## Compute advantage estimates based on the current value function
    estimated_values = critic_net(torch.tensor(np.array(states), dtype=torch.float32)).squeeze().detach()
    advantages = rewards_to_go - estimated_values
    advantages = (advantages - advantages.mean()) / advantages.std() # improves stability?

    if k%1==0: 
      print("sum of reward per episode:", sum(rewards)/EPISODES_PER_BATCH)
      #print(sum([abs(x)<1 for x in rewards]))
    if k%10==9:
      plt.clf()
      plt.plot([x[0] for x in log], [x[1] for x in log], linewidth=1)
      plt.show()

    assert states.shape[0] == actions.shape[0]
    assert states.shape[0] == rewards_to_go.shape[0]
    assert states.shape[0] == log_probs.shape[0]
    assert states.shape[0] == advantages.shape[0]
        
    actor_optimizer = torch.optim.Adam(actor_net.parameters(), lr=OPTIMIZER_LR)
    critic_optimizer = torch.optim.Adam(critic_net.parameters(), lr=OPTIMIZER_LR)

    ## Update policy by maximising the PPO-Clip objective (via SGA - Adam)
    for i in range(UPDATE_EPOCHS):

        L = states.shape[0]
        indices = torch.randperm(L)
        for j in range(L//MINI_BATCH_SIZE):
            mb_idx = indices[j*MINI_BATCH_SIZE:(j+1)*MINI_BATCH_SIZE] # minibatch indices

            #  PPO-Clip loss
            _, new_log_probs, entropy = actor_net.get_action(states[mb_idx], actions[mb_idx])
            ratio = torch.exp(new_log_probs - log_probs[mb_idx])
            # clip(r(theta), 1+eps, 1-eps)
            clipped_ratio = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) 
            # L_CLIP = min(r*A, clipped(r)*A) (objective to maximize)
            actor_loss = - torch.min(ratio * advantages[mb_idx], clipped_ratio * advantages[mb_idx]).mean() #-ENTROPY_COEFFICIENT*entropy
            #print("actor loss", actor_loss.item())

            actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            actor_optimizer.step()

            predicted_rtgs = critic_net(states[mb_idx]).squeeze()
            critic_loss = nn.MSELoss()(predicted_rtgs, rewards_to_go[mb_idx])

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
        