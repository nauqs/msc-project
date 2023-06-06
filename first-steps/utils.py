import numpy as np
import matplotlib.pyplot as plt

def plot_logs(timesteps, rewards, episode_lengths, actor_loss, critic_loss, step, smooth=True, title="ppo", save_path="ppo.png"):
    alpha_non_smoothed, n_smooth = 1, 5
    smooth = smooth and len(rewards) > n_smooth
    if smooth:
        conv_smooth = np.ones((n_smooth,))/n_smooth
        smoothed_rewards = np.convolve(rewards, conv_smooth, mode='valid')
        smoothed_episode_lengths = np.convolve(episode_lengths, conv_smooth, mode='valid')
        smoothed_actor_loss = np.convolve(actor_loss, conv_smooth, mode='valid')
        smoothed_critic_loss = np.convolve(critic_loss, conv_smooth, mode='valid')
        alpha_non_smoothed = 0.2

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot rewards and episode lengths
    ax1 = axs[0]
    ax1.plot(timesteps, rewards, 'b-', alpha=alpha_non_smoothed)
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Average reward', color='b')
    ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    ax2.plot(timesteps, episode_lengths, 'r-', alpha=alpha_non_smoothed)
    ax2.set_ylabel('Average episode length', color='r')
    ax2.tick_params('y', colors='r')
    if smooth:
        ax1.plot(timesteps[n_smooth-1:], smoothed_rewards, 'b-')
        ax2.plot(timesteps[n_smooth-1:], smoothed_episode_lengths, 'r-')

    # Plot actor and critic losses
    ax3 = axs[1]
    ax3.plot(timesteps, actor_loss, 'g-', alpha=alpha_non_smoothed)
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Actor loss', color='g')
    ax3.tick_params('y', colors='g')
    ax4 = ax3.twinx()
    ax4.plot(timesteps, critic_loss, 'm-', alpha=alpha_non_smoothed)
    ax4.set_ylabel('Critic loss', color='m')
    ax4.tick_params('y', colors='m')
    if smooth:
        ax3.plot(timesteps[n_smooth-1:], smoothed_actor_loss, 'g-')
        ax4.plot(timesteps[n_smooth-1:], smoothed_critic_loss, 'm-')

    plt.xlim(0, timesteps[-1]*1.05)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
