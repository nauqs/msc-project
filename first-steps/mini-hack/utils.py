import numpy as np
import matplotlib.pyplot as plt


def plot_logs(timesteps, rewards, episode_lengths, step, smooth=True, title="ppo", save_path="ppo.png"):
    # plot both rewards and episode lengths in same figure, but different scales
    alpha_non_smoothed, n_smooth = 1, 5
    smooth = smooth and len(rewards) > n_smooth
    if smooth:
        conv_smooth = np.ones((n_smooth,))/n_smooth
        smoothed_rewards = np.convolve(rewards, conv_smooth, mode='valid')
        smoothed_episode_lengths = np.convolve(episode_lengths, conv_smooth, mode='valid')
        alpha_non_smoothed = 0.2
    fig, ax1 = plt.subplots()
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
    plt.xlim(0, step+1)
    plt.title(title)
    plt.savefig(save_path, dpi=200)
    plt.close()