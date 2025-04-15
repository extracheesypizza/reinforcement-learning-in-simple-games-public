from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


def plot_blackjack_policy(agent):
    """Visualizing the learned Blackjack policy"""
    if "Blackjack" not in str(agent.env):
        print("Policy visualization is implemented only for Blackjack environment")
        return
    
    # creating separate plots for when ace is usable and when it's not
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    titles = ["Usable Ace", "No Usable Ace"]
    
    for ace_idx, usable_ace in enumerate([True, False]):
        policy_grid = np.zeros((10, 10))
        
        # filling the policy grid
        for player_sum in range(12, 22):
            for dealer_card in range(1, 11):
                state = (player_sum, dealer_card, usable_ace)
                action = agent.policy[state]
                # Adjust indices for the grid
                policy_grid[dealer_card-1, player_sum-12] = action
        
        # plotting the policy
        ax = axes[ace_idx]

        cmap = ListedColormap(['#FF7F7F', '#7FFF7F'])
        im = ax.imshow(policy_grid.T, cmap=cmap, origin='lower', aspect='auto', extent=[1 - 0.5, 10 + 0.5, 12 - 0.5, 21 + 0.5], vmin=0, vmax=1) 
        ax.set_title(titles[ace_idx])
        ax.set_xlabel("Dealer showing")
        ax.set_ylabel("Player sum")

        player_sums = np.arange(12, 21 + 1)
        dealer_cards = np.arange(1, 10 + 1)

        for i in range(len(player_sums)):
            for j in range(len(dealer_cards)):
                action = policy_grid.T[i, j]
                text = "S" if action == 0 else "H"
                ax.text(dealer_cards[j], player_sums[i], text, ha="center", va="center", color="black", fontsize=8)
                
    title = "Blackjack Policy - Monte Carlo Constant Alpha"
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def test_policy(agent, env, num_episodes):
    """Evaluating the given agent in the given environment"""
    total_rewards = 0
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.policy[state]
            state, reward, done, truncated, _ = env.step(action)
            
        total_rewards += reward
    
    avg_reward = total_rewards / num_episodes
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.4f}")
    return avg_reward