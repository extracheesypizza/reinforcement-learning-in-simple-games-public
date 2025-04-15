from monte_carlo_alpha import MonteCarloAlpha
from functions import plot_blackjack_policy, test_policy
import gymnasium as gym
import numpy as np

if __name__ == "__main__":
    # creating environment (non-slippery version for easier learning)
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    # env = gym.make('Blackjack-v1', sab=True)
    
    # initializing agent
    agent = MonteCarloAlpha(env, alpha=0.01, gamma=1.0, epsilon=0.15)
    
    # Training parameters
    n_episodes = 50_000

    # training loop
    agent.train_agent(n_episodes)

    # testing loop
    test_policy(agent, env, 2_000)
    # plot_blackjack_policy(agent) 