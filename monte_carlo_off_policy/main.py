from monte_carlo_off_policy import MonteCarloOffPolicy
from functions import plot_blackjack_policy, test_policy
import gymnasium as gym
import numpy as np

def choose_env(name):
    if name == 'blackjack':
        envs = [gym.make('Blackjack-v1', sab=True),
                gym.make('Blackjack-v1', sab=True, render_mode='human')]
    if name == 'frozen_lake':
        envs = [gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True),
                gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human')]
    return envs

if __name__ == "__main__":
    envs = choose_env('blackjack') # 'blackjack' or 'frozen_lake'
    agent = MonteCarloOffPolicy(envs[0], gamma=0.99, epsilon=0.1)

    # training the model
    agent.train_agent(n_episodes=75_000)
    
    # testing loop
    test_policy(agent, envs[0], 2_000)
    # final_reward = agent.test(n_episodes=2_000)
    # print(f"\nFinal success rate: {final_reward:.4f}")
