from functions import plot_blackjack_policy, test_policy
from monte_carlo import MonteCarloES
import gymnasium as gym
import numpy as np
import os

test_vars = None # [100, 50_000]

if __name__ == "__main__":
    # creating the Blackjack environment
    env = gym.make('Blackjack-v1', sab=True)

    mc_agent = MonteCarloES(env, gamma=1.0)
    mc_results = mc_agent.train_agent(num_episodes=150_001, test_vars=test_vars)

    if test_vars:
        file_directory = os.path.split(os.path.realpath(__file__))[0]
        np.savetxt('mc_results.txt', np.array(mc_results), delimiter=' ', fmt='%f')
    else:
        test_policy(mc_agent, env, 100_000) # may take some time
        plot_blackjack_policy(mc_agent)