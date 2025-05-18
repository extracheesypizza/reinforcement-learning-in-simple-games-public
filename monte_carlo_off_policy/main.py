from monte_carlo_off_policy import MonteCarloOffPolicy
from functions import plot_blackjack_policy, test_policy, create_state_space
import gymnasium as gym
import numpy as np
import os

test_vars = None # [100, 50_000]

def choose_env(num):
    if num == 0:
        envs = [gym.make('Blackjack-v1', sab=True),
                gym.make('Blackjack-v1', sab=True, render_mode='human')]
    if num == 1:
        envs = [gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True),
                gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human')]
    return envs

if __name__ == "__main__":
    play_game = 0   # 0 -> blackjack | 1 ->  frozen_lake
    envs = choose_env(play_game) # 'blackjack' or 'frozen_lake'
    mc_agent = MonteCarloOffPolicy(envs[0], gamma=1.0, epsilon=0.1)

    # training the model
    mc_results = mc_agent.train_agent(num_episodes=150_001, test_vars=test_vars)

    if test_vars:
        file_directory = os.path.split(os.path.realpath(__file__))[0]
        np.savetxt('mcoffpolicy_results.txt', np.array(mc_results), delimiter=' ', fmt='%f')
    else:
        test_policy(mc_agent, envs[0], 1_000) # may take some time
        # plot_blackjack_policy(mc_agent) # todo