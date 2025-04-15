from monte_carlo import MonteCarloES
from functions import plot_blackjack_policy, test_policy
import gymnasium as gym

# creating the Blackjack environment
env = gym.make('Blackjack-v1', sab=True)

# initializing and training the Monte Carlo ES agent
mc_agent = MonteCarloES(env, gamma=1.0)
mc_agent.train_agent(num_episodes=700_000)

# visualizing the learned policy
plot_blackjack_policy(mc_agent)
test_policy(mc_agent, env, 20_000)