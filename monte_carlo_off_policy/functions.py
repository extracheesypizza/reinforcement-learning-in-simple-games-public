from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

def plot_blackjack_policy(agent):
    """Visualizing the learned Blackjack policy"""
    # todo

def test_policy(agent, env, num_episodes, render=False):
    """Evaluating the given agent in the given environment"""
    total_rewards = 0
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            if render:
                env.render()
            action = int(agent.greedy_action(agent.get_state_index(state)))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
        total_rewards += reward
    
    avg_reward = total_rewards / num_episodes
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.4f}")
    return avg_reward

def create_state_space(n_ks):
    results = []
    n_ks_array = np.array(n_ks)
    n = len(n_ks_array)
    for k in n_ks_array:
        ranges = [np.arange(k + 1)] * n
        mesh = np.meshgrid(*ranges)
        combinations_array = np.stack(mesh, axis=-1).reshape(-1, n)
        combinations_tuples = [tuple(int(val) for val in row) for row in combinations_array]
        results.append(combinations_tuples)
    return results[0]