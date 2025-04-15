from gymnasium.spaces import Tuple, Discrete
from collections import defaultdict
import gymnasium as gym
from tqdm import tqdm
import numpy as np

class MonteCarloAlpha:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        if not isinstance(env.action_space, Discrete):
            raise ValueError("Discrete action space required")
        
        self.n_actions = env.action_space.n
        self.Q = defaultdict(float)  # Q-values stored as (state, action) -> float
        self.policy = defaultdict(float)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = [self.Q[(state, a)] for a in range(self.n_actions)]
            max_q = max(q_values)
            actions_with_max_q = [a for a, q in enumerate(q_values) if q == max_q]
            return np.random.choice(actions_with_max_q)
    
    def generate_episode(self):
        episode = []
        state, _ = self.env.reset()
        done = False
        
        # generating episode
        while not done:
            action = self.choose_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            episode.append((state, action, reward))
            state = next_state

        return episode

    def train_episode(self, episode):
        # calculating returns and update Q-values
        G = 0
        for t in range(len(episode)-1, -1, -1):
            state, action, reward = episode[t]
            G = self.gamma * G + reward
            current_q = self.Q[(state, action)]
            self.Q[(state, action)] += self.alpha * (G - current_q)
            self.policy[state] = np.argmax([self.Q[(state, action)] for action in range(self.n_actions)])

    def train_agent(self, n_episodes):
        for _ in tqdm(range(n_episodes), desc="Training"):
            episode = self.generate_episode()
            self.train_episode(episode)

    def test_episode(self, render=False):
        original_epsilon = self.epsilon
        self.epsilon = 0.0
        state, _ = self.env.reset()
        done = False
        total_reward = 0
        
        while not done:
            if render:
                self.env.render()
            action = self.choose_action(state)
            state, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            
        self.epsilon = original_epsilon
        return total_reward