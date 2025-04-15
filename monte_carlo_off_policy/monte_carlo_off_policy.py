import numpy as np
import gymnasium as gym
from tqdm import tqdm
from gymnasium.spaces import Tuple, Discrete
from functions import create_state_space

class MonteCarloOffPolicy:
    def __init__(self, env, gamma=1.0, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon

        # handling different observation spaces
        if isinstance(env.observation_space, Tuple):
            self.tuple_observation()
        elif isinstance(env.observation_space, Discrete):
            self.discrete_observation()
        else:
            raise ValueError("Unsupported observation space")

        # handling action space
        if not isinstance(env.action_space, Discrete):
            raise ValueError("Discrete action space required")
        self.n_actions = env.action_space.n

        # initializing Q-table with small random values to break symmetry
        self.Q = np.random.uniform(low=-0.01, high=0.01, size=(self.n_states, self.n_actions))
        self.C = np.zeros((self.n_states, self.n_actions))

    def discrete_observation(self):
        """Handling discrete observation spaces (like FrozenLake)"""
        self.n_states = self.env.observation_space.n
        self.get_state_index = lambda state: state

    def tuple_observation(self):
        """Handling tuple observation spaces (like Blackjack)"""
        self.state_space = create_state_space([dim.n for dim in self.env.observation_space])
        self.n_states = len(self.state_space)
        self.state_to_idx = {state: idx for idx, state in enumerate(self.state_space)}
        self.get_state_index = lambda state: self.state_to_idx.get(state, -1)

    def choose_action(self, state_idx):
        """Epsilon-greedily choosing an action"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.greedy_action(state_idx)

    def greedy_action(self, state_idx):
        """Random choice between max Q-value actions"""
        max_q = np.max(self.Q[state_idx])
        candidates = np.where(self.Q[state_idx] == max_q)[0]
        return np.random.choice(candidates)

    def generate_episode(self):
        """Generating episode with exploration"""
        episode = []
        observation, _ = self.env.reset()
        state_idx = self.get_state_index(observation)
        done = False
        
        while not done:
            action = self.choose_action(state_idx)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            next_state_idx = self.get_state_index(next_obs)
            
            episode.append([state_idx, action, reward])
            state_idx = next_state_idx
            
        return episode

    def train_episode(self, episode):
        """Importance sampling with random tie-breaking"""
        G = 0.0
        W = 1.0
        
        for t in reversed(range(len(episode))):
            state = episode[t][0]
            action = episode[t][1]
            G = self.gamma * G + episode[t][2] # reward
            
            self.C[state, action] += W
            self.Q[state, action] += (W / self.C[state, action]) * (G - self.Q[state, action])
            
            target_action = self.greedy_action(state)
            if action != target_action:
                break
                
            behavior_prob = self.epsilon / self.n_actions
            if action == target_action:
                behavior_prob += 1 - self.epsilon
                
            W *= 1.0 / max(behavior_prob, 1e-10)  # preventing division by zero
            
            if W < 1e-10:  # early exitting to prevent numerical issues
                break

    def train_agent(self, n_episodes):
        """Training with epsilon decay and progress tracking"""
        for _ in tqdm(range(n_episodes), desc="Training"):
            episode = self.generate_episode()
            self.train_episode(episode)
            

    def test(self, n_episodes=100):
        """Greedy policy evaluation"""
        total_reward = 0.0
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            state_idx = self.get_state_index(obs)
            done = False
            
            while not done:
                action = self.greedy_action(state_idx)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                state_idx = self.get_state_index(next_obs)
                
        return total_reward / n_episodes


