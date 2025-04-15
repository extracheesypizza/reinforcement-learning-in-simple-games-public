from collections import defaultdict
import gymnasium as gym
from tqdm import tqdm
import numpy as np

class MonteCarloES:
    def __init__(self, env, gamma=1.0):
        """
        Monte Carlo Exploring Starts algorithm implementation
        
        Args:
            env: OpenAI Gymnasium environment.
            gamma: discount factor.
        """
        self.env = env
        self.gamma = gamma
        
        # initializing empty policy and value function
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        self.policy = defaultdict(lambda: 0)
        self.returns = defaultdict(list)
        
        # tracking learning progress
        self.episode_rewards = []
        self.episode_lengths = []
    
    def generate_episode(self, exploring_starts=True):
        """Generating an episode using the current policy with exploring starts"""
        episode = []
        state, _ = self.env.reset()
        
        # exploring starts: choosing a random first action
        if exploring_starts:
            action = self.env.action_space.sample()
        else:
            action = self.policy[state]
        
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            episode.append((state, action, reward))
            total_reward += reward
            steps += 1
            state = next_state
            
            action = self.policy[state] # following the policy
        
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        
        return episode
    
    def train_episode(self, episode):
        """Updating action-value function and policy based on the episode"""
        states_actions = [(s, a) for s, a, _ in episode]
        
        # calculating returns for each step
        G = 0
        for t in range(len(episode)-1, -1, -1):
            state, action, reward = episode[t]
            G = self.gamma * G + reward
            
            # first-visit MC: updating only if this is the first occurrence of a (s,a) pair
            if (state, action) not in states_actions[:t]:
                self.returns[(state, action)].append(G)
                self.Q[state][action] = np.mean(self.returns[(state, action)])
                
                # updating the policy to be greedy with respect to Q
                self.policy[state] = np.argmax(self.Q[state])
    
    def train_agent(self, num_episodes=10000):
        """Training the agent using Monte Carlo ES"""
        for _ in tqdm(range(num_episodes), desc="Training"):
            episode = self.generate_episode(exploring_starts=True)
            self.train_episode(episode)