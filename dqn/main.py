from functions import test, make_env
from dqn import DQN, ReplayBuffer
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import random
import ale_py
import os

play_game = 'pong' # 'pong' / 'space_invaders' / 'breakout'

# creating the gane environment
env = make_env(play_game) 

print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# creating the agent
n_actions = env.action_space.n
input_shape = (4, 84, 84)
model = DQN(input_shape, n_actions).to(device)
target_model = DQN(input_shape, n_actions).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

# creating the optimizer and the replay buffer objects
optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=0.00025,         # learning rate [paper uses 0.00025]
        alpha=0.95,         # smoothing constant is called \rho in the paper
        eps=0.01,           # epsilon for numerical stability, paper uses 0.01
        weight_decay=0,
        momentum=0,         
        centered=False      # non-centered RMSprop [original version]
    )
buffer = ReplayBuffer(100_000)

# training hyperparameters
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 1_000_000
gamma = 0.99
batch_size = 32
sync_target_freq = 10_000
save_interval = 50
steps = 50_000_000 # not limiting the learning process, just in case

# the training loop
def train(episodes=10_000_000, save_interval=50, model_name='dqn_latest.pth'):
    # initializing environment
    state, _ = env.reset()
    state = np.array(state)
    episode_reward = 0
    n_episide = 0
    cur_rewards = []
    cur_best_reward = -999
    for step in range(1, episodes + 1):
        # epsilon decay
        epsilon = np.max([epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (step / epsilon_decay)])
        
        # epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
            action = q_values.argmax().item()
        
        # taking an action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = np.array(next_state)
        buffer.add(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        
        # resetting the environment if episode has ended
        if done:
            state, _ = env.reset()
            state = np.array(state)
            cur_rewards.append(episode_reward)
            n_episide += 1
            if n_episide % save_interval == 0:
                np.savetxt('cur_rewards_1.txt', np.array(cur_rewards), delimiter=' ', fmt='%f')
                if np.mean(cur_rewards[-50:]) > cur_best_reward: 
                    cur_best_reward = np.mean(cur_rewards[-50:])
                    target_model.save_model(model_name)
            print(f"Episode: {n_episide}\tEpisode Reward: {episode_reward}\tEpsilon: {epsilon:.2f}\tStep: {step}\tAvg.Reward: {np.mean(cur_rewards[-50:])}")
            episode_reward = 0
        
        # training
        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            
            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).to(device)
            
            current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                next_q = target_model(next_states).max(1)[0]
            
            target_q = rewards + gamma * next_q * (1 - dones)
            loss = nn.MSELoss()(current_q, target_q)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # DeepMind paper uses 1.0
            optimizer.step()
        
        # syncing target network
        if step % sync_target_freq == 0:
            target_model.load_state_dict(model.state_dict())
        
# True -> train a new model | False -> load the trained model from the file
train_mode = False

file_directory = os.path.split(os.path.realpath(__file__))[0]
model_name_save = file_directory + f'/dqn_{play_game}_latest_model.pth'
model_name_load = file_directory + f'/dqn_{play_game}_best_model.pth'

if __name__ == "__main__":
    if train_mode: # training the model
        train(episodes=steps, save_interval=save_interval, model_name=model_name_save)

    else: # loading and testing the trained model
        trained_model = DQN(input_shape, n_actions).to(device)
        trained_model.load_model(model_name_load, device)
        test(trained_model, env_name=play_game, num_episodes=3)