import gymnasium as gym
import numpy as np
import torch

# helper functions used for creating a suitable environments for the agent
def make_pong_env(render=None):
    env = gym.make('Pong-v4', frameskip=1, render_mode=render, mode=1, difficulty=0)  # creating base environment
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, grayscale_obs=True, scale_obs=True)
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env

def make_breakout_env(render=None):
    env = gym.make('Breakout-v4', frameskip=1, render_mode=render, mode=0, difficulty=0)  # creating base environment
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, grayscale_obs=True, scale_obs=True) # terminal_on_life_loss=True
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env

def make_space_invaders_env(render=None):
    env = gym.make('SpaceInvaders-v4', frameskip=1, render_mode=render)  # creating base environment
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=3, screen_size=84, grayscale_obs=True, scale_obs=True)
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env

def make_env(env_name, render=None):
    if env_name == 'pong':
        return make_pong_env(render=render)
    if env_name == 'breakout':
        return make_breakout_env(render=render)
    if env_name == 'space_invaders':
        return make_space_invaders_env(render=render)
    return 0

# helper function used for testing the agent
def test(model, env_name, num_episodes=5, render='human'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = make_env(env_name, render=render)

    model = model.to(device)
    model.eval()

    results = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.array(state)
        done = False
        total_reward = 0
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_tensor)
            action = q_values.argmax().item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = np.array(next_state)
            total_reward += reward
            if render:
                env.render()

        results.append(total_reward)
        print(f"Test Episode {episode+1}, Total Reward: {total_reward}")
    print(f"Average Total Reward: {np.mean(results)}")
    env.close()