# Reinforcement Learning in Simple Games

## Introduction

This GitHub repository is accompanying the corresponding thesis paper, which can be accessed via the `report.pdf` file. The report goes through basics of Reinforcement Learning, theory behind some of the RL algorithms and describes the implementation details of the algorithms found in this repository.


## How to run?
1. Download the repository
2. Create and launch a virtual environment:
    - For windows systems: `python -m rl_env env && .\rl_env\Scripts\activate`
    - For UNIX systems: `python -m rl_env env && source ./rl_env/bin/activate`
3. Install dependencies:
    - `pip install -r requirements.txt`
4. Run any algorithm you wish:
    - `python dqn/main.py`

## Description
Here is the list of algorithms available in this repository. For more detail please consult the report.

### Monte Carlo Methods
- `/monte_carlo/` folder contains the implementation of the standard Monte Carlo Exploring Starts algorithm
- `/monte_carlo_alpha/` folder contains the implementation of the Constant-α Monte Carlo algorithm
- `/monte_carlo_off_policy/` folder contains the implementation of the off-policy Monte Carlo algorithm [uses *importance sampling*]
- `/monte_carlo_tree_search/`folder contains implementations of a Checkers game environment and the Monte Carlo Tree Search algorithm

### DQN
- `/dqn/` directory contains the implementation of the Deep Q-Network, which learns to play Atari games 

Each folder contains a `main.py` file which contains the training/evaluation loop, feel free to play around with them.