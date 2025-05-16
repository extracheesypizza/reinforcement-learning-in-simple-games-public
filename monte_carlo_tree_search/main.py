from checkers_env import CheckersEnv, P1_PAWN, P2_PAWN
from functions import test_mcts

MCTS_ITERATIONS = 100 # number of MCTS simulations per move
NUM_EPISODES_TEST = 500 # number of episodes for testing
mcts_plays_first = False # deciding whether MCTS is P1 or P2

if __name__ == "__main__":
    # initializing environment
    env = CheckersEnv()

    if mcts_plays_first:
        print('MCTS Agent is P1 (o)')
        players = [P1_PAWN, P2_PAWN]
    else:
        print('MCTS Agent is P2 (x)')
        players = [P2_PAWN, P1_PAWN]
        
    test_mcts(env, mcts_player=players[0], opponent_player=players[1], num_episodes=NUM_EPISODES_TEST, mcts_iterations=MCTS_ITERATIONS, render_env=False)