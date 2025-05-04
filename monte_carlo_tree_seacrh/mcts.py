from checkers_env import CheckersEnv, P1_PAWN, P2_PAWN
from mcts_node import MCTSNode
import random
import time

# MCTS search function
def MCTS_Search(root_environment_state, iterations):
    """
    Performs MCTS search from the root state.
    Args:
        root_environment_state: A CheckersEnv object representing the current state.
        iterations: The number of MCTS simulations to run.
    Returns:
        best_action: The best action tuple found.
    """
    if root_environment_state.is_game_over():
        return None # no action to take if game is over

    root_node = MCTSNode(environment_state=root_environment_state.copy()) # starting with a copy

    if not root_node.untried_actions and not root_node.children:
         print("Warning: Root node has no legal actions at the start.")
         return None # no legal moves from the start

    for _ in range(iterations):
        node = root_node

        # 1. Selection
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.select_child()
            if node is None: # should only happen if selection fails unexpectedly
                 print("Error: Selection returned None. Breaking MCTS iteration.")
                 break
        if node is None: continue # skipping to the next iteration if selection failed

        # 2. Expansion
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()
            if node is None: # should only happen if expansion fails unexpectedly
                 print("Error: Expansion returned None. Breaking MCTS iteration.")
                 continue # skipping to next iteration

        # 3. Simulation
        if node.is_terminal(): # if node is terminal, simulation result is determined directly
             # getting result from perspective of the player whose turn it *would have been* (parent's player)
             result_perspective = node.parent.player_turn if node.parent else node.player_turn # fallback needed?
             result = node.env_state.get_result(result_perspective)
             # adjusting result based on who's turn it is at the node being simulated *from*
             result = node.env_state.get_result(node.player_turn)

        else: # node was newly expanded or selected leaf
             result = node.simulate() # simulating from the expanded/selected node

        # 4. Backpropagation
        node.backpropagate(result)

    # choosing the best action from the root after iterations
    if not root_node.children:
         # this might happen if iterations is very low or only one move possible
         print("Warning: MCTS finished but root has no children. No action possible?")
         legal_root_actions = root_environment_state.get_legal_actions()
         if len(legal_root_actions) == 1:
             return legal_root_actions[0]
         return None


    # selecting the child with the highest visit count [most robust]
    best_child = max(root_node.children, key=lambda c: c.visit_count)
    # alternative: highest win rate (value_sum / visit_count), careful with division by zero
    # best_child = max(root_node.children, key=lambda c: (c.value_sum / c.visit_count) if c.visit_count > 0 else -float('inf'))

    # debugging snippet of code
    # print("---- MCTS Root Children Stats ----")
    # for child in sorted(root_node.children, key=lambda c: c.visit_count, reverse=True):
    #     win_rate = (child.value_sum / child.visit_count * 100) if child.visit_count > 0 else 0
    #     print(f"  Action: {child.action}, Visits: {child.visit_count}, Win Rate (for child player {child.player_turn}): {win_rate:.2f}%")
    # print(f"Selected Action: {best_child.action} based on visits.")
    # print("----------------------------------")
    return best_child.action


# helper function for random agent
def choose_random_action(env):
    legal_actions = env.get_legal_actions()
    if not legal_actions:
        return None
    return random.choice(legal_actions)