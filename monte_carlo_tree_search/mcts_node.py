import numpy as np
import math
import random
from checkers_env import P1_PAWN, P2_PAWN

# exploration constant
UCT_C = np.sqrt(2) # sqrt(2) is a common value

# MCTS Node
class MCTSNode:
    def __init__(self, environment_state, parent=None, action=None):
        self.env_state = environment_state # stores a CheckersEnv object
        self.parent = parent
        self.action = action # action that led to this state
        self.children = []
        self.untried_actions = self.env_state.get_legal_actions() # actions not yet explored from this node
        self.visit_count = 0
        self.value_sum = 0.0 # sum of results from simulations through this node
        self.player_turn = self.env_state.current_player # player whose turn it is in this state

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self):
        return self.env_state.is_game_over()

    def select_child(self, exploration_value=UCT_C):
        """Selects the best child node using the UCT formula"""
        best_score = -float('inf')
        best_child = None

        for child in self.children:
            if child.visit_count == 0:
                 # prefernce for unvisited children to ensure exploration
                 uct_score = float('inf') # assigning a very high score
            else:
                # UCT = (win_rate) + C * sqrt(log(parent_visits) / child_visits)
                # win rate is from the perspective of the player *at the child node*
                # value_sum is updated relative to the player at the node being updated
                # so, value_sum / visit_count is the average outcome for the player whose turn it is at the child node
                # the parent node wants to choose the child that maximizes the outcome for the parent's player

                # we are sticking to the standard UCT: average value + exploration bonus
                # the value should represent the value for the player whose turn it is *at the parent node*
                # we adjust the value during backpropagation

                # standard UCT calculation [value relative to the node's player]:
                exploit_term = child.value_sum / child.visit_count
                explore_term = exploration_value * math.sqrt(math.log(self.visit_count) / child.visit_count)

                # adjusting score based on whose turn it is at the *parent* node [self]:
                # - if it's the same player's turn at the child as the parent (shouldn't happen in checkers)
                # - if it's the opponent's turn at the child node, the parent wants to *minimize* the child's value, which is equivalent to maximizing the negative of the child's value

                uct_score = (child.value_sum / child.visit_count) + exploration_value * math.sqrt(math.log(self.visit_count) / child.visit_count)

            # if the child node represents the opponent's turn, the parent wants to pick the one
            # with the lowest value for the opponent (highest value for the parent)
            # the uct_score calculated above is from the child's perspective, we need the score from the parent's perspective
            score_for_parent = uct_score
            if child.player_turn != self.player_turn:
                 # if child is opponent's turn, parent wants to maximize -1 * (value/visits)
                 exploit_term_parent_view = - (child.value_sum / child.visit_count)
                 score_for_parent = exploit_term_parent_view + explore_term # keeping exploration bonus positive

            if score_for_parent > best_score:
                best_score = score_for_parent
                best_child = child

        # sanity check: should always find a child if children exist
        if best_child is None and self.children:
             print("Warning: No best child found in selection, picking first")
             return self.children[0]

        return best_child


    def expand(self):
        """Expands the node by creating one new child node"""
        if not self.untried_actions:
             print("Warning: Expand called on fully expanded or terminal node")
             return None # should not happen if called correctly, but still

        action = self.untried_actions.pop() # taking one untried action
        # creating the next state by applying the action
        next_env_state = self.env_state.copy()
        next_env_state.step(action) # applying the action

        # creating the new child node
        child_node = MCTSNode(next_env_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def simulate(self):
        """Simulates a random playout from the current state until the game ends"""
        simulation_env = self.env_state.copy()
        while not simulation_env.is_game_over():
            legal_actions = simulation_env.get_legal_actions()
            if not legal_actions:
                # determining winner based on who has no moves
                simulation_env.winner = P2_PAWN if simulation_env.current_player == P1_PAWN else P1_PAWN
                break
            random_action = random.choice(legal_actions)
            simulation_env.step(random_action)

        # returning the result from the perspective of the player whose turn it was *at the start of the simulation* [self.player_turn]
        return simulation_env.get_result(self.player_turn) # +1 for win, -1 for loss, 0 for draw


    def backpropagate(self, result):
        """Backpropagates the simulation result up the tree"""
        node = self
        while node is not None:
            node.visit_count += 1
            # the result is from the perspective of the player at the node *where simulation started*
            # since self.simulate() returns result relative to self.player_turn [node where sim started],
            # we need to check whether the current node in backprop has the same player
            if node.player_turn == self.player_turn:
                 node.value_sum += result # result is already from this player's perspective
            else:
                 node.value_sum -= result # result needs to be flipped for the opponent

            node = node.parent
