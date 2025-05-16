import numpy as np

# checkers environment configuration
BOARD_SIZE = 8

# values for players' pieces
EMPTY = 0
P1_PAWN = 1
P1_KING = 2
P2_PAWN = -1
P2_KING = -2

class CheckersEnv:
    """
    A simplified Checkers environment, adapted for MCTS:
    - Pawns move forward diagonally one step
    - Kings move diagonally one step in any direction
    - Simple captures [jumping over opponent]
    - No forced captures
    - Game ends when one player has no pieces or no legal moves
    """
    def __init__(self, board=None, current_player=None):
        self.board_size = BOARD_SIZE

        if board is not None:
            self.board = board.copy()
            self.current_player = current_player
        else:
            self.board = np.zeros((self.board_size, self.board_size), dtype=int)
            self.current_player = P1_PAWN # player 1 starts
            self.reset() # calling reset to initialize properly

        self.winner = None


    def reset(self):
        """Resets the board to the starting position."""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        # placing pieces
        for r in range(3):
            for c in range(self.board_size):
                if (r + c) % 2 == 1:
                    self.board[r, c] = P2_PAWN # player 2

        for r in range(self.board_size - 3, self.board_size):
             for c in range(self.board_size):
                if (r + c) % 2 == 1:
                    self.board[r, c] = P1_PAWN # player 1

        self.current_player = P1_PAWN
        self.winner = None

    def copy(self):
        """Creates a deep copy of the environment state."""
        new_env = CheckersEnv(board=self.board, current_player=self.current_player)
        new_env.winner = self.winner
        return new_env

    def _get_state(self):
        """Returns the current board state as a flattened numpy array."""
        return self.board.flatten().astype(np.float32)

    def _is_valid_pos(self, r, c):
        """Checks if a position (row, col) is within the board."""
        return 0 <= r < self.board_size and 0 <= c < self.board_size

    def _get_piece_moves(self, r, c):
        """Gets possible moves (no captures) for a piece at (r, c)."""
        moves = []
        piece = self.board[r, c]
        # determining player based on piece value
        player = P1_PAWN if piece > 0 else P2_PAWN
        if player != self.current_player: # ensuring piece belongs to current player
             return []

        directions = []
        if piece == P1_PAWN:
            directions = [(-1, -1), (-1, 1)] # P1 moves up [starts at the bottom]
        elif piece == P2_PAWN:
            directions = [(1, -1), (1, 1)] # P2 moves down [starts at the top]
        elif abs(piece) == P1_KING: # kings move any diagonal
             directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        # regular moves
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if self._is_valid_pos(nr, nc) and self.board[nr, nc] == EMPTY:
                moves.append((r, c, nr, nc))
        return moves

    def get_piece_captures(self, r, c):
        """Gets possible captures for a piece at (r, c)."""
        captures = []
        piece = self.board[r, c]
        # determining player based on piece value
        player = P1_PAWN if piece > 0 else P2_PAWN
        if player != self.current_player: # ensuring piece belongs to the current player
             return []

        opponent_pawn = P2_PAWN if player == P1_PAWN else P1_PAWN
        opponent_king = P2_KING if player == P1_PAWN else P1_KING

        directions = []
        if piece == P1_PAWN:
             directions = [(-1, -1), (-1, 1)]
        elif piece == P2_PAWN:
             directions = [(1, -1), (1, 1)]
        elif abs(piece) == P1_KING: # kings move any diagonal
             directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        # looking for captures
        for dr, dc in directions:
            mr, mc = r + dr, c + dc # middle square [where opponent is supposed to be]
            nr, nc = r + 2 * dr, c + 2 * dc # landing square [second one in the direction of the opponent]
            if self._is_valid_pos(nr, nc) and self.board[nr, nc] == EMPTY:
                if self._is_valid_pos(mr, mc) and (self.board[mr, mc] == opponent_pawn or self.board[mr, mc] == opponent_king):
                    captures.append((r, c, nr, nc)) # storing the move
        return captures


    def get_legal_actions(self):
        """
        Returns a list of legal actions for the *current* player.
        Returns actions as (from_r, from_c, to_r, to_c).
        Simplified: does not enforce forced captures rigorously but prioritizes them.
        """
        legal_moves = []
        legal_captures = []
        player = self.current_player
        piece_type_pawn = P1_PAWN if player == P1_PAWN else P2_PAWN
        piece_type_king = P1_KING if player == P1_PAWN else P2_KING

        # checking all pieces for captures first
        for r in range(self.board_size):
            for c in range(self.board_size):
                piece = self.board[r, c]
                if piece == piece_type_pawn or piece == piece_type_king:
                    legal_captures.extend(self.get_piece_captures(r, c))

        if legal_captures:
            return legal_captures # standard checkers rule: if a capture is available, it must be taken

        # if no captures, find regular moves
        for r in range(self.board_size):
            for c in range(self.board_size):
                 piece = self.board[r, c]
                 if piece == piece_type_pawn or piece == piece_type_king:
                    legal_moves.extend(self._get_piece_moves(r, c))

        return legal_moves

    def promote_pawns(self):
        """Promotes pawns that reach the opposite end to kings."""
        for c in range(self.board_size):
            if self.board[0, c] == P1_PAWN: # P1 reaches top row
                self.board[0, c] = P1_KING
            if self.board[self.board_size - 1, c] == P2_PAWN: # P2 reaches bottom row
                self.board[self.board_size - 1, c] = P2_KING

    def is_game_over(self):
        """Checks if the game has ended. Sets self.winner."""
        # checking piece counts
        p1_pieces = np.count_nonzero((self.board == P1_PAWN) | (self.board == P1_KING))
        p2_pieces = np.count_nonzero((self.board == P2_PAWN) | (self.board == P2_KING))

        if p1_pieces == 0:
            self.winner = P2_PAWN # P2 wins
            return True
        if p2_pieces == 0:
            self.winner = P1_PAWN # P1 wins
            return True

        # checking for legal moves
        current_player_moves = self.get_legal_actions()
        if not current_player_moves:
            # the player whose turn it is has no moves, they lose
            self.winner = P2_PAWN if self.current_player == P1_PAWN else P1_PAWN
            return True # game is over

        return False # game is not over

    def step(self, action_tuple):
        """
        Executes an action (move) for the current player.
        Assumes action_tuple is a legal move obtained from get_legal_actions.
        Returns the environment object after the move (for MCTS convenience).
        """
        if action_tuple is None:
            print("Warning: step() called with None action.")
            return self

        from_r, from_c, to_r, to_c = action_tuple
        piece = self.board[from_r, from_c]

        # executing the move 
        self.board[to_r, to_c] = piece
        self.board[from_r, from_c] = EMPTY

        # checking for capture [if move distance is 2]
        if abs(from_r - to_r) == 2:
            capture_r, capture_c = (from_r + to_r) // 2, (from_c + to_c) // 2
            self.board[capture_r, capture_c] = EMPTY

        # promoting pawns if they reach the end
        self.promote_pawns()

        # switching players
        self.current_player = P2_PAWN if self.current_player == P1_PAWN else P1_PAWN

        # checking for game over after switching player [as the state is now ready for the NEXT player]
        self.is_game_over()

        return self # returning the modified environment

    def get_result(self, player_perspective):
        """
        Returns the game result from the perspective of a given player.
        Should only be called when the game is over.
        Args:
            player_perspective: The player (P1_PAWN or P2_PAWN) for whom the result is requested.
        Returns:
            1.0 for win, -1.0 for loss, 0.5 for draw (or 0.0).
        """
        if not self.is_game_over(): # ensuring game is actually over
             return 0.0

        if self.winner == player_perspective:
            return 1.0
        elif self.winner == -player_perspective: # opponent won
            return -1.0
        else:
            return 0.0 # draw

    # render function
    def render(self):
        """Prints the board to the console."""
        print("  " + " ".join(map(str, range(self.board_size))))
        print(" +" + "--"*self.board_size + "+")
        for r in range(self.board_size):
            row_str = str(r) + "|"
            for c in range(self.board_size):
                piece = self.board[r, c]
                if piece == P1_PAWN: char = 'o'
                elif piece == P1_KING: char = 'O'
                elif piece == P2_PAWN: char = 'x'
                elif piece == P2_KING: char = 'X'
                else: char = '.'
                row_str += " " + char
            print(row_str + " |")
        print(" +" + "--"*self.board_size + "+")
        player_str = '?'
        if self.current_player == P1_PAWN: player_str = 'P1 (o)'
        elif self.current_player == P2_PAWN: player_str = 'P2 (x)'
        print(f"Current Player: {player_str}")
        if self.is_game_over():
             winner_str = '?'
             if self.winner == P1_PAWN: winner_str = 'P1 (o)'
             elif self.winner == P2_PAWN: winner_str = 'P2 (x)'
             elif self.winner is None: winner_str = 'Draw'
             print(f"Game Over! Winner: {winner_str}")