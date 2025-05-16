import time
from mcts import *

# testing loop
def test_mcts(env, mcts_player, opponent_player, num_episodes=1_000, mcts_iterations=100, render_env=True):
    print(f"--- Starting Testing (MCTS vs Random) ---")
    print(f"MCTS Iterations per move: {mcts_iterations}")
    wins_mcts = 0
    wins_random = 0
    draws = 0

    # determining which player is MCTS
    mcts_is_p1 = (mcts_player == P1_PAWN)
    print(f"MCTS playing as: {'P1 (o)' if mcts_is_p1 else 'P2 (x)'}")

    for i_episode in range(num_episodes):
        env.reset()
        done = False
        turn = 0
        if render_env:
            print(f"\nTest game #{i_episode+1}")
            env.render()

        while not done:
            current_player = env.current_player
            action_tuple = None

            if current_player == mcts_player:
                # MCTS agent's turn
                # print(f"Turn {turn+1}: MCTS (Player {'1' if mcts_is_p1 else '2'}) thinking...")
                start_time = time.time()
                action_tuple = MCTS_Search(env, mcts_iterations) # Pass current env state
                end_time = time.time()
                # print(f"MCTS chose action {action_tuple} in {end_time - start_time:.2f} seconds.")
                if action_tuple is None:
                    print(f"Error: MCTS returned None action for Player {'1' if mcts_is_p1 else '2'}. Opponent wins.")
                    env.winner = opponent_player
                    done = True
                    break
            else:
                # random Agent's turn
                action_tuple = choose_random_action(env)
                if action_tuple is None:
                    # print(f"Turn {turn+1}: Random Player ({'1' if not mcts_is_p1 else '2'}) has no moves. MCTS wins.")
                    env.winner = mcts_player
                    done = True
                    break
                # print(f"Turn {turn+1}: Random Player ({'1' if not mcts_is_p1 else '2'}) moves {action_tuple}")


            # environment step
            if action_tuple:
                 env.step(action_tuple) # env.step modifies the env object
                 if render_env:
                    env.render() # render each step during testing
                    time.sleep(0.1) # slowing down visualization
            else:
                 # should be handled by the 'no moves' checks above or mcts error
                 print("Error: No action tuple generated, but game not flagged as done.")
                 done = True # ending the game if something went wrong

            # checking game status after move
            done = env.is_game_over()
            turn += 1
            if turn > 150: # adding a turn limit to prevent infinite games
                 print("Game reached turn limit (150). Declaring draw.")
                 env.winner = None # draw
                 done = True


        # end of the game
        if env.winner == mcts_player:
            wins_mcts += 1
            print(f"Game {i_episode+1}: MCTS won.")
        elif env.winner == opponent_player:
            wins_random += 1
            print(f"Game {i_episode+1}: Random Player won.")
        else:
            draws += 1
            print(f"Game {i_episode+1}: Draw.")


    win_rate_mcts = (wins_mcts / num_episodes) * 100
    win_rate_random = (wins_random / num_episodes) * 100
    draw_rate = (draws / num_episodes) * 100

    print(f"\nTesting finished. Results over {num_episodes} games:")
    print(f"- MCTS ({'P1' if mcts_is_p1 else 'P2'}) Wins: {wins_mcts} ({win_rate_mcts:.2f}%)")
    print(f"- Random ({'P2' if mcts_is_p1 else 'P1'}) Wins: {wins_random} ({win_rate_random:.2f}%)")
    print(f"- Draws: {draws} ({draw_rate:.2f}%)")
