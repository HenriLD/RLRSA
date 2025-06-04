# random_agents.py
import gymnasium as gym
# Assuming environment.py is in the same directory or accessible in PYTHONPATH
from environment import CooperativeChickenEnv 

def run_random_agents_episode(env, render_episode_to_console=False):
    """
    Runs a single episode with two random agents.

    Args:
        env: An instance of the CooperativeChickenEnv.
        render_episode_to_console (bool): Whether to print the state of the environment to console.

    Returns:
        tuple: (total_reward_agent1, total_reward_agent2, episode_length, history_log)
               history_log is a list of dictionaries, each detailing a step.
    """
    obs, info = env.reset()
    if render_episode_to_console:
        print("--- Initial State ---")
        print(env.render())
        print(f"Observation: {obs}")
        print(f"Info: {info}")

    history_log = []
    # Log initial state before any action
    history_log.append({
        'grid_render_str': env.render(), # For debug
        'agent1_pos': env.agent1_pos,
        'agent2_pos': env.agent2_pos,
        'chicken_pos': env.chicken_pos,
        'acting_agent': info.get("current_player_to_act", -1), 
        'action_taken': None, 
        'reward_received': 0, 
        'total_reward_agent1_so_far': 0,
        'total_reward_agent2_so_far': 0,
        'terminated': False,
        'truncated': False,
        'info': info.copy(),
        'round_step': env.current_step_in_episode, # Should be 0
        'L': env.L, 'H': env.H, 'walls': list(env.walls) # Grid params for replay
    })

    terminated = False
    truncated = False
    total_reward_agent1 = 0
    total_reward_agent2 = 0
    
    current_loop_iter = 0
    # max_loop_iters: agent1, agent2, (chicken is part of agent2's step) for each step in episode
    # plus a small buffer. Max steps is for full A1-A2-C rounds.
    max_loop_iters = env.max_episode_steps * 2 + 10 

    while not terminated and not truncated and current_loop_iter < max_loop_iters:
        current_player_idx_before_step = info.get("current_player_to_act", -1)
        action_taken_this_step = None
        reward_for_this_step = 0

        if current_player_idx_before_step == 0 or current_player_idx_before_step == 1: # Agent's turn
            action_taken_this_step = env.action_space.sample() 

            if render_episode_to_console:
                print(f"\n--- Round {env.current_step_in_episode + 1}, Agent {current_player_idx_before_step + 1} takes action: {action_taken_this_step} ---")

            obs, reward_for_this_step, terminated, truncated, info = env.step(action_taken_this_step)
            
            if current_player_idx_before_step == 0:
                total_reward_agent1 += reward_for_this_step
            else: # current_player_idx_before_step == 1
                total_reward_agent2 += reward_for_this_step

            if render_episode_to_console:
                print(env.render())
                print(f"Observation: {obs}")
                print(f"Reward for Agent {current_player_idx_before_step + 1}: {reward_for_this_step}")
                print(f"Terminated: {terminated}, Truncated: {truncated}")
                print(f"Info: {info}")
                print(f"Running Total Rewards: A1={total_reward_agent1}, A2={total_reward_agent2}")
        
        else: # Game ended in previous step or unexpected state. Loop condition (term/trunc) should handle.
            if render_episode_to_console:
                print(f"\n--- Loop will exit. Current state: Terminated={terminated}, Truncated={truncated}, Info: {info} ---")
            break 

        # Log state AFTER action (and potential chicken move if agent 2 acted)
        history_log.append({
            'grid_render_str': env.render(),
            'agent1_pos': env.agent1_pos,
            'agent2_pos': env.agent2_pos,
            'chicken_pos': env.chicken_pos,
            'acting_agent': current_player_idx_before_step, # Agent that JUST acted
            'action_taken': action_taken_this_step,
            'reward_received': reward_for_this_step,
            'total_reward_agent1_so_far': total_reward_agent1,
            'total_reward_agent2_so_far': total_reward_agent2,
            'terminated': terminated,
            'truncated': truncated,
            'info': info.copy(), 
            'round_step': env.current_step_in_episode, # Updated after A1, A2, C cycle
            'L': env.L, 'H': env.H, 'walls': list(env.walls)
        })
        
        current_loop_iter += 1
        if current_loop_iter >= max_loop_iters:
            if render_episode_to_console:
                print("Warning: Reached maximum loop iterations for the episode.")
            if not terminated and not truncated: # Force truncation if loop limit hit
                truncated = True
                history_log[-1]['truncated'] = True # Update last history entry
                if 'status' not in history_log[-1]['info']: history_log[-1]['info']['status'] = "max_loop_iters_reached"


    final_episode_length = env.current_step_in_episode # Number of full A1-A2-Chicken rounds completed
    if render_episode_to_console:
        print("\n--- Episode Ended ---")
        if terminated: print("Chicken caught!")
        elif truncated: print("Max episode steps reached or loop limit hit.")
        print(f"Final Rewards: Agent1={total_reward_agent1}, Agent2={total_reward_agent2}")
        print(f"Episode Length (full rounds): {final_episode_length}")

    return total_reward_agent1, total_reward_agent2, final_episode_length, history_log


if __name__ == '__main__':
    custom_walls = set([(1,1), (1,2), (2,1), (3,4), (4,3), (4,4), (5,1)])
    env_config = {
        "L": 7, "H": 7, "internal_wall_coords": custom_walls, "max_episode_steps": 10 # Short for testing
    }
    try:
        test_env = CooperativeChickenEnv(**env_config)
    except ValueError as e:
        print(f"Error initializing environment: {e}")
        exit()
    
    print("Running one test episode with history logging (console render OFF)...")
    r1, r2, length, history = run_random_agents_episode(test_env, render_episode_to_console=False)
    print(f"Episode finished. A1 Reward: {r1}, A2 Reward: {r2}, Length: {length} rounds.")
    print(f"Number of history steps recorded: {len(history)}")
    if history:
        print("Sample - First history step (initial state):")
        for key, val in history[0].items():
            if key not in ['grid_render_str', 'walls']: # Keep sample concise
                 print(f"  {key}: {val}")
        print("Sample - Last history step:")
        for key, val in history[-1].items():
            if key not in ['grid_render_str', 'walls']:
                 print(f"  {key}: {val}")

    # Example with console rendering enabled:
    # print("\nRunning one test episode with console rendering ON...")
    # test_env_render = CooperativeChickenEnv(**env_config) # Fresh env
    # r1_render, r2_render, length_render, history_render = run_random_agents_episode(test_env_render, render_episode_to_console=True)
    # print(f"\nRendered episode finished. A1 Reward: {r1_render}, A2 Reward: {r2_render}, Length: {length_render} rounds.")