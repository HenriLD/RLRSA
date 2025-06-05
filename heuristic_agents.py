import gymnasium as gym
import numpy as np
from environment import CooperativeChickenEnv

def _manhattan_distance(pos1, pos2):
    """Helper function to calculate Manhattan distance."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def choose_heuristic_action(agent_pos, chicken_pos, env):
    """
    Chooses an action to minimize Manhattan distance to the chicken.
    Args:
        agent_pos (tuple): Current position of the agent (r, c).
        chicken_pos (tuple): Current position of the chicken (r, c).
        env (CooperativeChickenEnv): The environment instance.

    Returns:
        int: The chosen action.
    """
    best_action = env.ACTION_STAY # Default to staying
    min_dist = _manhattan_distance(agent_pos, chicken_pos)

    candidate_actions = [] # To store (distance, action)

    for action_val, (dr, dc) in env.ACTION_DELTAS.items():
        next_r, next_c = agent_pos[0] + dr, agent_pos[1] + dc
        
        # Check if the *intended* next position is valid.
        # The environment's step function will handle final validity (e.g. if agent hits wall, it stays).
        # Here, we are interested in the heuristic value of the *intended* move.
        # We don't need to check env._is_valid_pos explicitly for action choice,
        # as the heuristic is based on intended position. The env handles consequences.
        
        potential_pos = (next_r, next_c)
        dist = _manhattan_distance(potential_pos, chicken_pos)
        candidate_actions.append((dist, action_val))

    if not candidate_actions:
        return env.ACTION_STAY # Should not happen if ACTION_STAY is always an option

    # Sort by distance (ascending), then by action value (arbitrary tie-break)
    candidate_actions.sort(key=lambda x: (x[0], x[1]))
    
    # Choose the action that leads to the minimum distance
    best_action = candidate_actions[0][1]
    
    return best_action


def run_heuristic_agents_episode(env, render_episode_to_console=False):
    """
    Runs a single episode with two heuristic agents.

    Args:
        env: An instance of the CooperativeChickenEnv.
        render_episode_to_console (bool): Whether to print the state of the environment to console.

    Returns:
        tuple: (total_reward_agent1, total_reward_agent2, episode_length, history_log)
               history_log is a list of dictionaries, each detailing a step.
    """
    obs, info = env.reset()
    if render_episode_to_console:
        print("--- Initial State (Heuristic Agents) ---")
        print(env.render())
        print(f"Observation: {obs}")
        print(f"Info: {info}")

    history_log = []
    history_log.append({
        'grid_render_str': env.render(),
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
        'round_step': env.current_step_in_episode,
        'L': env.L, 'H': env.H, 'walls': list(env.walls)
    })

    terminated = False
    truncated = False
    total_reward_agent1 = 0
    total_reward_agent2 = 0
    
    current_loop_iter = 0
    max_loop_iters = env.max_episode_steps * 2 + 10

    while not terminated and not truncated and current_loop_iter < max_loop_iters:
        current_player_idx_before_step = info.get("current_player_to_act", -1)
        action_taken_this_step = None
        reward_for_this_step = 0

        if current_player_idx_before_step == 0 or current_player_idx_before_step == 1:
            agent_pos = env.agent1_pos if current_player_idx_before_step == 0 else env.agent2_pos
            chicken_pos = env.chicken_pos
            
            action_taken_this_step = choose_heuristic_action(agent_pos, chicken_pos, env)

            if render_episode_to_console:
                print(f"\n--- Round {env.current_step_in_episode + 1}, Agent {current_player_idx_before_step + 1} (Heuristic) takes action: {action_taken_this_step} ---")

            obs, reward_for_this_step, terminated, truncated, info = env.step(action_taken_this_step)
            
            if current_player_idx_before_step == 0:
                total_reward_agent1 += reward_for_this_step
            else:
                total_reward_agent2 += reward_for_this_step

            if render_episode_to_console:
                print(env.render())
                print(f"Observation: {obs}")
                print(f"Reward for Agent {current_player_idx_before_step + 1}: {reward_for_this_step}")
                print(f"Terminated: {terminated}, Truncated: {truncated}")
                print(f"Info: {info}")
                print(f"Running Total Rewards: A1={total_reward_agent1}, A2={total_reward_agent2}")
        
        else:
            if render_episode_to_console:
                print(f"\n--- Loop will exit. Current state: Terminated={terminated}, Truncated={truncated}, Info: {info} ---")
            break

        history_log.append({
            'grid_render_str': env.render(),
            'agent1_pos': env.agent1_pos,
            'agent2_pos': env.agent2_pos,
            'chicken_pos': env.chicken_pos,
            'acting_agent': current_player_idx_before_step,
            'action_taken': action_taken_this_step,
            'reward_received': reward_for_this_step,
            'total_reward_agent1_so_far': total_reward_agent1,
            'total_reward_agent2_so_far': total_reward_agent2,
            'terminated': terminated,
            'truncated': truncated,
            'info': info.copy(),
            'round_step': env.current_step_in_episode,
            'L': env.L, 'H': env.H, 'walls': list(env.walls)
        })
        
        current_loop_iter += 1
        if current_loop_iter >= max_loop_iters:
            if render_episode_to_console:
                print("Warning: Reached maximum loop iterations for the episode (Heuristic).")
            if not terminated and not truncated:
                truncated = True
                history_log[-1]['truncated'] = True
                if 'status' not in history_log[-1]['info']: history_log[-1]['info']['status'] = "max_loop_iters_reached"

    final_episode_length = env.current_step_in_episode
    if render_episode_to_console:
        print("\n--- Episode Ended (Heuristic Agents) ---")
        if terminated: print("Chicken caught!")
        elif truncated: print("Max episode steps reached or loop limit hit.")
        print(f"Final Rewards: Agent1={total_reward_agent1}, Agent2={total_reward_agent2}")
        print(f"Episode Length (full rounds): {final_episode_length}")

    return total_reward_agent1, total_reward_agent2, final_episode_length, history_log


if __name__ == '__main__':
    custom_walls = set([(1,1), (1,2), (2,1), (3,4), (4,3), (4,4), (5,1)])
    env_config = {
        "L": 7, "H": 7, "internal_wall_coords": custom_walls, "max_episode_steps": 20 # Short for testing
    }
    try:
        test_env = CooperativeChickenEnv(**env_config)
    except ValueError as e:
        print(f"Error initializing environment: {e}")
        exit()
    
    print("Running one test episode with Heuristic agents (console render OFF)...")
    r1, r2, length, history = run_heuristic_agents_episode(test_env, render_episode_to_console=False)
    print(f"Episode finished. A1 Reward: {r1}, A2 Reward: {r2}, Length: {length} rounds.")
    print(f"Number of history steps recorded: {len(history)}")
    if history:
        print("Sample - First history step (initial state):")
        for key, val in history[0].items():
            if key not in ['grid_render_str', 'walls']:
                 print(f"  {key}: {val}")
        print("Sample - Last history step:")
        for key, val in history[-1].items():
            if key not in ['grid_render_str', 'walls']:
                 print(f"  {key}: {val}")

    print("\nRunning one test episode with Heuristic agents (console render ON)...")
    test_env_render = CooperativeChickenEnv(**env_config) # Fresh env
    r1_render, r2_render, length_render, history_render = run_heuristic_agents_episode(test_env_render, render_episode_to_console=True)
    print(f"\nRendered episode finished. A1 Reward: {r1_render}, A2 Reward: {r2_render}, Length: {length_render} rounds.")