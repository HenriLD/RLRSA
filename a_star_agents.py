import heapq # For priority queue in A*
import numpy as np
from environment import CooperativeChickenEnv # Assuming environment.py is in the same directory

def _manhattan_distance(pos1, pos2):
    """Helper function to calculate Manhattan distance."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class Node:
    """Node class for A* search."""
    def __init__(self, position, parent=None, action_from_parent=None):
        self.position = position
        self.parent = parent
        self.action_from_parent = action_from_parent # The action that led to this node
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic cost from current node to goal
        self.f = 0  # Total cost (g + h)

    def __lt__(self, other): # For priority queue comparison
        return self.f < other.f

    def __eq__(self, other): # For checking if a node is in a list/set
        return self.position == other.position

def a_star_search(start_pos, goal_pos, env):
    """
    A* search to find the best next action towards the goal_pos.
    Args:
        start_pos (tuple): Agent's current position (r, c).
        goal_pos (tuple): Chicken's current position (r, c).
        env (CooperativeChickenEnv): The environment instance.
    Returns:
        int: The first action (from env.ACTION_DELTAS) to take on the optimal path.
             Returns ACTION_STAY if no path is found or if already at goal.
    """
    start_node = Node(start_pos)
    start_node.h = _manhattan_distance(start_pos, goal_pos)
    start_node.f = start_node.h

    open_list = []
    heapq.heappush(open_list, start_node)
    closed_set = set()

    action_map = {v: k for k, v in env.ACTION_DELTAS.items()} # (dr, dc) -> action_id

    while open_list:
        current_node = heapq.heappop(open_list)

        if current_node.position == goal_pos:
            # Reconstruct path to find the first action
            path = []
            temp = current_node
            while temp.parent is not None:
                path.append(temp.action_from_parent)
                temp = temp.parent
            if not path: # Already at goal or only one step was needed
                if current_node.action_from_parent is not None:
                    return current_node.action_from_parent
                return env.ACTION_STAY # Already at goal, or only option is to stay
            return path[-1] # The first action taken from start_node

        closed_set.add(current_node.position)

        for action_val, (dr, dc) in env.ACTION_DELTAS.items():
            neighbor_pos = (current_node.position[0] + dr, current_node.position[1] + dc)

            if not env._is_valid_pos(neighbor_pos[0], neighbor_pos[1]): # Check if valid (not wall, in bounds)
                continue
            if neighbor_pos in closed_set:
                continue

            neighbor_node = Node(neighbor_pos, current_node, action_val)
            neighbor_node.g = current_node.g + 1 # Cost of each step is 1
            neighbor_node.h = _manhattan_distance(neighbor_pos, goal_pos)
            neighbor_node.f = neighbor_node.g + neighbor_node.h

            # Check if neighbor is in open_list and if this path is better
            found_in_open = False
            for i, open_node in enumerate(open_list):
                if open_node == neighbor_node:
                    found_in_open = True
                    if neighbor_node.g < open_node.g:
                        open_list[i] = neighbor_node # Update with better path
                        heapq.heapify(open_list) # Re-sort
                    break
            
            if not found_in_open:
                heapq.heappush(open_list, neighbor_node)
                
    return env.ACTION_STAY # No path found, stay put


def run_a_star_agents_episode(env, render_episode_to_console=False):
    """
    Runs a single episode with two A* agents.
    """
    obs, info = env.reset()
    if render_episode_to_console:
        print("--- Initial State (A* Agents) ---")
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
            
            if agent_pos == chicken_pos: # Already on chicken, try to stay or best guess if forced to move
                 action_taken_this_step = env.ACTION_STAY
            else:
                action_taken_this_step = a_star_search(agent_pos, chicken_pos, env)

            if render_episode_to_console:
                print(f"\n--- Round {env.current_step_in_episode + 1}, Agent {current_player_idx_before_step + 1} (A*) takes action: {action_taken_this_step} ---")

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
                print("Warning: Reached maximum loop iterations for the episode (A*).")
            if not terminated and not truncated:
                truncated = True
                history_log[-1]['truncated'] = True
                if 'status' not in history_log[-1]['info']: history_log[-1]['info']['status'] = "max_loop_iters_reached"

    final_episode_length = env.current_step_in_episode
    if render_episode_to_console:
        print("\n--- Episode Ended (A* Agents) ---")
        if terminated: print("Chicken caught!")
        elif truncated: print("Max episode steps reached or loop limit hit.")
        print(f"Final Rewards: Agent1={total_reward_agent1}, Agent2={total_reward_agent2}")
        print(f"Episode Length (full rounds): {final_episode_length}")

    return total_reward_agent1, total_reward_agent2, final_episode_length, history_log

if __name__ == '__main__':
    # L=7, H=7, custom_walls = set([(1,1), (1,2), (2,1), (3,4), (4,3), (4,4), (5,1)])
    # Simpler test case:
    test_walls = set([(2,0), (2,1), (2,2), (2,3), (2,4)]) # A horizontal wall
    env_config = {
        "L": 5, "H": 5, "internal_wall_coords": test_walls, "max_episode_steps": 30
    }
    try:
        test_env = CooperativeChickenEnv(**env_config)
    except ValueError as e:
        print(f"Error initializing environment: {e}")
        exit()
    
    print("Running one test episode with A* agents (console render OFF)...")
    r1, r2, length, history = run_a_star_agents_episode(test_env, render_episode_to_console=False)
    print(f"Episode finished. A1 Reward: {r1}, A2 Reward: {r2}, Length: {length} rounds.")
    print(f"Number of history steps recorded: {len(history)}")
    if history:
        print("Sample - First history step (initial state):")
        for key, val in history[0].items():
            if key not in ['grid_render_str', 'walls']: print(f"  {key}: {val}")
        print("Sample - Last history step:")
        for key, val in history[-1].items():
            if key not in ['grid_render_str', 'walls']: print(f"  {key}: {val}")

    print("\nRunning one test episode with A* agents (console render ON)...")
    test_env_render = CooperativeChickenEnv(**env_config)
    r1_render, r2_render, length_render, history_render = run_a_star_agents_episode(test_env_render, render_episode_to_console=True)
    print(f"\nRendered episode finished. A1 Reward: {r1_render}, A2 Reward: {r2_render}, Length: {length_render} rounds.")