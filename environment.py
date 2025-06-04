import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class CooperativeChickenEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}

    ACTION_STAY = 0
    ACTION_NORTH = 1
    ACTION_SOUTH = 2
    ACTION_WEST = 3
    ACTION_EAST = 4

    ACTION_DELTAS = {
        ACTION_STAY: (0, 0),
        ACTION_NORTH: (-1, 0), # Row decreases
        ACTION_SOUTH: (1, 0),  # Row increases
        ACTION_WEST: (0, -1),  # Col decreases
        ACTION_EAST: (0, 1),   # Col increases
    }

    def __init__(self, L=10, H=10, internal_wall_coords=None, max_episode_steps=100):
        super().__init__()

        if L < 1 or H < 1:
            raise ValueError("Grid dimensions L and H must be at least 1.")

        self.L = L  # Height (rows)
        self.H = H  # Width (cols)

        self.walls = set()
        if internal_wall_coords:
            for r, c in internal_wall_coords:
                if 0 <= r < self.L and 0 <= c < self.H:
                    self.walls.add((r, c))
                else:
                    raise ValueError(f"Internal wall coordinate ({r},{c}) is out of bounds for grid {L}x{H}.")

        # Check if enough space for entities
        num_available_cells = self.L * self.H - len(self.walls)
        if num_available_cells < 3:
            raise ValueError("Not enough available cells to place 2 agents and 1 chicken. "
                             "Grid size is too small or too many walls.")

        self.agent1_pos = None
        self.agent2_pos = None
        self.chicken_pos = None

        # Action space: 5 actions for each agent (Stay, N, S, W, E)
        self.action_space = spaces.Discrete(5)

        # Observation space:
        # [a1_r, a1_c, a2_r, a2_c, c_r, c_c]
        # The 'current_player_to_act' is implicitly managed by the environment's turn system.
        # An agent policy could be conditioned on its ID (0 or 1) if needed.
        low_bounds = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
        high_bounds = np.array([self.L - 1, self.H - 1,
                                self.L - 1, self.H - 1,
                                self.L - 1, self.H - 1], dtype=np.int32)
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.int32)

        self.current_player_idx = 0  # 0 for Agent 1, 1 for Agent 2
        self.max_episode_steps = max_episode_steps
        self.current_step_in_episode = 0 # Counts full A1-A2-Chicken cycles

        self.capture_reward = 100
        self.step_penalty = -1

        self.render_mode = 'ansi' # Default, can be changed by user

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _is_valid_pos(self, r, c):
        return 0 <= r < self.L and 0 <= c < self.H and (r, c) not in self.walls

    def _place_entities(self):
        available_cells = []
        for r_idx in range(self.L):
            for c_idx in range(self.H):
                if (r_idx, c_idx) not in self.walls:
                    available_cells.append((r_idx, c_idx))

        if len(available_cells) < 3: # Should have been caught in __init__
             raise Exception("Cannot place entities, not enough valid cells.")

        pos_a1, pos_a2, pos_c = random.sample(available_cells, 3)
        self.agent1_pos = tuple(pos_a1)
        self.agent2_pos = tuple(pos_a2)
        self.chicken_pos = tuple(pos_c)

    def _get_observation(self):
        return np.array([self.agent1_pos[0], self.agent1_pos[1],
                         self.agent2_pos[0], self.agent2_pos[1],
                         self.chicken_pos[0], self.chicken_pos[1]], dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Important for reproducibility via seeding
        self._place_entities()
        self.current_player_idx = 0  # Agent 1 starts
        self.current_step_in_episode = 0
        observation = self._get_observation()
        info = {"current_player_to_act": self.current_player_idx}
        return observation, info

    def _chicken_move(self):
        c_pos = self.chicken_pos
        a1_pos = self.agent1_pos
        a2_pos = self.agent2_pos

        valid_next_positions_deltas = []
        for action_val, (dr, dc) in self.ACTION_DELTAS.items():
            next_r, next_c = c_pos[0] + dr, c_pos[1] + dc
            if self._is_valid_pos(next_r, next_c):
                valid_next_positions_deltas.append(((next_r, next_c), (dr,dc)))

        if not valid_next_positions_deltas: # Should not happen if Stay is always possible
            return # Chicken stays if no valid moves (e.g. completely boxed in)

        d_c_a1_curr = self._manhattan_distance(c_pos, a1_pos)
        d_c_a2_curr = self._manhattan_distance(c_pos, a2_pos)

        scored_moves = [] # Stores ( (primary_dist_metric, secondary_dist_metric), next_pos_tuple )
        for next_pos, _ in valid_next_positions_deltas:
            nd1 = self._manhattan_distance(next_pos, a1_pos)
            nd2 = self._manhattan_distance(next_pos, a2_pos)
            scored_moves.append(((nd1, nd2), next_pos))

        # Determine sorting key based on which agent is closer or if tied
        if d_c_a1_curr < d_c_a2_curr: # A1 is closer, move away from A1
            # Maximize distance to A1. Tie-break randomly later.
            scored_moves.sort(key=lambda x: x[0][0], reverse=True)
            best_primary_score = scored_moves[0][0][0]
            candidate_moves = [m for m in scored_moves if m[0][0] == best_primary_score]

        elif d_c_a2_curr < d_c_a1_curr: # A2 is closer, move away from A2
            # Maximize distance to A2. Tie-break randomly later.
            scored_moves.sort(key=lambda x: x[0][1], reverse=True)
            best_primary_score = scored_moves[0][0][1]
            candidate_moves = [m for m in scored_moves if m[0][1] == best_primary_score]

        else: # Equidistant: move away from A1 (primary), then A2 (secondary)
            # Maximize distance to A1, then use distance to A2 as tie-breaker
            scored_moves.sort(key=lambda x: (x[0][0], x[0][1]), reverse=True)
            best_score_tuple = scored_moves[0][0]
            candidate_moves = [m for m in scored_moves if m[0] == best_score_tuple]

        if candidate_moves:
            # "Break these ties randomly if they are insufficient"
            chosen_move_score, chosen_next_pos = random.choice(candidate_moves)
            self.chicken_pos = chosen_next_pos
        else: # Should not be reached if Stay is a valid move
            pass # Chicken stays if logic somehow fails to find a move


    def step(self, action):
        if self.current_player_idx not in [0, 1]:
            raise ValueError("Invalid current_player_idx. Must be 0 (Agent 1) or 1 (Agent 2).")

        terminated = False
        truncated = False
        reward = self.step_penalty # Default step penalty for the acting agent

        # --- Agent's turn ---
        agent_pos_before_move = self.agent1_pos if self.current_player_idx == 0 else self.agent2_pos
        dr, dc = self.ACTION_DELTAS[action]
        new_r, new_c = agent_pos_before_move[0] + dr, agent_pos_before_move[1] + dc

        if self._is_valid_pos(new_r, new_c):
            if self.current_player_idx == 0:
                self.agent1_pos = (new_r, new_c)
            else: # Agent 2
                self.agent2_pos = (new_r, new_c)
        # Else: agent hits a wall or boundary, stays in place (pos doesn't change)

        # Check for capture
        if (self.current_player_idx == 0 and self.agent1_pos == self.chicken_pos) or \
           (self.current_player_idx == 1 and self.agent2_pos == self.chicken_pos):
            reward += self.capture_reward
            terminated = True

        if terminated:
            observation = self._get_observation()
            info = {"current_player_to_act": -1, "status": "capture"} # Game ended
            return observation, reward, terminated, truncated, info

        # --- Transition turn ---
        if self.current_player_idx == 0: # Agent 1 just moved
            self.current_player_idx = 1 # Now Agent 2's turn
            observation = self._get_observation()
            info = {"current_player_to_act": self.current_player_idx}
            return observation, reward, terminated, truncated, info
        else: # Agent 2 just moved
            # --- Chicken's turn ---
            self._chicken_move()
            # Capture is only by agent moving to chicken's square, not other way around per prompt

            self.current_player_idx = 0 # Back to Agent 1 for next round
            self.current_step_in_episode += 1 # A full round (A1, A2, C) is completed

        # Check for truncation (max steps)
        if self.current_step_in_episode >= self.max_episode_steps:
            truncated = True
            # No special reward for truncation itself beyond penalties already applied

        observation = self._get_observation()
        info = {"current_player_to_act": self.current_player_idx}
        if truncated:
            info["status"] = "truncated"

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'ansi':
            grid_repr = [["." for _ in range(self.H)] for _ in range(self.L)]
            for r_wall, c_wall in self.walls:
                grid_repr[r_wall][c_wall] = "#"

            if self.chicken_pos: grid_repr[self.chicken_pos[0]][self.chicken_pos[1]] = "C"
            # Agents overwrite chicken if on same spot for rendering order
            if self.agent1_pos: grid_repr[self.agent1_pos[0]][self.agent1_pos[1]] = "1"
            if self.agent2_pos: grid_repr[self.agent2_pos[0]][self.agent2_pos[1]] = "2"
            # If A1 and A2 on same spot (unlikely if they don't move simultaneously to same empty cell)
            if self.agent1_pos and self.agent2_pos and self.agent1_pos == self.agent2_pos:
                grid_repr[self.agent1_pos[0]][self.agent1_pos[1]] = "X" # Both agents

            output = "\n".join([" ".join(row) for row in grid_repr])
            output += f"\nTurn: Agent {self.current_player_idx + 1 if self.current_player_idx in [0,1] else 'Chicken/End'}"
            output += f" | Round Step: {self.current_step_in_episode}/{self.max_episode_steps}"
            output += f"\nA1: {self.agent1_pos}, A2: {self.agent2_pos}, C: {self.chicken_pos}"
            return output
        elif self.render_mode == 'human':
            # For a graphical representation, you'd typically use a library like Pygame.
            # This is a placeholder for human-readable console output.
            print(self.render())


if __name__ == '__main__':
    # Example Usage:
    # Define some internal walls (optional)
    # Walls are (row, col) tuples
    custom_walls = set([(1,1), (1,2), (1,3), (3,3), (3,4), (4,4), (5,4)])

    env = CooperativeChickenEnv(L=7, H=7, internal_wall_coords=custom_walls, max_episode_steps=50)
    obs, info = env.reset()
    print("Initial State:")
    print(env.render())
    print(f"Observation: {obs}")
    print(f"Info: {info}")

    terminated = False
    truncated = False
    total_reward_agent1 = 0
    total_reward_agent2 = 0

    # Simulate a few random steps
    for i in range(100): # Max iterations for this demo
        if terminated or truncated:
            break

        current_agent_id = info["current_player_to_act"]
        action = env.action_space.sample() # Random action

        print(f"\n--- Round {env.current_step_in_episode + 1}, Agent {current_agent_id + 1} takes action: {action} ---")

        obs, reward, terminated, truncated, info = env.step(action)

        if current_agent_id == 0:
            total_reward_agent1 += reward
        else:
            total_reward_agent2 += reward

        print(env.render())
        print(f"Observation: {obs}")
        print(f"Reward for Agent {current_agent_id + 1}: {reward}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        print(f"Info: {info}")
        print(f"Total Rewards: A1={total_reward_agent1}, A2={total_reward_agent2}")

    print("\n--- Simulation Ended ---")
    if terminated:
        print("Chicken caught!")
    elif truncated:
        print("Max episode steps reached.")
    print(f"Final Rewards: Agent1={total_reward_agent1}, Agent2={total_reward_agent2}")