# random_agents.py
import gymnasium as gym
from environment import CooperativeChickenEnv # Assuming environment.py is in the same directory or accessible in PYTHONPATH

def run_random_agents_episode(env, render_episode=True):
    """
    Runs a single episode with two random agents.

    Args:
        env: An instance of the CooperativeChickenEnv.
        render_episode (bool): Whether to print the state of the environment at each step.

    Returns:
        tuple: (total_reward_agent1, total_reward_agent2, episode_length)
    """
    obs, info = env.reset()
    if render_episode:
        print("--- Initial State ---")
        print(env.render())
        print(f"Observation: {obs}")
        print(f"Info: {info}")

    terminated = False
    truncated = False
    total_reward_agent1 = 0
    total_reward_agent2 = 0
    episode_length = 0 # Counts full A1-A2-Chicken cycles

    # The main loop runs as long as the episode is not done.
    # The environment's step function handles the turn progression internally.
    # We just need to ensure we're passing actions for the correct agent when it's their turn.
    # The 'info' dict tells us whose turn it is.

    current_loop_iter = 0 # To prevent infinite loops in case of unexpected issues
    max_loop_iters = env.max_episode_steps * 3 + 10 # A safe upper bound for iterations

    while not terminated and not truncated and current_loop_iter < max_loop_iters:
        current_player_idx = info.get("current_player_to_act", -1)

        if current_player_idx == 0 or current_player_idx == 1: # Agent 1 or Agent 2's turn
            action = env.action_space.sample() # Choose a random action

            if render_episode:
                print(f"\n--- Round {env.current_step_in_episode + 1}, Agent {current_player_idx + 1} takes action: {action} ---")

            obs, reward, terminated, truncated, info = env.step(action)
            episode_length = env.current_step_in_episode # Update based on full rounds completed

            if current_player_idx == 0:
                total_reward_agent1 += reward
            else: # current_player_idx == 1
                total_reward_agent2 += reward

            if render_episode:
                print(env.render())
                print(f"Observation: {obs}")
                print(f"Reward for Agent {current_player_idx + 1}: {reward}")
                print(f"Terminated: {terminated}, Truncated: {truncated}")
                print(f"Info: {info}")
                print(f"Running Total Rewards: A1={total_reward_agent1}, A2={total_reward_agent2}")
        else:
            # This case should ideally not be reached if current_player_to_act is always 0 or 1
            # when step is called by the agent script.
            # If the game terminated in the previous step, the loop condition will catch it.
            if render_episode:
                print(f"\n--- Game ended or unexpected state: {info} ---")
            break # Exit loop if game ended or state is unexpected

        current_loop_iter += 1
        if current_loop_iter >= max_loop_iters and render_episode:
            print("Warning: Reached maximum loop iterations for the episode.")


    if render_episode:
        print("\n--- Episode Ended ---")
        if terminated:
            print("Chicken caught!")
        elif truncated:
            print("Max episode steps reached.")
        print(f"Final Rewards: Agent1={total_reward_agent1}, Agent2={total_reward_agent2}")
        print(f"Episode Length (full rounds): {episode_length}")

    return total_reward_agent1, total_reward_agent2, episode_length


if __name__ == '__main__':
    # Define some internal walls (optional)
    # Walls are (row, col) tuples
    custom_walls = set([(1,1), (1,2), (2,1), (3,4), (4,3), (4,4), (5,1)])

    # Create the environment instance
    # You can adjust L, H, walls, and max_episode_steps as needed
    env_config = {
        "L": 7,
        "H": 7,
        "internal_wall_coords": custom_walls,
        "max_episode_steps": 50
    }
    try:
        env = CooperativeChickenEnv(**env_config)
    except ValueError as e:
        print(f"Error initializing environment: {e}")
        exit()

    num_episodes_to_run = 3
    all_rewards_a1 = []
    all_rewards_a2 = []
    all_lengths = []

    print(f"Running {num_episodes_to_run} episodes with random agents...")

    for i in range(num_episodes_to_run):
        print(f"\n\n<<<<<<<<<< Starting Episode {i + 1} >>>>>>>>>>")
        # Set render_episode to True for the first episode, False for others to speed up
        r1, r2, length = run_random_agents_episode(env, render_episode=(i == 0))
        all_rewards_a1.append(r1)
        all_rewards_a2.append(r2)
        all_lengths.append(length)
        print(f"<<<<<<<<<< Episode {i + 1} Finished. Rewards: A1={r1}, A2={r2}. Length: {length} >>>>>>>>>>")

    env.close() # Good practice to close the environment

    print("\n\n--- Simulation Summary ---")
    print(f"Number of episodes: {num_episodes_to_run}")
    if num_episodes_to_run > 0:
        print(f"Average Reward Agent 1: {sum(all_rewards_a1) / num_episodes_to_run:.2f}")
        print(f"Average Reward Agent 2: {sum(all_rewards_a2) / num_episodes_to_run:.2f}")
        print(f"Average Episode Length: {sum(all_lengths) / num_episodes_to_run:.2f} rounds")

        captures = sum(1 for r1, r2 in zip(all_rewards_a1, all_rewards_a2) if env.capture_reward + env.step_penalty in [r1, r2] or env.capture_reward in [r1,r2]) # Approximate check
        print(f"Approximate number of captures: {captures} out of {num_episodes_to_run}")
