import cv2
import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris

# Create the Tetris environment.
# - "tetris_gymnasium/Tetris": environment name
# - render_mode="human": rendering mode
#   * "human": show a visible game window
#   * "rgb_array": return image arrays for training or recording
#   * None: disable rendering for batch training
env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
# Reset the environment to its initial state.
# - seed: set a random seed so the result is reproducible
#   The same seed generates the same game sequence, including piece order.
env.reset(seed=42)

# Game status flags.
# - terminated: whether the game ended normally, such as pieces reaching the top
# - truncated: whether the game ended early due to external limits
terminated = False
truncated = False

while not terminated:
    # Render or generate the current game state based on render_mode.
    # In "human" mode, this displays the Tetris game window.
    env.render()

    # Sample a random action from the action space.
    # action_space defines all possible actions.
    action = env.action_space.sample()

    # step() executes the action and returns a five-item tuple:
    #   - observation: current game state, usually a dictionary
    #   - reward: reward obtained at this step
    #   - terminated: whether the game has ended
    #   - truncated: whether the episode was cut short by a limit
    #   - info: extra information dictionary
    observation, reward, terminated, truncated, info = env.step(action)

    # cv2.waitKey(100) waits for 100 milliseconds.
    key= cv2.waitKey(100)

print("Game Over!")

env.close()
