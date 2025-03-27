import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from maze_generator import generate_maze  # Import random maze generator

class MazeEnv(gym.Env):
    """Custom Maze Environment for Reinforcement Learning using Random Mazes."""

    def __init__(self, height=11, width=11):
        super(MazeEnv, self).__init__()
        self.height = height
        self.width = width

        # Define action space: 4 possible moves (Up, Down, Left, Right)
        self.action_space = gym.spaces.Discrete(4)

        # Define observation space: 2D grid representing the maze
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(height, width), dtype=np.uint8
        )

        # Initial positions and goal
        self.agent_pos = [0, 0]  # Start at top-left
        self.goal_pos = [height - 1, width - 1]  # Goal at bottom-right
        self.max_steps = 500  # Max steps per episode
        self.steps_taken = 0

        # Generate initial maze
        self._generate_maze()

    def _generate_maze(self):
        """Generate a random maze using maze_generator."""
        self.maze = generate_maze(self.height, self.width)  # Get random maze from maze_generator
        # Set start and goal positions to be empty
        self.maze[0, 0] = 0
        self.maze[self.height - 1, self.width - 1] = 0

    def _move_agent(self, action):
        """Move agent based on action and return new position."""
        x, y = self.agent_pos
        if action == 0 and x > 0 and self.maze[x - 1, y] == 0:  # Up
            x -= 1
        elif action == 1 and x < self.height - 1 and self.maze[x + 1, y] == 0:  # Down
            x += 1
        elif action == 2 and y > 0 and self.maze[x, y - 1] == 0:  # Left
            y -= 1
        elif action == 3 and y < self.width - 1 and self.maze[x, y + 1] == 0:  # Right
            y += 1
        return [x, y]

    def step(self, action):
        """Take action and return new state, reward, and done."""
        self.steps_taken += 1

        # Move agent and get new position
        next_state = self._move_agent(action)

        # Default penalty for taking a step
        reward = -0.01

        # Goal reward if agent reaches goal
        if next_state == self.goal_pos:
            reward = 100  # Goal reached, success!
            done = True
        elif self.steps_taken >= self.max_steps:
            done = True
        else:
            # Small positive reward for moving closer to the goal
            if self.distance_to_goal(next_state) < self.distance_to_goal(self.agent_pos):
                reward += 1
            done = False

        # Update agent's position
        self.agent_pos = next_state
        return self._get_observation(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state and regenerate maze."""
        self.agent_pos = [0, 0]
        self.steps_taken = 0
        self._generate_maze()  # Generate a new random maze on reset
        return self._get_observation(), {}

    def _get_observation(self):
        """Return current observation with agent and goal positions."""
        obs = self.maze.copy()
        agent_x, agent_y = self.agent_pos
        goal_x, goal_y = self.goal_pos
        obs[agent_x, agent_y] = 2  # Mark agent position
        obs[goal_x, goal_y] = 3  # Mark goal position
        return np.array(obs)  # Ensure the observation is a numpy array

    def distance_to_goal(self, state):
        """Returns Manhattan distance from agent to goal."""
        goal_x, goal_y = self.goal_pos
        agent_x, agent_y = state
        return abs(goal_x - agent_x) + abs(goal_y - agent_y)

    def render(self, mode="human"):
        """Render the maze and agent position."""
        maze_copy = self._get_observation()
        plt.imshow(maze_copy, cmap="coolwarm", interpolation="nearest")
        plt.title("Maze Environment")
        plt.show()

    def close(self):
        """Close the environment."""
        plt.close()
