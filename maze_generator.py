import numpy as np

def generate_maze(height, width):
    """Generate a random maze as a 2D numpy array."""
    maze = np.ones((height, width), dtype=np.uint8)  # Start with a fully blocked maze
    
    # Carve out a simple random maze with open paths (0's)
    for i in range(height):
        for j in range(width):
            if np.random.rand() < 0.7:  # 70% chance for a free space (path)
                maze[i, j] = 0
    # Ensure the start and goal are open
    maze[0, 0] = 0  # Start
    maze[height - 1, width - 1] = 0  # Goal
    return maze
