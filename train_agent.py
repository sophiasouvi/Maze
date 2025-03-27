import gymnasium as gym
from stable_baselines3 import DQN
from maze_env import MazeEnv
import numpy as np
import matplotlib.pyplot as plt

# Initialize the environment
env = MazeEnv(height=11, width=11)

# Initialize DQN model with correct hyperparameters for learning
model = DQN(
    "MlpPolicy", 
    env, 
    verbose=1,
    learning_rate=0.0005,  # Reduced learning rate
    buffer_size=100000,  # Size of the experience replay buffer
    learning_starts=10,  # Wait for experience to accumulate before starting to train
    batch_size=64,  # Size of the training batch
    gamma=0.95,  # Discount factor for future rewards
    train_freq=4,  # Train after every 4 steps
    gradient_steps=1,  # Number of gradient steps per update
    exploration_fraction=0.1,  # Fraction of total training to be spent exploring
    exploration_initial_eps=1.0,  # Initial exploration rate (100% exploration)
    exploration_final_eps=0.05,  # Final exploration rate (5% exploration)
    target_update_interval=100,  # How often to update the target network
    tensorboard_log="./dqn_maze_tensorboard/"
)

# List to store rewards for plotting the reward curve
episode_rewards = []
episode_lengths = []

# Train the agent
num_episodes = 500
for episode in range(num_episodes):
    obs, _ = env.reset()  # Reset environment
    done = False
    episode_reward = 0
    episode_length = 0

    while not done:
        # Take an action and get the next observation, reward, and done signal
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        
        # Update reward and episode length
        episode_reward += reward
        episode_length += 1

        # If the agent reaches the goal, exit the loop
        if done:
            break

    # Append the results for each episode
    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_length)

    # Log the progress (optional)
    if episode % 10 == 0:
        print(f"Episode {episode}/{num_episodes} - Reward: {episode_reward}, Length: {episode_length}")

    # Train the model after each episode
    model.learn(total_timesteps=1000)

# Save the trained model
model.save("dqn_maze_model")

# Plot reward vs. episode
plt.plot(range(num_episodes), episode_rewards)
plt.title("Reward vs. Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

# Evaluate the trained model
total_rewards = []
for episode in range(10):  # Evaluate for 10 episodes
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
    total_rewards.append(total_reward)

# Print the average reward
avg_reward = np.mean(total_rewards)
print(f"Average reward over 10 evaluation episodes: {avg_reward}")
