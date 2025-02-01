import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import matplotlib.patches as patches

class VEXHighStakesEnv(gym.Env):
    def __init__(self):
        super(VEXHighStakesEnv, self).__init__()

        # Define observation space (robot position, orientation, goal & ring positions, has goal flag, top_goal_rings)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.pi] + [-1]*10 + [-1]*38 + [0, 0], dtype=np.float32),
            high=np.array([12, 12, np.pi] + [12]*10 + [12]*38 + [1, 10], dtype=np.float32)
        )
        # Define action space (drive-to, pick up goal, intake rings, climb, drop goal)
        # 0: drive-to, 1: pick up goal, 2: intake rings, 3: climb, 4: drop goal
        self.action_space = spaces.Discrete(5)

        # Initialize state variables
        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.robot_position = np.array([0.5, 6.0], dtype=np.float32)
        self.robot_orientation = 0.0  
        self.mobile_goal_positions = np.array([[4.0, 2.0], [4.0, 10.0], [8.0,4.0], [8.0, 8.0], [10.0, 6.0]], dtype=np.float32)
        # Add this line to track available goals (True if not picked up)
        self.goal_available = np.array([True] * len(self.mobile_goal_positions))
        self.ring_positions = np.array([
            [2.0, 2.0], [2.0, 6.0], [2.0, 10.0], [4.0, 4.0], [4.0, 8.0],
            [5.7, 5.7], [5.7, 6.3], [6.3, 5.7], [6.3, 6.3],
            [6.0, 2.0], [6.0, 10.0], [8.0, 2.0], [8.0, 10.0], [10.0, 2.0], [10.0, 10.0],
            [0.3,0.3], [11.7, 11.7], [11.7, 0.3], [0.3, 11.7]
            ], dtype=np.float32)
        self.has_goal = 0  
        self.carried_goal_index = None  # track which goal is held
        self.time_remaining = 120  
        self.rings_on_goals = [0, 0]
        self.top_goal_rings = 0  

        # Pad mobile_goal_positions flattened to length 10 (5 goals x 2)
        mobile_goals = self.mobile_goal_positions.flatten()
        if mobile_goals.size < 10:
            mobile_goals = np.pad(mobile_goals, (0, 10 - mobile_goals.size), 'constant', constant_values=(-1,))

        # Pad or truncate ring observations to length 38 (i.e., 19 rings x 2)
        rings_flat = self.ring_positions.flatten()
        if rings_flat.shape[0] < 38:
            rings_flat = np.pad(rings_flat, (0, 38 - rings_flat.shape[0]), 'constant', constant_values=(-1,))
        else:
            rings_flat = rings_flat[:38]

        # Ensure observation matches (53,)
        observation = np.concatenate([
            self.robot_position,                                  # (2,)
            [self.robot_orientation],                             # (1,)
            mobile_goals,                                         # (10,)
            rings_flat,                                           # (38,)
            [self.has_goal, self.top_goal_rings]                  # (2,)
        ], dtype=np.float32)

        return observation, {}

    def step(self, action):
        reward = self.reward_function(action)
        done = False
        truncated = False
        time_cost = 0.0

        if action == 0:  # Drive-to
            # Only drive to goals that are available and within the field (0-12)
            valid_idx = np.where(
                (self.goal_available) & 
                (self.mobile_goal_positions[:,0] >= 0) & (self.mobile_goal_positions[:,0] <= 12) &
                (self.mobile_goal_positions[:,1] >= 0) & (self.mobile_goal_positions[:,1] <= 12)
            )[0]
            if valid_idx.size > 0:
                target_idx = valid_idx[0]
                old_position = self.robot_position.copy()
                target_position = self.mobile_goal_positions[target_idx]
                direction = target_position - old_position
                self.robot_position = target_position.copy()
                self.robot_orientation = np.arctan2(direction[1], direction[0])
                distance = np.linalg.norm(target_position - old_position)
                time_cost = distance / 5.0
            else:
                pass  # No valid mobile goal available
        elif action == 1:  # Pick up goal
            time_cost = 0.5
        elif action == 2:  # Intake rings
            time_cost = 0.5
        elif action == 3:  # Climb
            time_cost = 10.0
        elif action == 4:  # Drop goal
            time_cost = 0.5

        self.time_remaining -= time_cost

        if self.time_remaining <= 0:
            done = True

        # Pad mobile_goal_positions flattened to length 10
        mobile_goals = self.mobile_goal_positions.flatten()
        if mobile_goals.size < 10:
            mobile_goals = np.pad(mobile_goals, (0, 10 - mobile_goals.size), 'constant', constant_values=(-1,))

        # Pad or truncate ring observations to length 38
        rings_flat = self.ring_positions.flatten()
        if rings_flat.shape[0] < 38:
            rings_flat = np.pad(rings_flat, (0, 38 - rings_flat.shape[0]), 'constant', constant_values=(-1,))
        else:
            rings_flat = rings_flat[:38]

        # Ensure observation has the correct shape (53,)
        observation = np.concatenate([
            self.robot_position,
            [self.robot_orientation],
            mobile_goals,                   # Use padded mobile_goal_positions
            rings_flat,  # Ensure exactly 38 ring position values
            [self.has_goal, self.top_goal_rings]
        ], dtype=np.float32)

        return observation, reward, done, truncated, {}

    def reward_function(self, action):
        reward = 0
        if action == 0:  # Drive-to
            reward = +0.001
        elif action == 1:  # Pick up goal
            # Only pick up if not already holding a goal
            if self.has_goal == 0:
                # Filter for available goals in the field
                valid_idx = np.where(
                    (self.goal_available) & 
                    (self.mobile_goal_positions[:,0] >= 0) & (self.mobile_goal_positions[:,0] <= 12) &
                    (self.mobile_goal_positions[:,1] >= 0) & (self.mobile_goal_positions[:,1] <= 12)
                )[0]
                if valid_idx.size > 0:
                    goal = self.mobile_goal_positions[valid_idx[0]]
                    distance = np.linalg.norm(goal - self.robot_position)
                    if distance < 1.0:
                        reward = +5
                        self.has_goal = 1
                        self.carried_goal_index = valid_idx[0]
                        # Mark this goal as picked up (make it unavailable)
                        self.goal_available[valid_idx[0]] = False
                    else:
                        reward = -10
                else:
                    reward = -5
            else:
                reward = -5
        elif action == 2:  # Intake rings
            # Only attempt pickup if a ring is available and robot is in proper contact with it
            if self.ring_positions.shape[0] > 0:
                ring = self.ring_positions[0]
                # Compute distance to ring
                distance = np.linalg.norm(ring - self.robot_position)
                # Compute direction to ring and compare with robot orientation
                ring_direction = np.arctan2(ring[1] - self.robot_position[1], ring[0] - self.robot_position[0])
                orientation_diff = np.abs(((ring_direction - self.robot_orientation + np.pi) % (2 * np.pi)) - np.pi)
                if distance < 1.0 and orientation_diff < np.pi/6:
                    if np.random.rand() < 0.2:
                        self.top_goal_rings += 1
                        reward += 3  # Top goal ring worth 3 points
                    else:
                        self.rings_on_goals[0] += 1
                        reward += 1  # Regular ring worth 1 point
                    # Remove ring after successful pickup
                    self.ring_positions = np.delete(self.ring_positions, 0, axis=0)
        elif action == 3:  # Climb
            if self.time_remaining <= 30:
                reward = +10  # Reward for climbing
            else:
                reward = -10  # Penalty for climbing too early
        elif action == 4:  # Drop goal
            if self.has_goal == 1:
                # Drop the carried goal at the robot's current position
                self.mobile_goal_positions[self.carried_goal_index] = self.robot_position.copy()
                # Mark the dropped goal as available again
                self.goal_available[self.carried_goal_index] = True
                self.has_goal = 0
                self.carried_goal_index = None
                reward = +1
            else:
                reward = -5
        return reward

    def render(self, mode='human', save_path=None, step_num=0, action=None):
        # Convert action from numpy array to int if necessary
        if isinstance(action, (np.ndarray,)):
            action = int(action)
        # Map action numbers to names
        action_names = {0: "drive-to", 1: "pick up goal", 2: "intake rings", 3: "climb", 4: "drop goal"}
        action_str = action_names.get(action, f"action {action}")
        # Convert robot position to string first before formatting
        robot_pos_str = f"({self.robot_position[0]:.2f}, {self.robot_position[1]:.2f})"
        print(f"Step: {step_num:<3} | {action_str:<15} | Reward: {self.reward_function(action):<5} | Time remaining: {self.time_remaining:<7.2f} | Robot position: {robot_pos_str}")
        # Create a plot
        fig, ax = plt.subplots()
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 12)
        ax.set_aspect('equal')  # Ensure the field is rendered as a square

        # Draw the robot as a square
        robot = patches.Rectangle(self.robot_position - 0.5, 1, 1, color='blue')
        ax.add_patch(robot)
        # Represent orientation with an arrow from the robot's center
        center = self.robot_position
        arrow_dx = np.cos(self.robot_orientation)
        arrow_dy = np.sin(self.robot_orientation)
        orientation_arrow = patches.FancyArrow(center[0], center[1], arrow_dx, arrow_dy, width=0.1, color='yellow', length_includes_head=True)
        ax.add_patch(orientation_arrow)

        # Draw mobile goals as hexagons
        for goal in self.mobile_goal_positions:
            hexagon = patches.RegularPolygon(goal, numVertices=6, radius=0.5, orientation=np.pi/6, color='green')
            ax.add_patch(hexagon)

        # Draw rings as circles
        for ring in self.ring_positions:
            circle = patches.Circle(ring, 0.3, color='red')
            ax.add_patch(circle)

        plt.title(f'Step {step_num}')
        if save_path:
            plt.savefig(f"{save_path}/step_{step_num}.png")
            plt.close()
        else:
            plt.show()

    def close(self):
        pass


# Check if the environment follows Gymnasium API
env = VEXHighStakesEnv()
check_env(env, warn=True)

# Train a PPO agent on the environment
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)

# Save the trained model
model.save("vex_high_stakes_ppo")

# Load and test the model
del model  # Remove to demonstrate loading
model = PPO.load("vex_high_stakes_ppo")

# Create a directory to save the images
save_path = "simulation_steps"
os.makedirs(save_path, exist_ok=True)

done = False
obs, _ = env.reset()
step_num = 0
images = []

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, _ = env.step(action)
    env.render(save_path=save_path, step_num=step_num, action=action)
    images.append(imageio.imread(f"{save_path}/step_{step_num}.png"))
    step_num += 1

imageio.mimsave('simulation.gif', images, fps=10)
# Create a GIF
imageio.mimsave('simulation.gif', images, fps=10)

env.close()

env.close()