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
        max_goals = 5
        max_rings = 24  # updated from 19 to 24 rings
        # Updated observation_space with ring_status array, and holding_goal remains as before
        self.observation_space = spaces.Dict({
            "robot_x": spaces.Box(low=0, high=12, shape=(1,), dtype=np.float32),
            "robot_y": spaces.Box(low=0, high=12, shape=(1,), dtype=np.float32),
            "robot_orientation": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            "holding_goal": spaces.Box(low=0, high=1, shape=(max_goals,), dtype=np.int32),
            "holding_rings": spaces.Discrete(3),
            "rings": spaces.Box(low=-1, high=12, shape=(max_rings*2,), dtype=np.float32),  # modified shape to be flat
            "goals": spaces.Box(low=-1, high=12, shape=(max_goals*2,), dtype=np.float32),  # modified to be flat
            "ring_status": spaces.Box(low=0, high=6, shape=(max_rings,), dtype=np.int32)  # 0: ground, 1: robot, 2-6: attached to goal index (goal+2)
        })
        # Update action_space from 14 to 15 to add "drive to nearest ring" action
        self.action_space = spaces.Discrete(15)
        # Initialize state variables
        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.robot_position = np.array([0.5, 6.0], dtype=np.float32)
        self.robot_orientation = 0.0  
        self.mobile_goal_positions = np.array([[4.0, 2.0], [4.0, 10.0], [8.0,4.0], [8.0, 8.0], [10.0, 6.0]], dtype=np.float32)
        self.goal_available = np.array([True] * len(self.mobile_goal_positions))
        # Set ring_positions and initialize ring_status (all rings start on ground: 0)
        self.ring_positions = np.array([
            [6.0, 1.0], [6.0, 11.0],
            [2.0, 2.0], [2.0, 6.0], [2.0, 10.0], [4.0, 4.0], [4.0, 8.0],
            [5.7, 5.7], [5.7, 6.3], [6.3, 5.7], [6.3, 6.3],
            [6.0, 2.0], [6.0, 10.0], [8.0, 2.0], [8.0, 10.0], [10.0, 2.0], [10.0, 10.0],
            [0.3, 0.3], [11.7, 11.7], [11.7, 0.3], [0.3, 11.7],
            [10.0, 4.0], [10.0, 8.0], [11.0, 6.0]  # new rings added here
            ], dtype=np.float32)
        self.ring_status = np.zeros(self.ring_positions.shape[0], dtype=np.int32)  # all rings on ground
        # Replace has_goal and carried_goal_index with holding_goal array (0: not holding, 1: holding)
        self.holding_goal = np.zeros(self.mobile_goal_positions.shape[0], dtype=np.int32)
        self.time_remaining = 120  
        # Remove old ring counters and use a single counter for rings held.
        self.holding_rings = 0  

        # Pad mobile_goal_positions flattened to length 10 (5 goals x 2)
        mobile_goals = self.mobile_goal_positions.flatten()
        if mobile_goals.size < 10:
            mobile_goals = np.pad(mobile_goals, (0, 10 - mobile_goals.size), 'constant', constant_values=(-1,))

        # Pad or truncate ring observations to length 48 (i.e., 24 rings x 2)
        rings_flat = self.ring_positions.flatten()
        if rings_flat.shape[0] < 48:
            rings_flat = np.pad(rings_flat, (0, 48 - rings_flat.shape[0]), 'constant', constant_values=(-1,))
        else:
            rings_flat = rings_flat[:48]

        # Ensure observation matches (63,)
        max_goals = 5
        max_rings = 24
        padded_goals = self.mobile_goal_positions.copy()
        if padded_goals.shape[0] < max_goals:
            pad_count = max_goals - padded_goals.shape[0]
            padded_goals = np.pad(padded_goals, ((0, pad_count),(0,0)), 'constant', constant_values=-1)
        else:
            padded_goals = padded_goals[:max_goals]
        padded_goals = padded_goals.flatten()  # modified: flatten goals
        padded_rings = self.ring_positions.copy()
        if padded_rings.shape[0] < max_rings:
            pad_count = max_rings - padded_rings.shape[0]
            padded_rings = np.pad(padded_rings, ((0, pad_count),(0,0)), 'constant', constant_values=-1)
        else:
            padded_rings = padded_rings[:max_rings]
        observation = {
            "robot_x": np.array([self.robot_position[0]], dtype=np.float32),  # changed: return 1d array
            "robot_y": np.array([self.robot_position[1]], dtype=np.float32),  # changed: return 1d array
            "robot_orientation": np.array([self.robot_orientation], dtype=np.float32),  # changed: return 1d array
            "holding_goal": self.holding_goal,  # modified observation
            "holding_rings": int(np.sum(self.ring_status==1)),  # updated to count rings held by robot
            "rings": padded_rings.flatten(),  # modified: flatten rings into 1D vector
            "goals": padded_goals,  # modified: now a flat vector
            "ring_status": self.ring_status  # added ring_status to observation
        }
        return observation, {}

    def step(self, action):
        # Ensure action is an integer (converts numpy.ndarray if needed)
        if isinstance(action, (np.ndarray,)):
            action = int(action)
        # Reset temporary success flags each step
        self.last_pickup_success = False
        self.last_ring_success = False
        self.last_drop_success = False
        # Reset drop-related flags each step
        self.last_drop_in_corner = False
        self.last_rings_dropped = 0

        done = False
        truncated = False
        time_cost = 0.0

        if 0 <= action <= 4:  # Drive-to a specific goal index
            specific_idx = action
            if specific_idx < self.mobile_goal_positions.shape[0] and self.goal_available[specific_idx]:
                old_position = self.robot_position.copy()
                target_position = self.mobile_goal_positions[specific_idx]
                direction = target_position - old_position
                self.robot_position = target_position.copy()
                self.robot_orientation = np.arctan2(direction[1], direction[0])
                distance = np.linalg.norm(target_position - old_position)
                time_cost = distance / 5.0
            else:
                time_cost = 0.5

        elif action == 5:  # Pick up goal
            time_cost = 0.5
            # Check if not holding any goal (i.e. all zeros)
            if np.sum(self.holding_goal) == 0:
                valid_idx = np.where(
                    (self.goal_available) &
                    (self.mobile_goal_positions[:,0] >= 0) & (self.mobile_goal_positions[:,0] <= 12) &
                    (self.mobile_goal_positions[:,1] >= 0) & (self.mobile_goal_positions[:,1] <= 12)
                )[0]
                if valid_idx.size > 0:
                    # Compute distances to all valid goals.
                    goals = self.mobile_goal_positions[valid_idx]
                    distances = np.linalg.norm(goals - self.robot_position, axis=1)
                    min_index = np.argmin(distances)
                    if distances[min_index] < 1.0:  # pickup threshold
                        chosen_idx = valid_idx[min_index]
                        self.holding_goal[chosen_idx] = 1  # mark as holding
                        self.goal_available[chosen_idx] = False
                        self.last_pickup_success = True

        elif action == 6:  # Intake ring: pick up a ring from ground if less than 2 held
            time_cost = 0.5
            if np.sum(self.ring_status == 1) < 2:
                # Only consider rings on the ground (status 0) not on a goal
                ground_idx = np.where(self.ring_status == 0)[0]
                if ground_idx.size > 0:
                    # Compute distances to each candidate ring
                    rings = self.ring_positions[ground_idx]
                    distances = np.linalg.norm(rings - self.robot_position, axis=1)
                    min_index = np.argmin(distances)
                    if distances[min_index] < 1.0:
                        chosen_idx = ground_idx[min_index]
                        self.ring_status[chosen_idx] = 1  # now on robot
                        # Update ring position to follow robot
                        self.ring_positions[chosen_idx] = self.robot_position.copy()
                        self.last_ring_success = True

        elif action == 7:  # Climb
            time_cost = 10.0

        elif action == 8:  # Drop goal
            time_cost = 0.5
            # Check if holding any goal
            if np.sum(self.holding_goal) > 0:
                held_indices = np.where(self.holding_goal == 1)[0]
                dropped_idx = held_indices[0]  # drop the first held goal
                self.mobile_goal_positions[dropped_idx] = self.robot_position.copy()
                self.goal_available[dropped_idx] = True
                self.holding_goal[dropped_idx] = 0  # mark as not holding
                self.last_drop_success = True
                # Check if drop location is at a corner (within 0.5 units)
                corners = [np.array([0.5, 0.5]), np.array([11.5, 0.5]), np.array([0.5, 11.5]), np.array([11.5, 11.5])]
                in_corner = any(np.linalg.norm(self.robot_position - corner) < 0.5 for corner in corners)
                self.last_drop_in_corner = in_corner
                if in_corner:
                    self.last_rings_dropped = np.sum(self.ring_status == (dropped_idx + 2))
                else:
                    self.last_rings_dropped = 0

        elif 9 <= action <= 12:  # Drive-to a specific corner
            corners = {
                9: np.array([0.5, 0.5]),
                10: np.array([11.5, 0.5]),
                11: np.array([0.5, 11.5]),
                12: np.array([11.5, 11.5])
            }
            old_position = self.robot_position.copy()
            target_position = corners[action]
            direction = target_position - old_position
            self.robot_position = target_position.copy()
            self.robot_orientation = np.arctan2(direction[1], direction[0])
            distance = np.linalg.norm(target_position - old_position)
            time_cost = distance / 5.0

        elif action == 13:  # Add ring to goal: move one ring from robot to held goal
            time_cost = 0.5
            if np.sum(self.holding_goal) > 0 and np.sum(self.ring_status==1) > 0:
                held_goal_idx = np.where(self.holding_goal == 1)[0][0]
                robot_ring_idx = np.where(self.ring_status == 1)[0][0]
                # Attach ring to the held goal: set status to (held_goal_idx+2)
                self.ring_status[robot_ring_idx] = held_goal_idx + 2
                # Update ring position to match the goal's position
                self.ring_positions[robot_ring_idx] = self.mobile_goal_positions[held_goal_idx].copy()

        elif action == 14:  # Drive to nearest ring action
            candidate = np.where(self.ring_status == 0)[0]
            if candidate.size > 0:
                rings = self.ring_positions[candidate]
                distances = np.linalg.norm(rings - self.robot_position, axis=1)
                min_index = np.argmin(distances)
                if distances[min_index] < 1.0:
                    # Drive to the nearest ring position
                    target_position = self.ring_positions[candidate[min_index]]
                    old_position = self.robot_position.copy()
                    self.robot_position = target_position.copy()
                    self.robot_orientation = np.arctan2(target_position[1] - old_position[1],
                                                        target_position[0] - old_position[0])
                    time_cost = distances[min_index] / 5.0
                else:
                    time_cost = 0.5
            else:
                time_cost = 0.5

        self.time_remaining -= time_cost
        if self.time_remaining <= 0:
            done = True

        # Pad mobile_goal_positions flattened to length 10
        mobile_goals = self.mobile_goal_positions.flatten()
        if mobile_goals.size < 10:
            mobile_goals = np.pad(mobile_goals, (0, 10 - mobile_goals.size), 'constant', constant_values=(-1,))

        # Pad or truncate ring observations to length 48
        rings_flat = self.ring_positions.flatten()
        if rings_flat.shape[0] < 48:
            rings_flat = np.pad(rings_flat, (0, 48 - rings_flat.shape[0]), 'constant', constant_values=(-1,))
        else:
            rings_flat = rings_flat[:48]

        # Ensure observation has the correct shape (63,)
        max_goals = 5
        max_rings = 24
        padded_goals = self.mobile_goal_positions.copy()
        if padded_goals.shape[0] < max_goals:
            pad_count = max_goals - padded_goals.shape[0]
            padded_goals = np.pad(padded_goals, ((0, pad_count),(0,0)), 'constant', constant_values=-1)
        else:
            padded_goals = padded_goals[:max_goals]
        padded_goals = padded_goals.flatten()  # modified: flatten goals
        padded_rings = self.ring_positions.copy()
        if padded_rings.shape[0] < max_rings:
            pad_count = max_rings - padded_rings.shape[0]
            padded_rings = np.pad(padded_rings, ((0, pad_count),(0,0)), 'constant', constant_values=-1)
        else:
            padded_rings = padded_rings[:max_rings]
        observation = {
            "robot_x": np.array([self.robot_position[0]], dtype=np.float32),  # changed: return 1d array
            "robot_y": np.array([self.robot_position[1]], dtype=np.float32),  # changed: return 1d array
            "robot_orientation": np.array([self.robot_orientation], dtype=np.float32),  # changed: return 1d array
            "holding_goal": self.holding_goal,  # updated observation
            "holding_rings": int(np.sum(self.ring_status==1)),  # updated to count rings held by robot
            "rings": padded_rings.flatten(),  # modified: flatten rings into 1D vector
            "goals": padded_goals,  # modified: now a flat vector
            "ring_status": self.ring_status  # added ring_status to observation
        }
        # Now compute reward after state update
        reward = self.reward_function(action)
        return observation, reward, done, truncated, {}

    def reward_function(self, action):
        # For drive-to specific goal actions, reward only if no goal is held
        if 0 <= action <= 4:
            return 1 if np.sum(self.holding_goal) == 0 else -10
        # For drive-to corner actions, reward only if you have a held goal with rings
        if 9 <= action <= 12:
            held_goals = np.where(self.holding_goal == 1)[0]
            valid = False
            for i in held_goals:
                if np.sum(self.ring_status == (i + 2)) > 0:
                    valid = True
                    break
            return 1 if valid else -10
        if action == 5:
            return 5 if self.last_pickup_success else -1
        elif action == 6:
            # Only reward ring intake if a goal is held; otherwise, penalize
            if np.sum(self.holding_goal) > 0:
                return 1 if self.last_ring_success else -10
            else:
                return -10
        elif action == 7:
            return 10 if self.time_remaining <= 30 else -10
        elif action == 8:
            if self.last_drop_success:
                if self.last_drop_in_corner:
                    return 5 * self.last_rings_dropped
                else:
                    return -5
            else:
                return -5
        elif action == 14:
            # Reward only if holding a goal and have less than 2 rings held on robot
            return 1 if (np.sum(self.ring_status == 1) < 2 and np.sum(self.holding_goal) > 0) else -10
        return 0

    def render(self, mode='human', save_path=None, step_num=0, action=None):
        # Convert action from numpy array to int if necessary
        if isinstance(action, (np.ndarray,)):
            action = int(action)
        # Determine action description for printing
        if 0 <= action <= 4:
            action_str = f"drive-to goal {action}"
        elif 9 <= action <= 12:
            action_str = f"drive-to corner {action}"
        else:
            action_names = {5: "pick up goal", 6: "intake rings", 7: "climb", 8: "drop goal", 13: "add ring to goal"}
            action_str = action_names.get(action, f"action {action}")
        # Convert robot position to string first before formatting
        robot_pos_str = f"({self.robot_position[0]:.2f}, {self.robot_position[1]:.2f})"
        print(f"Step: {step_num:<3} | {action_str:<25} | Reward: {self.reward_function(action):<5} | Time remaining: {self.time_remaining:<7.2f} | Robot pos: {robot_pos_str}")
        # Create a plot
        fig, ax = plt.subplots()
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 12)
        ax.set_aspect('equal')  # Ensure the field is rendered as a square

        # Draw the robot as a slightly transparent square
        robot = patches.Rectangle(self.robot_position - 0.5, 1, 1, color='blue', alpha=0.7)
        ax.add_patch(robot)
        
        # Represent orientation with an arrow (with transparency)
        center = self.robot_position
        arrow_dx = np.cos(self.robot_orientation)
        arrow_dy = np.sin(self.robot_orientation)
        orientation_arrow = patches.FancyArrow(center[0], center[1], arrow_dx, arrow_dy, width=0.1, 
                                                color='yellow', length_includes_head=True, alpha=0.7)
        ax.add_patch(orientation_arrow)
        
        # Draw mobile goals as transparent hexagons if available
        for goal, available in zip(self.mobile_goal_positions, self.goal_available):
            if available:
                hexagon = patches.RegularPolygon(goal, numVertices=6, radius=0.5, orientation=np.pi/6, 
                                                  color='green', alpha=0.7)
                ax.add_patch(hexagon)
        
        # Draw rings based on ring_status:
        for i, ring in enumerate(self.ring_positions):
            status = self.ring_status[i]
            if status == 0:
                pos = ring  # on ground
            elif status == 1:
                pos = self.robot_position  # on robot
            else:
                goal_idx = status - 2
                pos = self.mobile_goal_positions[goal_idx]  # attached to a goal
            circle = patches.Circle(pos, 0.3, color='red', alpha=0.7)
            ax.add_patch(circle)
        
        # If the robot is holding any goal, overlay a hexagon on top of the robot
        if np.sum(self.holding_goal) > 0:
            held_hex = patches.RegularPolygon(self.robot_position, numVertices=6, radius=0.5, 
                                              orientation=np.pi/6, color='green', ec='black', alpha=0.7)
            ax.add_patch(held_hex)

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
model = PPO("MultiInputPolicy", env, verbose=1)

print("Training model...")
model.learn(total_timesteps=5000)
print("Training complete.")

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

env.close()