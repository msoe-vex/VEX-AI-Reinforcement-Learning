import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import matplotlib.patches as patches
from enum import Enum

class Actions(Enum):
    DRIVE_TO_GOAL_0 = 0
    DRIVE_TO_GOAL_1 = 1
    DRIVE_TO_GOAL_2 = 2
    DRIVE_TO_GOAL_3 = 3
    DRIVE_TO_GOAL_4 = 4
    PICK_UP_GOAL = 5
    INTAKE_RING = 6
    CLIMB = 7
    DROP_GOAL = 8
    DRIVE_TO_CORNER_0 = 9
    DRIVE_TO_CORNER_1 = 10
    DRIVE_TO_CORNER_2 = 11
    DRIVE_TO_CORNER_3 = 12
    ADD_RING_TO_GOAL = 13
    DRIVE_TO_NEAREST_RING = 14
    DRIVE_TO_WALL_STAKE_0 = 15
    DRIVE_TO_WALL_STAKE_1 = 16
    DRIVE_TO_WALL_STAKE_2 = 17
    DRIVE_TO_WALL_STAKE_3 = 18
    ADD_RING_TO_WALL_STAKE = 19  # Single action for adding ring to nearest wall stake

class VEXHighStakesEnv(gym.Env):
    def __init__(self):
        super(VEXHighStakesEnv, self).__init__()
        max_goals = 5
        max_rings = 24  # updated from 19 to 24 rings
        max_wall_stakes = 4  # 2 short and 2 tall wall stakes
        self.wall_stakes_positions = np.array([
            [0.0, 6.0],  # left middle (short)
            [12.0, 6.0],  # right middle (short)
            [6.0, 0.0],  # bottom middle (tall)
            [6.0, 12.0]  # top middle (tall)
        ])
        # Updated observation_space to include time_remaining and holding_goal as a single value
        self.observation_space = spaces.Dict({
            "robot_x": spaces.Box(low=0, high=12, shape=(1,), dtype=np.float32),  # Robot's x position (1,)
            "robot_y": spaces.Box(low=0, high=12, shape=(1,), dtype=np.float32),  # Robot's y position (1,)
            "robot_orientation": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),  # Robot's orientation (1,)
            "holding_goal": spaces.Discrete(2),  # Whether the robot is holding a goal (0: not holding, 1: holding)
            "holding_rings": spaces.Discrete(3),  # Number of rings the robot is holding (0, 1, or 2)
            "rings": spaces.Box(low=-1, high=12, shape=(max_rings*2,), dtype=np.float32),  # Positions of all rings (max_rings*2,)
            "goals": spaces.Box(low=-1, high=12, shape=(max_goals*2,), dtype=np.float32),  # Positions of all goals (max_goals*2,)
            "wall_stakes": spaces.Box(low=0, high=6, shape=(max_wall_stakes,), dtype=np.int32),  # Number of rings on each wall stake (max_wall_stakes,)
            "holding_goal_full": spaces.Discrete(2),  # Whether the held goal is full (0: not holding or goal not full, 1: holding goal and full)
            "time_remaining": spaces.Box(low=0, high=120, shape=(1,), dtype=np.float32)  # Time remaining in the episode (1,)
        })
        self.action_space = spaces.Discrete(20)
        # Initialize state variables
        self.total_score = 0  # Initialize total score
        self.reset()

        self.ignore_randomness = False

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.robot_position = np.array([0.5, 6.0], dtype=np.float32)
        self.last_robot_position = self.robot_position.copy()
        self.robot_orientation = 0.0  
        self.mobile_goal_positions = np.array([[4.0, 2.0], [4.0, 10.0], [8.0, 4.0], [8.0, 8.0], [10.0, 6.0]], dtype=np.float32)
        self.goal_available = np.ones(len(self.mobile_goal_positions), dtype=bool)
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
        # Replace has_goal and carried_goal_index with holding_goal (0: not holding, 1: holding)
        self.holding_goal = 0
        self.holding_goal_index = -1  # Initialize holding_goal_index to -1 (not holding any goal)
        self.holding_goal_full = 0  # Initialize holding_goal_full
        self.time_remaining = 120  
        # Remove old ring counters and use a single counter for rings held.
        self.holding_rings = 0  
        self.total_score = 0  # Reset total score
        
        # Initialize wall stakes (2 short and 2 tall)
        self.wall_stakes = np.zeros(4, dtype=np.int32)  # 0: empty, 1-6: number of rings

        # Path tracking
        # self.robot_path = [np.array(self.robot_position)]
        self.climbed = False

        # Pre-allocate padded arrays
        self.padded_goals = np.full((10,), -1, dtype=np.float32)
        self.padded_rings = np.full((48,), -1, dtype=np.float32)

        # Flatten and pad mobile_goal_positions
        mobile_goals_flat = self.mobile_goal_positions.flatten()
        self.padded_goals[:mobile_goals_flat.size] = mobile_goals_flat

        # Flatten and pad ring_positions
        rings_flat = self.ring_positions.flatten()
        self.padded_rings[:rings_flat.size] = rings_flat

        observation = {
            "robot_x": np.array([self.robot_position[0]], dtype=np.float32),  # changed: return 1d array
            "robot_y": np.array([self.robot_position[1]], dtype=np.float32),  # changed: return 1d array
            "robot_orientation": np.array([self.robot_orientation], dtype=np.float32),  # changed: return 1d array
            "holding_goal": self.holding_goal,  # modified observation to be an integer
            "holding_rings": self.holding_rings,  # updated to count rings held by robot
            "rings": self.padded_rings,  # modified: use pre-allocated array
            "goals": self.padded_goals,  # modified: use pre-allocated array
            "wall_stakes": self.wall_stakes,  # added wall_stakes to observation
            "holding_goal_full": self.holding_goal_full,  # added holding_goal_full to observation
            "time_remaining": np.array([self.time_remaining], dtype=np.float32)  # added time_remaining to observation
        }
        return observation, {}

    def step(self, action):
        # Ensure action is an integer (converts numpy.ndarray if needed)
        if isinstance(action, (np.ndarray,)):
            action = int(action)

        done = False
        truncated = False
        time_cost = 0.1  # Default time cost for actions

        # Compute the initial score before taking the action
        initial_score = self.compute_field_score()
        initial_time_remaining = self.time_remaining

        self.last_robot_position = self.robot_position.copy()

        if Actions.DRIVE_TO_GOAL_0.value <= action <= Actions.DRIVE_TO_GOAL_4.value:  # Drive-to a specific goal index
            specific_idx = action
            if specific_idx < self.mobile_goal_positions.shape[0] and self.goal_available[specific_idx]:
                old_position = self.robot_position
                target_position = self.mobile_goal_positions[specific_idx]
                distance = np.linalg.norm(target_position - old_position)
                if distance > 0.1:  # Check if the robot is not already at the goal (with a small threshold)
                    direction = target_position - old_position
                    self.robot_position = target_position
                    self.robot_orientation = np.arctan2(direction[1], direction[0])
                    time_cost = distance + 0.1
                else:
                    time_cost = 0.1  # Minimal time cost if already at the goal
            else:
                time_cost = 0.5

        elif action == Actions.PICK_UP_GOAL.value:  # Pick up goal
            time_cost = 0.5
            if np.random.rand() > 0.05 or self.ignore_randomness:  # 95% chance to succeed
                if self.holding_goal == 0:
                    valid_idx = np.where(
                        (self.goal_available) &
                        (self.mobile_goal_positions[:, 0] >= 0) & (self.mobile_goal_positions[:, 0] <= 12) &
                        (self.mobile_goal_positions[:, 1] >= 0) & (self.mobile_goal_positions[:, 1] <= 12)
                    )[0]
                    if valid_idx.size > 0:
                        # Compute distances to all valid goals.
                        goals = self.mobile_goal_positions[valid_idx]
                        distances = np.linalg.norm(goals - self.robot_position, axis=1)
                        min_index = np.argmin(distances)
                        if distances[min_index] < 1.0:  # pickup threshold
                            chosen_idx = valid_idx[min_index]
                            self.holding_goal = 1  # mark as holding
                            self.holding_goal_index = chosen_idx  # store the index of the held goal
                            self.goal_available[chosen_idx] = False
                            # Update holding_goal_full
                            self.holding_goal_full = 1 if np.sum(self.ring_status == (chosen_idx + 2)) == 6 else 0

        elif action == Actions.INTAKE_RING.value:  # Intake ring: pick up a ring from ground if less than 2 held
            time_cost = 0.5
            if np.random.rand() > 0.05 or self.ignore_randomness:  # 95% chance to succeed
                if self.holding_rings < 2:
                    # Only consider rings on the ground (status 0) not on a goal or wall stake
                    ground_idx = np.where(self.ring_status == 0)[0]
                    if ground_idx.size > 0:
                        # Compute distances to each candidate ring
                        rings = self.ring_positions[ground_idx]
                        distances = np.linalg.norm(rings - self.robot_position, axis=1)
                        min_index = np.argmin(distances)
                        if distances[min_index] < 1.0:
                            chosen_idx = ground_idx[min_index]
                            self.ring_status[chosen_idx] = 1  # now on robot
                            self.holding_rings += 1  # increment rings held by robot
                            # Update ring position to follow robot
                            self.ring_positions[chosen_idx] = self.robot_position

        elif action == Actions.CLIMB.value:
            if np.random.rand() > 0.1 or self.ignore_randomness:
                time_cost = self.time_remaining  # End of episode
                self.climbed = True

        elif action == Actions.DROP_GOAL.value:  # Drop goal
            time_cost = 0.5
            # Check if holding any goal
            if self.holding_goal == 1:
                drop_position = self.robot_position
                # Check if there are no other goals in the same area (within 1 unit distance)
                other_goals = np.delete(self.mobile_goal_positions, self.holding_goal_index, axis=0)
                distances = np.linalg.norm(other_goals - drop_position, axis=1)
                if np.all(distances >= 1.0):  # Ensure no other goals are within 1 unit distance
                    self.mobile_goal_positions[self.holding_goal_index] = drop_position
                    self.goal_available[self.holding_goal_index] = True
                    self.holding_goal = 0  # mark as not holding
                    self.holding_goal_index = -1  # reset the index of the held goal
                    self.holding_goal_full = 0  # Update holding_goal_full

        elif Actions.DRIVE_TO_CORNER_0.value <= action <= Actions.DRIVE_TO_CORNER_3.value:  # Drive-to a specific corner
            corners = np.array([
                [0.5, 0.5],
                [11.5, 0.5],
                [0.5, 11.5],
                [11.5, 11.5]
            ])
            old_position = self.robot_position
            target_position = corners[action - Actions.DRIVE_TO_CORNER_0.value]
            direction = target_position - old_position
            self.robot_position = target_position
            self.robot_orientation = np.arctan2(direction[1], direction[0])
            distance = np.linalg.norm(target_position - old_position)
            time_cost = distance + 0.1

        elif action == Actions.ADD_RING_TO_GOAL.value:  # Add ring to goal: move one ring from robot to held goal
            time_cost = 0.5
            if np.random.rand() > 0.05 or self.ignore_randomness:  # 95% chance to succeed
                if self.holding_goal == 1 and self.holding_rings > 0:
                    held_goal_idx = self.holding_goal_index
                    if np.sum(self.ring_status == (held_goal_idx + 2)) < 6:  # Check if goal has less than 6 rings
                        robot_ring_idx = np.where(self.ring_status == 1)[0][0]
                        # Attach ring to the held goal: set status to (held_goal_idx+2)
                        self.ring_status[robot_ring_idx] = held_goal_idx + 2
                        # Update ring position to match the goal's position
                        self.ring_positions[robot_ring_idx] = self.mobile_goal_positions[held_goal_idx]
                        self.holding_rings -= 1  # decrement rings held by robot
                        # Update holding_goal_full
                        self.holding_goal_full = 1 if np.sum(self.ring_status == (held_goal_idx + 2)) == 6 else 0

        elif action == Actions.DRIVE_TO_NEAREST_RING.value:  # Drive to nearest ring action
            candidate = np.where(self.ring_status == 0)[0]
            if candidate.size > 0:
                rings = self.ring_positions[candidate]
                distances = np.linalg.norm(rings - self.robot_position, axis=1)
                min_index = np.argmin(distances)
                if distances[min_index] > 1.0:
                    # Drive to the nearest ring position
                    target_position = self.ring_positions[candidate[min_index]]
                    old_position = self.robot_position
                    self.robot_position = target_position
                    self.robot_orientation = np.arctan2(target_position[1] - old_position[1],
                                                        target_position[0] - old_position[0])
                    time_cost = distances[min_index] + 0.1
                else:
                    time_cost = 0.5
            else:
                time_cost = 0.5

        elif Actions.DRIVE_TO_WALL_STAKE_0.value <= action <= Actions.DRIVE_TO_WALL_STAKE_3.value:  # Drive to a specific wall stake
            stake_idx = action - Actions.DRIVE_TO_WALL_STAKE_0.value
            old_position = self.robot_position
            target_position = self.wall_stakes_positions[stake_idx]
            direction = target_position - old_position
            self.robot_position = target_position
            self.robot_orientation = np.arctan2(direction[1], direction[0])
            distance = np.linalg.norm(target_position - old_position)
            time_cost = distance + 0.1

        elif action == Actions.ADD_RING_TO_WALL_STAKE.value:  # Add ring to nearest wall stake
            time_cost = 0.5
            if np.random.rand() > 0.05 or self.ignore_randomness:  # 95% chance to succeed
                if self.holding_rings > 0:
                    distances = np.linalg.norm(self.wall_stakes_positions - self.robot_position, axis=1)
                    nearest_stake_idx = np.argmin(distances)
                    max_rings_on_stake = 2 if nearest_stake_idx < 2 else 6  # short stakes can hold 2 rings, tall stakes can hold 6 rings
                    if self.wall_stakes[nearest_stake_idx] < max_rings_on_stake and distances[nearest_stake_idx] < 1.0:
                        robot_ring_idx = np.where(self.ring_status == 1)[0][0]
                        self.ring_status[robot_ring_idx] = nearest_stake_idx + 7  # set status to indicate ring is on wall stake
                        self.wall_stakes[nearest_stake_idx] += 1  # add ring to the wall stake
                        self.holding_rings -= 1  # decrement rings held by robot

        self.time_remaining = max(0, self.time_remaining - time_cost)
        if self.time_remaining <= 0:
            done = True

        # Update the position of the held goal to match the robot's position
        if self.holding_goal == 1:
            self.mobile_goal_positions[self.holding_goal_index] = self.robot_position

        # Flatten and pad mobile_goal_positions
        mobile_goals_flat = self.mobile_goal_positions.flatten()
        self.padded_goals[:mobile_goals_flat.size] = mobile_goals_flat

        # Flatten and pad ring_positions
        rings_flat = self.ring_positions.flatten()
        self.padded_rings[:rings_flat.size] = rings_flat

        observation = {
            "robot_x": np.array([self.robot_position[0]], dtype=np.float32),
            "robot_y": np.array([self.robot_position[1]], dtype=np.float32),
            "robot_orientation": np.array([self.robot_orientation], dtype=np.float32),
            "holding_goal": self.holding_goal,
            "holding_rings": int(np.sum(self.ring_status == 1)),
            "rings": self.padded_rings,
            "goals": self.padded_goals,
            "wall_stakes": self.wall_stakes,
            "holding_goal_full": self.holding_goal_full,
            "time_remaining": np.array([self.time_remaining], dtype=np.float32)
        }
        # Now compute reward after state update
        reward = self.reward_function(initial_score, initial_time_remaining)
        return observation, reward, done, truncated, {}

    def compute_field_score(self):
        score = 0
        corners = [np.array([0.5, 0.5]), np.array([11.5, 0.5]), np.array([0.5, 11.5]), np.array([11.5, 11.5])]
        
        # Mobile goals scoring
        for goal_idx, goal_pos in enumerate(self.mobile_goal_positions):
            if np.any(self.ring_status == (goal_idx + 2)):
                rings_on_goal = np.sum(self.ring_status == (goal_idx + 2))
                if self.holding_goal_index != goal_idx and any(np.linalg.norm(goal_pos - corner) < 0.5 for corner in corners):
                    multiplier = 2
                else:
                    multiplier = 1
                if rings_on_goal > 0:
                    score += 3 * multiplier  # First ring on a goal
                    score += (rings_on_goal - 1) * multiplier  # Other rings on goal

        # Wall stakes scoring
        for stake_idx in range(4):
            if np.any(self.ring_status == (stake_idx + 7)):
                rings_on_stake = np.sum(self.ring_status == (stake_idx + 7))
                if rings_on_stake > 0:
                    score += 3  # First ring on a stake
                    score += (rings_on_stake - 1)  # Other rings on stake
        
        # Climb scoring
        # if self.climbed:
        #     score += 3

        self.total_score = score
        return score

    def reward_function(self, initial_score, initial_time_remaining):
        # Compute the new score after taking the action
        new_score = self.compute_field_score()

        if(initial_time_remaining == 120):
            initial_estimated_final_score = initial_score  
        else:  
            initial_estimated_final_score = (initial_score / (120.0 - initial_time_remaining)) * 120.0
        
        if(self.time_remaining == 120):
            new_estimated_final_score = new_score
        else:
            new_estimated_final_score = (new_score / (120.0 - self.time_remaining)) * 120.0

        initial_scoring_potential = initial_estimated_final_score - initial_score
        new_scoring_potential = new_estimated_final_score - new_score

        delta_scoring_potential = new_scoring_potential - initial_scoring_potential

        delta_score = new_score - initial_score

        # Reward is the change in score
        reward = round(delta_score)
        return reward

    def render(self, save_path=None, step_num=0, action=None, reward=None):
        # Convert action from numpy array to int if necessary
        if isinstance(action, (np.ndarray,)):
            action = int(action)
        # Determine action description for printing
        if Actions.DRIVE_TO_GOAL_0.value <= action <= Actions.DRIVE_TO_GOAL_4.value:
            action_str = f"drive-to goal {action}"
        elif Actions.DRIVE_TO_CORNER_0.value <= action <= Actions.DRIVE_TO_CORNER_3.value:
            action_str = f"drive-to corner {action}"
        else:
            action_names = {
                Actions.PICK_UP_GOAL: "pick up goal",
                Actions.INTAKE_RING: "intake ring",
                Actions.CLIMB: "climb",
                Actions.DROP_GOAL: "drop goal",
                Actions.ADD_RING_TO_GOAL: "add ring to goal",
                Actions.DRIVE_TO_NEAREST_RING: "drive to nearest ring",
                Actions.DRIVE_TO_WALL_STAKE_0: "drive to wall stake 0",
                Actions.DRIVE_TO_WALL_STAKE_1: "drive to wall stake 1",
                Actions.DRIVE_TO_WALL_STAKE_2: "drive to wall stake 2",
                Actions.DRIVE_TO_WALL_STAKE_3: "drive to wall stake 3",
                Actions.ADD_RING_TO_WALL_STAKE: "add ring to wall stake"
            }
            action_str = action_names.get(Actions(action), f"action {action}")
        # Convert robot position to string first before formatting
        robot_pos_str = f"({self.robot_position[0]:.2f}, {self.robot_position[1]:.2f})"
        print(f"Step: {step_num:<3} | {action_str:<25} | Reward: {reward:<5} | Total score: {self.total_score:<3} | Time remaining: {self.time_remaining:<7.2f}")

        # Create a plot with specified figure size
        fig, ax = plt.subplots(figsize=(10,8))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 12)
        ax.set_aspect('equal')  # Ensure the field is rendered as a square
        ax.set_xticks([])  # Hide x-axis value labels
        ax.set_yticks([])  # Hide y-axis value labels
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        # Draw the robot as a slightly transparent square
        robot = patches.Rectangle(self.robot_position - 0.5, 1, 1, color='blue', alpha=0.25)
        ax.add_patch(robot)
        
        # Represent orientation with an arrow (with transparency)
        center = self.robot_position
        arrow_dx = np.cos(self.robot_orientation)
        arrow_dy = np.sin(self.robot_orientation)
        orientation_arrow = patches.FancyArrow(center[0], center[1], arrow_dx, arrow_dy, width=0.1, 
                                                color='yellow', length_includes_head=True, alpha=0.25)
        ax.add_patch(orientation_arrow)
        
        path_line = patches.FancyArrowPatch(self.last_robot_position, self.robot_position, lw=1, arrowstyle='-', alpha=0.25, color='black')
        ax.add_patch(path_line)
        
        # Draw mobile goals as transparent hexagons if available
        for goal_idx, (goal, available) in enumerate(zip(self.mobile_goal_positions, self.goal_available)):
            hexagon = patches.RegularPolygon(goal, numVertices=6, radius=0.5, orientation=np.pi/6, 
                                                color='green', alpha=0.25)
            ax.add_patch(hexagon)
            # Print the number of rings on the goal
            rings_on_goal = np.sum(self.ring_status == (goal_idx + 2))
            ax.text(goal[0], goal[1] + 0.6, str(rings_on_goal), color='black', ha='center')

        # Draw wall stakes as small black circles
        wall_stakes_positions = [
            np.array([0.0, 6.0]),  # left middle (short)
            np.array([12.0, 6.0]),  # right middle (short)
            np.array([6.0, 0.0]),  # bottom middle (tall)
            np.array([6.0, 12.0])  # top middle (tall)
        ]

        # Draw rings based on ring_status:
        for i, ring in enumerate(self.ring_positions):
            status = self.ring_status[i]
            if status == 0:
                pos = ring  # on ground
            elif status == 1:
                pos = self.robot_position  # on robot
            elif status >= 7:
                stake_idx = status - 7
                pos = self.wall_stakes_positions[stake_idx]  # on wall stake
            else:
                goal_idx = status - 2
                pos = self.mobile_goal_positions[goal_idx]  # attached to a goal
            circle = patches.Circle(pos, 0.3, color='red', alpha=0.7)
            ax.add_patch(circle)
        
        for pos in wall_stakes_positions:
            stake_circle = patches.Circle(pos, 0.2, color='black')
            ax.add_patch(stake_circle)

        # Print the number of rings the robot is holding just below the robot
        ax.text(self.robot_position[0], self.robot_position[1], f"{self.holding_rings}", color='black', ha='center')

        # Print the number of rings on each wall stake just outside of the field
        for idx, pos in enumerate(wall_stakes_positions):
            if idx == 0:  # left middle
                text_pos = pos + np.array([-0.5, 0.0])
            elif idx == 1:  # right middle
                text_pos = pos + np.array([0.5, 0.0])
            elif idx == 2:  # bottom middle
                text_pos = pos + np.array([0.0, -0.5])
            elif idx == 3:  # top middle
                text_pos = pos + np.array([0.0, 0.5])
            ax.text(text_pos[0], text_pos[1], f"{self.wall_stakes[idx]}", color='black', ha='center')

        # Print the total score off to the side of the field
        ax.text(-2.5, 6, f"Total Score: {self.total_score}", color='black', ha='center')
        ax.text(6, 13.25, f'Step {step_num}', color='black', ha='center')
        ax.text(6, -1.25, f'Action: {action_str}', color='black', ha='center')  # Print action name under the field

        if save_path:
            plt.savefig(f"{save_path}/step_{step_num}.png")
            plt.close()
        else:
            plt.show()

    def close(self):
        pass