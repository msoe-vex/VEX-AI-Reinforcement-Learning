import functools
import gymnasium
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.test import parallel_api_test
from ray.rllib.env import MultiAgentEnv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
import numpy as np
from enum import Enum
import os
import imageio.v2 as imageio
from matplotlib import transforms as mtransforms

# Try import path planner, handle if not present for standalone testing
try:
    from .path_planner import Obstacle, PathPlanner
except:
    from path_planner import Obstacle, PathPlanner

class Actions(Enum):
    PICK_UP_NEAREST_BLOCK = 0
    SCORE_IN_LONG_GOAL_TOP = 1
    SCORE_IN_LONG_GOAL_BOTTOM = 2
    SCORE_IN_CENTER_GOAL = 3
    DRIVE_TO_LOADER_TL = 4 # Top Left
    DRIVE_TO_LOADER_TR = 5 # Top Right
    DRIVE_TO_LOADER_BL = 6 # Bottom Left
    DRIVE_TO_LOADER_BR = 7 # Bottom Right
    CLEAR_LOADER = 8 # Removes blocks from loader onto field/robot
    PARK_RED = 9 # Drive to Red Park Zone
    IDLE = 10

class GameMode(Enum):
    SKILLS = "skills"  # Single robot, 60 seconds
    COMPETITION = "competition"  # Two alliance robots vs two opponent robots

class Team(Enum):
    RED = "red"
    BLUE = "blue"

# Constants based on V5RC Push Back Manual
FIELD_SIZE_INCHES = 144  # Full field is 12 feet = 144 inches
BUFFER_RADIUS_INCHES = 2  # Buffer radius in inches
ROBOT_WIDTH = 18.0  # Robot width in inches
ROBOT_LENGTH = 18.0  # Robot length in inches 
NUM_LOADERS = 4
NUM_LONG_GOALS = 2
NUM_CENTER_GOALS = 1 # Treated as one complex structure for simplicity
NUM_BLOCKS_FIELD = 36
NUM_BLOCKS_LOADER = 24 # 6 per loader
TOTAL_BLOCKS = 61 # 36 field + 24 loaders + 1 preload
TIME_LIMIT = 60 # Skills match
DEFAULT_PENALTY = -0.1
FOV = np.pi / 2

POSSIBLE_AGENTS = ["red_robot_0", "blue_robot_0"]  # Can be configured based on game mode

# Offsets for status IDs
AGENT_ID_OFFSET = 1
GOAL_ID_OFFSET = AGENT_ID_OFFSET + len(POSSIBLE_AGENTS)
# Status: 0=Field, 1=Held, 2=LongGoalTop, 3=LongGoalBot, 4=CenterGoal, 5=LoaderTL, 6=LoaderTR, 7=LoaderBL, 8=LoaderBR

def env_creator(config=None):
    config = config or {}
    return Push_Back_Multi_Agent_Env(
        render_mode=None, 
        randomize=config.get("randomize", True),
        game_mode=config.get("game_mode", GameMode.SKILLS),
        agents_config=config.get("agents_config", None)
    )

class Push_Back_Multi_Agent_Env(MultiAgentEnv, ParallelEnv):
    """
    VEX V5RC Push Back environment.
    
    COORDINATE SYSTEM:
    - All positions in this environment are in INCHES (-72 to 72)
    - Field size is 144 inches x 144 inches (12 feet x 12 feet), centered at origin
    - Path planner uses normalized coordinates (0.0 to 1.0)
    - Helper methods provided for conversion between systems
    """
    metadata = {"render_modes": ["human", "rgb_array"], "name": "vex_push_back"}

    def __init__(self, render_mode=None, output_directory="", randomize=True, game_mode=GameMode.SKILLS, agents_config=None):
        super().__init__()
        self.game_mode = game_mode
        
        # Configure agents based on game mode
        if agents_config:
            self.possible_agents = agents_config
        elif game_mode == GameMode.SKILLS:
            self.possible_agents = ["red_robot_0"]  # Single red robot for skills
        else:  # COMPETITION
            self.possible_agents = ["red_robot_0", "red_robot_1", "blue_robot_0", "blue_robot_1"]
        
        self._agent_ids = self.possible_agents
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}
        self.render_mode = render_mode
        self.agents = []
        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.possible_agents}
        self.action_spaces = {agent: self.action_space(agent) for agent in self.possible_agents}
        self.output_directory = output_directory
        self.randomize = randomize
        self.invalid_actions = []

        # Push Back Field Layout (Approximate based on Manual Appendix A)
        # All positions in inches (-72 to 72, centered at origin). Center Structure (Cross shape)
        self.permanent_obstacles = [
            Obstacle(0.0, 0.0, 18.0, False), # Center Goal Structure
            Obstacle(-24.0, 48.0, 6.0, False), # Long Goal Top - Left End
            Obstacle(24.0, 48.0, 6.0, False), # Long Goal Top - Right End
            Obstacle(-24.0, -48.0, 6.0, False), # Long Goal Bottom - Left End
            Obstacle(24.0, -48.0, 6.0, False), # Long Goal Bottom - Right End
        ]
        
        # Goal Locations (in inches, -72 to 72)
        self.goals = {
            "long_top": np.array([0.0, 48.0]),
            "long_bottom": np.array([0.0, -48.0]),
            "center": np.array([0.0, 0.0])
        }
        
        self.loader_positions = [
            np.array([-72, 48]), # TL
            np.array([72, 48]), # TR
            np.array([-72, -48]), # BL
            np.array([72, -48])  # BR
        ]
        
        self.park_zones = {
            Team.RED: {"center": np.array([-60.0, 0.0]), "bounds": (-72, -54, -12, 12)},  # (x_min, x_max, y_min, y_max)
            Team.BLUE: {"center": np.array([60.0, 0.0]), "bounds": (54, 72, -12, 12)}
        }

        self.score = 0
        
        # Path planner setup (uses 0-1 scale, so convert from inches)
        try:
            self.path_planner = PathPlanner(
                robot_length=ROBOT_LENGTH / FIELD_SIZE_INCHES,
                robot_width=ROBOT_WIDTH / FIELD_SIZE_INCHES,
                buffer_radius=BUFFER_RADIUS_INCHES / FIELD_SIZE_INCHES,
                max_velocity=80.0 / FIELD_SIZE_INCHES,
                max_accel=100.0 / FIELD_SIZE_INCHES)
        except:
            self.path_planner = None

    def _get_agent_team(self, agent_name):
        """Determine team from agent name."""
        if "red" in agent_name.lower():
            return Team.RED
        elif "blue" in agent_name.lower():
            return Team.BLUE
        return Team.RED  # Default

    def inches_to_planner_scale(self, pos_inches):
        """Convert position from inches (-72 to 72) to path planner scale (0-1)."""
        if isinstance(pos_inches, np.ndarray):
            return (pos_inches + 72.0) / FIELD_SIZE_INCHES
        return (np.array(pos_inches) + 72.0) / FIELD_SIZE_INCHES
    
    def planner_scale_to_inches(self, pos_normalized):
        """Convert position from path planner scale (0-1) to inches (-72 to 72)."""
        if isinstance(pos_normalized, np.ndarray):
            return pos_normalized * FIELD_SIZE_INCHES - 72.0
        return np.array(pos_normalized) * FIELD_SIZE_INCHES - 72.0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Observation vector size calculation:
        # Pos(2) + Orient(1) + HeldBlocks(1) + Parked(1) + Time(1)
        # Blocks (x,y,status) * TOTAL_BLOCKS (61 * 3 = 183)
        # Loaders Count (4)
        # Total ~= 193
        
        obs_size = 6 + (TOTAL_BLOCKS * 3) + 4
        
        low = np.full((obs_size,), -float('inf'), dtype=np.float32)
        high = np.full((obs_size,), float('inf'), dtype=np.float32)
        
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(len(Actions))

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        if self.randomize:
            self.environment_state = self._get_random_environment_state(seed)
        else:
            self.environment_state = self._get_initial_environment_state()
            
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _get_initial_environment_state(self):
        # Initialize 36 blocks on field (Manual RSC3)
        # Pattern: Clusters near goals and autonomous lines
        blocks = []
        
        # Helper to add block
        def add_block(x, y, status=0):
            blocks.append({"position": np.array([x, y], dtype=np.float32), "status": status})

        # Add 36 Field Blocks (Approximate distribution based on Fig RSC3-1)
        # All positions in inches (-72 to 72). 4 clusters of 4 near center
        for dx in [-18.0, 18.0]:
            for dy in [-18.0, 18.0]:
                add_block(0.0+dx, 0.0+dy)
                add_block(0.0+dx+6.0, 0.0+dy)
                add_block(0.0+dx, 0.0+dy+6.0)
                add_block(0.0+dx+6.0, 0.0+dy+6.0)
        
        # 20 blocks remaining scattered
        # Lines near long goals
        for i in range(5):
            add_block(-24.0 + i*12.0, -48.0) # Near bottom goal
            add_block(-24.0 + i*12.0, 48.0) # Near top goal
            
        # Side clusters
        for i in range(5):
            add_block(-54.0, -24.0 + (i*6.0)) # Left
            add_block(54.0, -24.0 + (i*6.0)) # Right

        # Fill remaining to get to 36 field blocks
        while len(blocks) < NUM_BLOCKS_FIELD:
            add_block(-60.0, -60.0)

        # Add Loader Blocks (Statuses 5,6,7,8 represent inside loaders 0-3)
        for loader_idx in range(4):
            for _ in range(6):
                # Loader blocks position is theoretically inside loader, but we store it
                # as the loader position for logic until it's cleared
                pos = self.loader_positions[loader_idx]
                add_block(pos[0], pos[1], status=5+loader_idx)

        # One Preload held by robot (Status 1)
        add_block(-60.0, 0.0, status=1)

        # Initialize agents based on game mode and team
        agents_dict = {}
        for agent in self.possible_agents:
            team = self._get_agent_team(agent)
            if team == Team.RED:
                start_pos = np.array([-60.0, 0.0], dtype=np.float32)
            else:  # BLUE
                start_pos = np.array([60.0, 0.0], dtype=np.float32)
            
            agents_dict[agent] = {
                "position": start_pos,
                "orientation": np.array([0.0], dtype=np.float32),
                "team": team,
                "held_blocks": 1 if agent == self.possible_agents[0] else 0,  # Only first agent has preload
                "parked": False,
                "gameTime": 0,
                "active": True,
            }

        return {
            "agents": agents_dict,
            "blocks": blocks, # Total 61 blocks
            "loaders": [6, 6, 6, 6] # Count of blocks in each loader
        }

    def _get_random_environment_state(self, seed=None):
        # For RL training, randomize field block positions slightly
        state = self._get_initial_environment_state()
        rng = np.random.default_rng(seed)
        
        for i in range(NUM_BLOCKS_FIELD):
            # Jitter positions (in inches)
            state["blocks"][i]["position"] += rng.uniform(-6.0, 6.0, size=2)
            
        return state

    def _get_observation(self, agent):
        agent_state = self.environment_state["agents"][agent]
        
        # Flatten block data
        block_data = []
        for b in self.environment_state["blocks"]:
            block_data.extend([b["position"][0], b["position"][1], float(b["status"])])
            
        obs = np.concatenate([
            agent_state["position"],
            agent_state["orientation"],
            [float(agent_state["held_blocks"])],
            [1.0 if agent_state["parked"] else 0.0],
            [float(TIME_LIMIT - agent_state["gameTime"])],
            np.array(block_data, dtype=np.float32),
            np.array(self.environment_state["loaders"], dtype=np.float32)
        ])
        
        return obs.astype(np.float32)

    def step(self, actions):
        if not actions:
            self.agents = []
            return {}, {}, {"__all__": True}, {"__all__": True}, {}
            
        self.num_moves += 1
        rewards = {agent: 0 for agent in self.agents}
        
        # Determine current game time
        minGameTime = min([self.environment_state["agents"][agent]["gameTime"] for agent in self.agents])
        
        terminations = {agent: minGameTime >= TIME_LIMIT for agent in self.agents}
        terminations["__all__"] = all(terminations.values())
        truncations = {agent: False for agent in self.agents}
        truncations["__all__"] = all(truncations.values())

        for agent, action in actions.items():
            agent_state = self.environment_state["agents"][agent]
            
            if agent_state["gameTime"] > minGameTime or not agent_state["active"]:
                continue

            initial_score = self._compute_score()
            penalty = 0
            duration = 0.5 # Default action duration
            
            # --- Actions Logic ---
            
            # Move to and Pickup Block
            if action == Actions.PICK_UP_NEAREST_BLOCK.value:
                # Robot can hold multiple blocks (plowing), but let's assume holding capacity logic
                # For "High Stakes" code logic adapted: Pick up changes status to 1
                target_block_idx = -1
                min_dist = float('inf')
                
                for i, block in enumerate(self.environment_state["blocks"]):
                    if block["status"] == 0: # On field
                        dist = np.linalg.norm(agent_state["position"] - block["position"])
                        if dist < min_dist:
                            min_dist = dist
                            target_block_idx = i
                            
                if target_block_idx != -1:
                    # Move to block
                    dist_travelled = np.linalg.norm(agent_state["position"] - self.environment_state["blocks"][target_block_idx]["position"])
                    agent_state["position"] = self.environment_state["blocks"][target_block_idx]["position"].copy()
                    duration += dist_travelled / 60.0 # Speed approx (60 inches/sec)
                    
                    # Pickup
                    self.environment_state["blocks"][target_block_idx]["status"] = 1 # Held
                    agent_state["held_blocks"] += 1
                else:
                    penalty = DEFAULT_PENALTY # No blocks found

            # Score in Goals
            elif action in [Actions.SCORE_IN_LONG_GOAL_TOP.value, Actions.SCORE_IN_LONG_GOAL_BOTTOM.value, Actions.SCORE_IN_CENTER_GOAL.value]:
                if agent_state["held_blocks"] > 0:
                    target_pos = None
                    target_status = 0
                    
                    if action == Actions.SCORE_IN_LONG_GOAL_TOP.value:
                        target_pos = self.goals["long_top"]
                        target_status = 2
                    elif action == Actions.SCORE_IN_LONG_GOAL_BOTTOM.value:
                        target_pos = self.goals["long_bottom"]
                        target_status = 3
                    elif action == Actions.SCORE_IN_CENTER_GOAL.value:
                        target_pos = self.goals["center"]
                        target_status = 4
                        
                    # Move to goal
                    dist = np.linalg.norm(agent_state["position"] - target_pos)
                    agent_state["position"] = target_pos.copy()
                    duration += dist / 60.0  # Speed in inches/sec
                    
                    # Dump all held blocks
                    count = 0
                    for block in self.environment_state["blocks"]:
                        if block["status"] == 1: # Held
                            block["status"] = target_status
                            block["position"] = target_pos.copy() # Visual update
                            count += 1
                    agent_state["held_blocks"] = 0
                    duration += 0.5 * count # Time to score
                else:
                    penalty = DEFAULT_PENALTY

            # Drive to Loaders
            elif action in [Actions.DRIVE_TO_LOADER_TL.value, Actions.DRIVE_TO_LOADER_TR.value, Actions.DRIVE_TO_LOADER_BL.value, Actions.DRIVE_TO_LOADER_BR.value]:
                idx = action - Actions.DRIVE_TO_LOADER_TL.value
                target_pos = self.loader_positions[idx]
                dist = np.linalg.norm(agent_state["position"] - target_pos)
                agent_state["position"] = target_pos.copy()
                duration += dist / 60.0  # Speed in inches/sec

            # Clear Loader
            elif action == Actions.CLEAR_LOADER.value:
                # Check which loader we are close to
                closest_loader = -1
                for i, pos in enumerate(self.loader_positions):
                    if np.linalg.norm(agent_state["position"] - pos) < 18.0:  # Within 18 inches
                        closest_loader = i
                        break
                
                if closest_loader != -1 and self.environment_state["loaders"][closest_loader] > 0:
                    # Dispense a block
                    # Find a block with status = 5 + closest_loader
                    for block in self.environment_state["blocks"]:
                        if block["status"] == 5 + closest_loader:
                            block["status"] = 0 # On field now (or 1 if robot catches it, simplified to 0)
                            block["position"] = agent_state["position"] + np.random.uniform(-6.0, 6.0, 2)
                            self.environment_state["loaders"][closest_loader] -= 1
                            break
                    duration += 1.0
                else:
                    penalty = DEFAULT_PENALTY

            # Park
            elif action == Actions.PARK_RED.value:
                team = agent_state.get("team", Team.RED)
                park_zone = self.park_zones[team]["center"]
                dist = np.linalg.norm(agent_state["position"] - park_zone)
                agent_state["position"] = park_zone.copy()
                duration += dist / 60.0  # Speed in inches/sec
                agent_state["parked"] = True
                
            elif action == Actions.IDLE.value:
                duration = 0.1

            agent_state["gameTime"] += duration
            
            # Calculate Reward
            new_score = self._compute_score()
            rewards[agent] = (new_score - initial_score) + penalty
            self.score = new_score

        # Update held block positions to follow robot
        for agent in self.agents:
            agent_state = self.environment_state["agents"][agent]
            if agent_state["held_blocks"] > 0:
                for block in self.environment_state["blocks"]:
                    if block["status"] == 1:
                        block["position"] = agent_state["position"]

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        
        if terminations["__all__"]:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _compute_score(self):
        # Based on Push Back Skills Scoring <RSC2>
        score = 0
        
        # 1. Each Block Scored in a Goal: 1 Point
        # Status 2, 3 (Long), 4 (Center)
        blocks_in_long_top = 0
        blocks_in_long_bot = 0
        blocks_in_center = 0
        
        for block in self.environment_state["blocks"]:
            s = block["status"]
            if s in [2, 3, 4]:
                score += 1
                if s == 2: blocks_in_long_top += 1
                elif s == 3: blocks_in_long_bot += 1
                elif s == 4: blocks_in_center += 1
        
        # 2. Filled Control Zones
        # Long Goal: 3 Blocks = 5 Points
        if blocks_in_long_top >= 3: score += 5
        if blocks_in_long_bot >= 3: score += 5
        
        # Center Goal: 7 Blocks = 10 Points
        if blocks_in_center >= 7: score += 10
        
        # 3. Cleared Loader: 5 Points
        for count in self.environment_state["loaders"]:
            if count == 0:
                score += 5
                
        # 4. Cleared Park Zone: 5 Points (per zone)
        # Check if any blocks (Status 0) are in park zones
        for team, zone_data in self.park_zones.items():
            x_min, x_max, y_min, y_max = zone_data["bounds"]
            zone_clear = True
            for block in self.environment_state["blocks"]:
                if block["status"] == 0:
                    if x_min <= block["position"][0] <= x_max and y_min <= block["position"][1] <= y_max:
                        zone_clear = False
                        break
            if zone_clear:
                score += 5
            
        # 5. Parked Robot: 15 Points
        for agent in self.agents:
            if self.environment_state["agents"][agent]["parked"]:
                score += 15
                
        return score

    def render(self):
        if self.render_mode is None:
            return

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-72, 72)
        ax.set_ylim(-72, 72)
        ax.set_facecolor('#cccccc') # Grey mat tiles
        
        # Draw Tape Lines (White)
        ax.plot([-72, 72], [0, 0], color='white', linewidth=2) # Auto line
        
        # Draw Park Zones
        # Red Park Zone (Left)
        rect_park_red = patches.Rectangle((-72, -12), 18, 24, linewidth=1, edgecolor='red', facecolor='none', hatch='//')
        ax.add_patch(rect_park_red)
        # Blue Park Zone (Right)
        rect_park_blue = patches.Rectangle((54, -12), 18, 24, linewidth=1, edgecolor='blue', facecolor='none', hatch='//')
        ax.add_patch(rect_park_blue)
        
        # Draw Goals
        # Long Top
        rect_lg_t = patches.Rectangle((-24, 46), 48, 4, color='orange', alpha=0.3)
        ax.add_patch(rect_lg_t)
        # Long Bottom
        rect_lg_b = patches.Rectangle((-24, -50), 48, 4, color='orange', alpha=0.3)
        ax.add_patch(rect_lg_b)
        # Center Structure
        # circle_center = patches.Circle((0, 0), 18, color='green', alpha=0.3)
        # ax.add_patch(circle_center)

        w, h = 24, 4
        # Create rectangles centered at (0,0) and rotate around the center
        rect_center_1 = patches.Rectangle((-w/2, -h/2), w, h, color='orange', alpha=0.3,
                          transform=mtransforms.Affine2D().rotate_deg_around(0, 0, 45) + ax.transData)
        ax.add_patch(rect_center_1)

        rect_center_2 = patches.Rectangle((-w/2, -h/2), w, h, color='orange', alpha=0.3,
                          transform=mtransforms.Affine2D().rotate_deg_around(0, 0, -45) + ax.transData)
        ax.add_patch(rect_center_2)
        
        # Draw Loaders
        for pos in self.loader_positions:
            circle = patches.Circle((pos[0], pos[1]), 6.0, color='orange')
            ax.add_patch(circle)
        
        # Draw Permanent Obstacles (black dotted outline circles)
        for obs in self.permanent_obstacles:
            circle = patches.Circle((obs.x, obs.y), obs.radius, fill=False, edgecolor='black', linestyle=':', linewidth=2)
            ax.add_patch(circle)

        # Draw Blocks as hexagons
        for block in self.environment_state["blocks"]:
            if block["status"] < 5: # Don't draw blocks hidden in loaders
                # Status 2,3,4 are scored -> draw darker
                # Status 1 is held -> draw blue
                # Status 0 is field -> draw red (default for skills blocks mostly)
                c = 'red'
                if block["status"] == 1: c = 'blue'
                elif block["status"] > 1: c = 'purple'
                
                # Blocks are hexagons (2.4 inch radius, approximately 4.8 inch diagonal)
                hexagon = patches.RegularPolygon(
                    (block["position"][0], block["position"][1]), 
                    numVertices=6, 
                    radius=2.4, 
                    orientation=0,
                    facecolor=c, 
                    edgecolor='black',
                    linewidth=1
                )
                ax.add_patch(hexagon)

        # Draw Robots as rectangles
        for agent in self.agents:
            st = self.environment_state["agents"][agent]
            team = st.get("team", Team.RED)
            robot_color = 'red' if team == Team.RED else 'blue'
            
            # Get position and orientation
            x, y = st["position"][0], st["position"][1]
            theta = st["orientation"][0]
            
            # Create rectangle centered at robot position
            robot_rect = patches.Rectangle(
                (-ROBOT_LENGTH/2, -ROBOT_WIDTH/2), 
                ROBOT_LENGTH, 
                ROBOT_WIDTH,
                edgecolor='black',
                facecolor=robot_color,
                alpha=0.7,
                linewidth=2,
                label=agent
            )
            
            # Apply rotation and translation
            t = mtransforms.Affine2D().rotate(theta).translate(x, y) + ax.transData
            robot_rect.set_transform(t)
            ax.add_patch(robot_rect)
            
            # Draw orientation arrow inside the robot (shorter to fit within rectangle)
            arrow_length = min(ROBOT_LENGTH, ROBOT_WIDTH) * 0.6  # 60% of smaller dimension to fit inside
            dx = np.cos(theta) * arrow_length
            dy = np.sin(theta) * arrow_length
            ax.arrow(x-dx/2-1.5, y, dx, dy, width=1, color='black', head_width=5, head_length=3, zorder=10)

        ax.set_title(f"V5RC Push Back - Score: {self.score}")
        
        # Handle Saving/Displaying
        if self.render_mode == "human":
            plt.show()
        else:
            # Save logic similar to original code
            step_num = self.num_moves
            os.makedirs(os.path.join(self.output_directory, "steps"), exist_ok=True)
            plt.savefig(os.path.join(self.output_directory, "steps", f"step_{step_num}.png"))
            plt.close()

    def close(self):
        pass
    
    def clearStepsDirectory(self):
        steps_dir = os.path.join(self.output_directory, "steps")
        if (os.path.exists(steps_dir)):
            for filename in os.listdir(steps_dir):
                os.remove(os.path.join(steps_dir, filename))

    def createGIF(self):
        steps_dir = os.path.join(self.output_directory, "steps")
        if not os.path.exists(steps_dir): return
        images = []
        files = sorted(os.listdir(steps_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))
        for filename in files:
            images.append(imageio.imread(os.path.join(steps_dir, filename)))
        if images:
            imageio.mimsave(os.path.join(self.output_directory, "simulation.gif"), images, fps=10)

if __name__ == "__main__":
    env = Push_Back_Multi_Agent_Env(render_mode="all", output_directory="vexEnv")
    print("Testing the environment...")
    
    # Basic random policy loop
    observations, infos = env.reset()
    env.clearStepsDirectory()
    
    done = False
    print("Running the environment...")
    
    while not done:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        done = terminations["__all__"] or truncations["__all__"]
        env.render()
        
    env.createGIF()
    print("Simulation complete. GIF saved.")