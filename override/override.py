# VEX U implementation of the 2026-2027 VEX V5 Override game.

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from vex_core.base_game import ActionEvent, ActionStep, VexGame
from vex_core.config import CommunicationOption
from vex_core.path_planner import Obstacle, PathPlanner
from vex_core.robot import Robot, RobotSize, Team
from vex_core.utils import vex_atan2, vex_normalize_angle, vex_shortest_angular_distance

FIELD_SIZE_INCHES = 144.0
FIELD_HALF = FIELD_SIZE_INCHES / 2.0
NUM_CUPS = 56
NUM_PINS = 63
NUM_TOGGLES = 4
MATCH_TIME = 120.0
AUTONOMOUS_TIME = 15.0
DRIVER_TIME = 105.0
FOV = 90.0
DEFAULT_DURATION = 0.5
DEFAULT_PENALTY = 1.0
MAX_HELD_PINS = 1
MAX_HELD_CUPS = 1


class Actions(Enum):
    PICKUP_PIN = 0
    PICKUP_CUP = 1
    SCORE_PIN = 2
    SCORE_CUP = 3
    TOGGLE_QUADRANT = 4
    PARK_MIDFIELD = 5
    TURN_TOWARD_CENTER = 6
    TAKE_FROM_LOADER_TL = 7
    TAKE_FROM_LOADER_TR = 8
    TAKE_FROM_LOADER_BL = 9
    TAKE_FROM_LOADER_BR = 10
    IDLE = 11


class ObjectStatus:
    ON_FIELD = 0
    HELD = 1
    SCORED = 2


class ObsIndex:
    SELF_POS_X = 0
    SELF_POS_Y = 1
    SELF_ORIENT = 2
    HELD_PINS = 3
    HELD_CUPS = 4
    PARKED = 5
    TIME_REMAINING = 6
    PIN_COUNT = 7
    CUP_COUNT = 8
    PIN_POSITIONS = 9
    CUP_POSITIONS = 29
    TOGGLES = 49
    LOADERS = 53
    TOTAL = 57


class GoalType(Enum):
    SHORT_1 = "short_1"
    SHORT_2 = "short_2"
    SHORT_3 = "short_3"
    SHORT_4 = "short_4"
    TALL = "tall"
    RED_1 = "red_1"
    RED_2 = "red_2"
    BLUE_1 = "blue_1"
    BLUE_2 = "blue_2"


GOAL_POSITIONS = {
    GoalType.SHORT_1: np.array([-36.0, 36.0], dtype=np.float32),
    GoalType.SHORT_2: np.array([36.0, 36.0], dtype=np.float32),
    GoalType.SHORT_3: np.array([-36.0, -36.0], dtype=np.float32),
    GoalType.SHORT_4: np.array([36.0, -36.0], dtype=np.float32),
    GoalType.TALL: np.array([0.0, 0.0], dtype=np.float32),
    GoalType.RED_1: np.array([-60.0, 24.0], dtype=np.float32),
    GoalType.RED_2: np.array([-60.0, -24.0], dtype=np.float32),
    GoalType.BLUE_1: np.array([60.0, 24.0], dtype=np.float32),
    GoalType.BLUE_2: np.array([60.0, -24.0], dtype=np.float32),
}
TOGGLE_POSITIONS = [
    np.array([0.0, 66.0], dtype=np.float32),
    np.array([66.0, 0.0], dtype=np.float32),
    np.array([0.0, -66.0], dtype=np.float32),
    np.array([-66.0, 0.0], dtype=np.float32),
]
PERMANENT_OBSTACLES = [Obstacle(0.0, 0.0, 8.0, False)] + [
    Obstacle(float(p[0]), float(p[1]), 4.0, False) for p in TOGGLE_POSITIONS
]


def _get_game_class(game_name: str):
    normalized_name = game_name.lower()
    if normalized_name in {"vexu_comp", "override_comp"}:
        from .vexu_comp import VexUCompGame
        return VexUCompGame
    if normalized_name in {"vexu_skills", "override_skills"}:
        from .vexu_skills import VexUSkillsGame
        return VexUSkillsGame
    if normalized_name in {"override", "vexu_override", "vex_override"}:
        return VexUOverrideGame
    raise ValueError(f"Unknown Override game: {game_name}")


class OverrideGame(VexGame):
    # Override mechanics exposed through the shared VEX game interface.

    def __init__(self, robots: Optional[list] = None,
                 communication_mode: CommunicationOption = CommunicationOption.NONE,
                 deterministic: bool = True):
        # Create an Override game with the supplied or default robot roster.
        robots = robots or [
            Robot("red_robot_0", Team.RED, RobotSize.INCH_24, np.array([-48.0, 24.0], dtype=np.float32)),
            Robot("red_robot_1", Team.RED, RobotSize.INCH_15, np.array([-48.0, -24.0], dtype=np.float32)),
            Robot("blue_robot_0", Team.BLUE, RobotSize.INCH_24, np.array([48.0, 24.0], dtype=np.float32)),
            Robot("blue_robot_1", Team.BLUE, RobotSize.INCH_15, np.array([48.0, -24.0], dtype=np.float32)),
        ]
        super().__init__(robots, communication_mode=communication_mode)
        self.deterministic = bool(deterministic)
        self.path_planner = PathPlanner()
        self.get_initial_state()

    @staticmethod
    def get_game(game_name: str, communication_mode: CommunicationOption = CommunicationOption.NONE,
                 deterministic: bool = True) -> VexGame:
        # Construct a registered Override variant by name.
        return _get_game_class(game_name)(communication_mode=communication_mode, deterministic=deterministic)

    @property
    def field_size_inches(self) -> float:
        # Return the square field width in inches.
        return FIELD_SIZE_INCHES

    @property
    def total_time(self) -> float:
        # Return the default total match duration in seconds.
        return MATCH_TIME

    @property
    def num_actions(self) -> int:
        # Return the number of high-level robot actions.
        return len(Actions)

    @property
    def fallback_action(self) -> int:
        # Return the safe action used when no action is available.
        return Actions.TURN_TOWARD_CENTER.value

    def get_action_name(self, action: int) -> str:
        # Convert an action value into its enum name.
        try:
            return Actions(int(action)).name
        except (TypeError, ValueError):
            return str(action)

    def reset(self) -> None:
        # Clear the current game state before the environment reinitializes it.
        self.state = None

    def _object(self, kind: str, position: np.ndarray, team: Optional[str] = None) -> Dict:
        # Create a field object record for a Pin or Cup.
        return {"kind": kind, "position": np.asarray(position, dtype=np.float32),
                "team": team, "status": ObjectStatus.ON_FIELD, "held_by": None, "goal": None}

    def get_initial_state(self, randomize: bool = False, seed: Optional[int] = None) -> Dict:
        # Create robots, field objects, Toggles, and available Loaders.
        if seed is not None:
            np.random.seed(seed)
        agents = {}
        for robot in self.robots:
            agents[robot.name] = {
                "position": robot.start_position.copy().astype(np.float32),
                "orientation": np.array([robot.start_orientation], dtype=np.float32),
                "camera_rotation_offset": float(robot.camera_rotation_offset),
                "team": robot.team.value, "robot_size": robot.size.value,
                "held_pins": 1, "held_cups": 0, "parked": False,
                "parked_zone": None, "toggled": [0] * NUM_TOGGLES,
                "inferred_toggle_colors": [None] * NUM_TOGGLES,
                "agent_name": robot.name, "current_action": None,
            }
        objects = []
        for index in range(NUM_PINS):
            position = np.array([np.linspace(-66, 66, NUM_PINS)[index], 0], dtype=np.float32)
            if randomize:
                position = np.random.uniform(-66, 66, 2).astype(np.float32)
            objects.append(self._object("pin", position, "red" if index % 2 == 0 else "blue"))
        for index in range(NUM_CUPS):
            position = np.array([0, np.linspace(-60, 60, NUM_CUPS)[index]], dtype=np.float32)
            if randomize:
                position = np.random.uniform(-66, 66, 2).astype(np.float32)
            objects.append(self._object("cup", position))
        self.state = {"agents": agents, "objects": objects, "toggles": [None] * NUM_TOGGLES,
                  "loaders": [6] * NUM_TOGGLES, "autonomous_winner": None}
        return self.state

    def _visible(self, agent: str, kind: str) -> List[Tuple[float, int]]:
        # Return visible field objects of a type sorted by distance.
        state = self.state["agents"][agent]
        camera = vex_normalize_angle(float(state["orientation"][0]) + state["camera_rotation_offset"])
        visible = []
        for index, obj in enumerate(self.state["objects"]):
            if obj["kind"] != kind or obj["status"] != ObjectStatus.ON_FIELD:
                continue
            direction = obj["position"] - state["position"]
            distance = float(np.linalg.norm(direction))
            if distance <= 72 and abs(vex_shortest_angular_distance(camera, vex_atan2(direction[0], direction[1]))) <= FOV / 2:
                visible.append((distance, index))
        return sorted(visible)

    def get_game_observation(self, agent: str, game_time: float = 0.0) -> np.ndarray:
        # Build the agent's partial observation, including tracker fields.
        state = self.state["agents"][agent]
        pin_visible = self._visible(agent, "pin")
        cup_visible = self._visible(agent, "cup")
        values = [state["position"][0], state["position"][1],
                  vex_normalize_angle(float(state["orientation"][0]) + state["camera_rotation_offset"]),
                  state["held_pins"], state["held_cups"], float(state["parked"]), self.total_time - game_time,
                  len(pin_visible), len(cup_visible)]
        for visible in (pin_visible[:10], cup_visible[:10]):
            values.extend([self.state["objects"][i]["position"][0] for _, i in visible])
            values.extend([self.state["objects"][i]["position"][1] for _, i in visible])
            values.extend([-144.0] * (20 - 2 * len(visible)))
        values.extend(float(toggle == state["team"]) for toggle in self.state["toggles"])
        values.extend(float(count > 0) for count in self.state["loaders"])
        return np.asarray(values, dtype=np.float32)

    def get_game_observation_space(self, agent: str) -> spaces.Space:
        # Return the fixed-size continuous observation space.
        return spaces.Box(-1e10, 1e10, shape=(ObsIndex.TOTAL,), dtype=np.float32)

    def get_game_action_space(self, agent: str) -> spaces.Space:
        # Return the discrete Override action space.
        return spaces.Discrete(self.num_actions)

    def _move(self, agent: str, target: np.ndarray, event: ActionEvent) -> List[ActionStep]:
        # Create a turn, movement, and event-completion action plan.
        state = self.state["agents"][agent]
        start = state["position"].copy()
        movement = np.asarray(target, dtype=np.float32) - start
        distance = float(np.linalg.norm(movement))
        orientation = np.array([vex_atan2(movement[0], movement[1])], dtype=np.float32) if distance else state["orientation"].copy()
        duration = distance / max(1.0, float(self.get_robot_speed(agent)))
        target = np.asarray(target, dtype=np.float32)
        return [ActionStep(DEFAULT_DURATION, start, orientation), ActionStep(duration, target, orientation),
                ActionStep(DEFAULT_DURATION, target, orientation), ActionStep(DEFAULT_DURATION, target, orientation, [event])]

    def execute_action(self, agent: str, action: int) -> Tuple[List[ActionStep], float]:
        # Translate a high-level action into timed steps and a penalty.
        state = self.state["agents"][agent]
        try:
            selected = Actions(int(action))
        except (TypeError, ValueError):
            return [ActionStep(0.1, state["position"].copy(), state["orientation"].copy())], DEFAULT_PENALTY
        if selected == Actions.IDLE:
            return [ActionStep(0.1, state["position"].copy(), state["orientation"].copy())], 0.0 if state["parked"] else DEFAULT_PENALTY
        if selected == Actions.TURN_TOWARD_CENTER:
            angle = vex_atan2(-state["position"][0], -state["position"][1]) - state["camera_rotation_offset"]
            return [ActionStep(DEFAULT_DURATION, state["position"].copy(), np.array([angle], dtype=np.float32), [ActionEvent("turn", {"angle": angle})])], 0.0
        if selected in (Actions.PICKUP_PIN, Actions.PICKUP_CUP):
            kind = "pin" if selected == Actions.PICKUP_PIN else "cup"
            visible = self._visible(agent, kind)
            held_key = f"held_{kind}s"
            capacity = MAX_HELD_PINS if kind == "pin" else MAX_HELD_CUPS
            if not visible or state[held_key] >= capacity:
                return [ActionStep(0.1, state["position"].copy(), state["orientation"].copy())], DEFAULT_PENALTY
            index = visible[0][1]
            return self._move(agent, self.state["objects"][index]["position"], ActionEvent("pickup", {"index": index})), 0.0
        if selected in (Actions.SCORE_PIN, Actions.SCORE_CUP):
            kind = "pin" if selected == Actions.SCORE_PIN else "cup"
            if state[f"held_{kind}s"] <= 0:
                return [ActionStep(0.1, state["position"].copy(), state["orientation"].copy())], DEFAULT_PENALTY
            goal = GoalType.RED_1 if state["team"] == "red" else GoalType.BLUE_1
            return self._move(agent, GOAL_POSITIONS[goal], ActionEvent("score", {"kind": kind, "goal": goal.value})), 0.0
        if selected in (Actions.TAKE_FROM_LOADER_TL, Actions.TAKE_FROM_LOADER_TR,
                        Actions.TAKE_FROM_LOADER_BL, Actions.TAKE_FROM_LOADER_BR):
            loader_index = selected.value - Actions.TAKE_FROM_LOADER_TL.value
            loader_count = self.state["loaders"][loader_index]
            if loader_count <= 0 or state["held_cups"] >= MAX_HELD_CUPS:
                return [ActionStep(0.1, state["position"].copy(), state["orientation"].copy())], DEFAULT_PENALTY
            loader_position = np.array(
                [-60.0 if loader_index % 2 == 0 else 60.0,
                 48.0 if loader_index < 2 else -48.0], dtype=np.float32)
            event = ActionEvent("clear_loader", {"loader_index": loader_index})
            return self._move(agent, loader_position, event), 0.0
        if selected == Actions.TOGGLE_QUADRANT:
            index = int(np.argmin([np.linalg.norm(state["position"] - p) for p in TOGGLE_POSITIONS]))
            return self._move(agent, TOGGLE_POSITIONS[index], ActionEvent("toggle", {"index": index})), 0.0
        if selected == Actions.PARK_MIDFIELD:
            return self._move(agent, np.zeros(2, dtype=np.float32), ActionEvent("park")), 0.0
        return [ActionStep(0.1, state["position"].copy(), state["orientation"].copy())], DEFAULT_PENALTY

    def update_tracker(self, agent: str, action: int) -> None:
        # Update inferred held-object, parking, and Toggle state after an action.
        state = self.state["agents"][agent]
        try:
            selected = Actions(int(action))
        except (TypeError, ValueError):
            return
        if selected == Actions.PICKUP_PIN and state["held_pins"] < MAX_HELD_PINS:
            state["held_pins"] += 1
        elif selected == Actions.PICKUP_CUP and state["held_cups"] < MAX_HELD_CUPS:
            state["held_cups"] += 1
        elif selected == Actions.SCORE_PIN:
            state["held_pins"] = 0
        elif selected == Actions.SCORE_CUP:
            state["held_cups"] = 0
        elif selected == Actions.PARK_MIDFIELD:
            state["parked"] = True
        elif selected == Actions.TOGGLE_QUADRANT:
            toggle_index = int(np.argmin([
                np.linalg.norm(state["position"] - position)
                for position in TOGGLE_POSITIONS
            ]))
            toggle_colors = list(state.get("inferred_toggle_colors", [None] * NUM_TOGGLES))
            toggle_colors[toggle_index] = state["team"]
            state["inferred_toggle_colors"] = toggle_colors

    def update_observation_from_tracker(self, agent: str, observation: np.ndarray) -> np.ndarray:
        # Overlay inferred state onto an externally supplied observation.
        state = self.state["agents"][agent]
        observation[ObsIndex.HELD_PINS] = state["held_pins"]
        observation[ObsIndex.HELD_CUPS] = state["held_cups"]
        observation[ObsIndex.PARKED] = float(state["parked"])
        toggle_colors = state.get("inferred_toggle_colors")
        if toggle_colors is not None:
            for index, color in enumerate(toggle_colors):
                observation[ObsIndex.TOGGLES + index] = float(color == state["team"])
        return observation

    def apply_events(self, agent: str, events: List[ActionEvent]) -> None:
        # Apply completed action events to objects, robots, Toggles, and Loaders.
        state = self.state["agents"][agent]
        for event in events:
            if event.type == "pickup":
                obj = self.state["objects"][event.data["index"]]
                held_key = f"held_{obj['kind']}s"
                capacity = MAX_HELD_PINS if obj["kind"] == "pin" else MAX_HELD_CUPS
                if obj["status"] == ObjectStatus.ON_FIELD and state[held_key] < capacity:
                    obj.update(status=ObjectStatus.HELD, held_by=agent)
                    state[held_key] += 1
            elif event.type == "score":
                kind = event.data["kind"]
                for obj in self.state["objects"]:
                    if obj["status"] == ObjectStatus.HELD and obj["held_by"] == agent and obj["kind"] == kind:
                        obj.update(status=ObjectStatus.SCORED, held_by=None, goal=event.data["goal"])
                state[f"held_{kind}s"] = 0
            elif event.type == "toggle":
                toggle_index = int(event.data["index"])
                self.state["toggles"][toggle_index] = state["team"]
                toggle_colors = list(state.get("inferred_toggle_colors", [None] * NUM_TOGGLES))
                toggle_colors[toggle_index] = state["team"]
                state["inferred_toggle_colors"] = toggle_colors
            elif event.type == "clear_loader":
                loader_index = int(event.data["loader_index"])
                if self.state["loaders"][loader_index] > 0:
                    state["held_cups"] = min(
                        MAX_HELD_CUPS, state["held_cups"] + 1
                    )
                    self.state["loaders"][loader_index] = 0
            elif event.type == "park":
                state.update(parked=True, parked_zone="midfield")
            elif event.type == "turn":
                state["orientation"] = np.array([event.data["angle"]], dtype=np.float32)

    def compute_score(self) -> Dict[str, int]:
        # Calculate alliance scores from scored Pins, parking, and bonuses.
        scores = {"red": 0, "blue": 0}
        for obj in self.state["objects"]:
            if obj["status"] == ObjectStatus.SCORED and obj["kind"] == "pin" and obj["team"] in scores:
                scores[obj["team"]] += 5
        for agent in self.state["agents"].values():
            if agent.get("parked_zone") == "midfield":
                scores[agent["team"]] += 8
        if self.state.get("autonomous_winner") in scores:
            scores[self.state["autonomous_winner"]] += 12
        return scores

    def get_team_for_agent(self, agent: str) -> str:
        # Return the alliance color assigned to an agent.
        return str(self.state["agents"].get(agent, {}).get("team", "red"))

    def is_agent_terminated(self, agent: str, game_time: float = 0.0) -> bool:
        # Return whether the agent's match clock has expired.
        return game_time >= self.total_time

    def is_valid_action(self, agent: str, action: int, observation: np.ndarray) -> bool:
        # Check whether an action is currently compatible with the observation.
        try:
            selected = Actions(int(action))
        except (TypeError, ValueError):
            return False
        if selected == Actions.PICKUP_PIN and observation[ObsIndex.HELD_PINS] >= MAX_HELD_PINS:
            return False
        if selected == Actions.PICKUP_CUP and observation[ObsIndex.HELD_CUPS] >= MAX_HELD_CUPS:
            return False
        if selected == Actions.SCORE_PIN and observation[ObsIndex.HELD_PINS] <= 0:
            return False
        if selected == Actions.SCORE_CUP and observation[ObsIndex.HELD_CUPS] <= 0:
            return False
        if selected == Actions.IDLE and observation[ObsIndex.PARKED] < 1:
            return False
        if selected in (Actions.TAKE_FROM_LOADER_TL, Actions.TAKE_FROM_LOADER_TR,
                        Actions.TAKE_FROM_LOADER_BL, Actions.TAKE_FROM_LOADER_BR):
            loader_index = selected.value - Actions.TAKE_FROM_LOADER_TL.value
            if observation[ObsIndex.LOADERS + loader_index] < 1:
                return False
        return not (selected == Actions.PARK_MIDFIELD and observation[ObsIndex.PARKED] >= 1)

    def get_permanent_obstacles(self) -> List[Obstacle]:
        # Return field structures used by the path planner.
        return PERMANENT_OBSTACLES

    def split_action(self, action: int, observation: np.ndarray, robot: Robot) -> List[str]:
        # Convert a high-level action into controller command strings.
        if action == Actions.IDLE.value:
            return ["WAIT;0.5"]
        if action == Actions.TURN_TOWARD_CENTER.value:
            return ["TURN_TO_POINT;(0.0,0.0);40"]
        if Actions.TAKE_FROM_LOADER_TL.value <= action <= Actions.TAKE_FROM_LOADER_BR.value:
            loader_index = action - Actions.TAKE_FROM_LOADER_TL.value
            loader_x = -60.0 if loader_index % 2 == 0 else 60.0
            loader_y = 48.0 if loader_index < 2 else -48.0
            return [f"FOLLOW;({loader_x:.1f}, {loader_y:.1f});50", "CLEAR_LOADER"]
        return ["WAIT;0.5"]

    def action_to_name(self, action: int) -> str:
        # Return the display name for an action value.
        return self.get_action_name(action)

    def render_game_elements(self, ax: Any) -> None:
        # Draw Goals, Toggles, and visible field objects on a Matplotlib axis.
        import matplotlib.patches as patches
        for position in GOAL_POSITIONS.values():
            ax.add_patch(patches.Circle(position, 5.0, fill=False, color="black"))
        for index, position in enumerate(TOGGLE_POSITIONS):
            ax.add_patch(patches.Circle(position, 3.0, color=self.state["toggles"][index] or "gray"))
        for obj in self.state["objects"]:
            if obj["status"] == ObjectStatus.ON_FIELD:
                ax.add_patch(patches.Circle(obj["position"], 2.4, color=obj["team"] or "gold"))

    def render_info_panel(self, ax_info: Any, agents: List[str] = None, actions: Optional[Dict] = None,
                          rewards: Optional[Dict] = None, num_steps: int = 0,
                          agent_times: Optional[Dict[str, float]] = None,
                          action_time_remaining: Optional[Dict[str, float]] = None) -> None:
        # Draw agent holdings and current alliance scores in the info panel.
        ax_info.axis("off")
        ax_info.text(0.05, 0.95, "Override", fontweight="bold", va="top")
        y = 0.85
        for agent in agents or self.state["agents"]:
            state = self.state["agents"][agent]
            ax_info.text(0.05, y, f"{agent}: {state['team']} P{state['held_pins']} C{state['held_cups']}", va="top")
            y -= 0.06
        ax_info.text(0.05, y, str(self.compute_score()), va="top")


class VexUOverrideGame(OverrideGame):
    # VEX U Override competition variant.
    pass


Override = VexUOverrideGame
