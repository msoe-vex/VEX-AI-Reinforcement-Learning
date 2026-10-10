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
MIDFIELD_SIZE_INCHES = 24.0
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
VISIBILITY_POSITION_THRESHOLD = 3.0
VISIBILITY_ORIENTATION_THRESHOLD = 15.0
OPPONENT_SCAN_INTERVAL = 0.25


class Actions(Enum):
    PICKUP_PIN = 0
    PICKUP_CUP = 1
    SCORE_GOAL_1 = 2
    SCORE_GOAL_2 = 3
    SCORE_GOAL_3 = 4
    SCORE_GOAL_4 = 5
    SCORE_GOAL_5 = 6
    SCORE_GOAL_6 = 7
    SCORE_GOAL_7 = 8
    SCORE_GOAL_8 = 9
    SCORE_GOAL_9 = 10
    TOGGLE_QUADRANT = 11
    PARK_MIDFIELD = 12
    TURN_TOWARD_CENTER = 13
    TAKE_FROM_LOADER_TL = 14
    TAKE_FROM_LOADER_TR = 15
    TAKE_FROM_LOADER_BL = 16
    TAKE_FROM_LOADER_BR = 17
    IDLE = 18
    ORIENT_NEXT_PIN = 19
    ORIENT_NEXT_CUP = 20


class ObjectStatus:
    ON_FIELD = 0
    HELD = 1
    SCORED = 2
    LOADER = 3


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
    HELD_PIN_FRONT_COLOR = 57
    HELD_PIN_BACK_COLOR = 58
    HELD_CUP_FACE_UP = 59
    TEAMMATE_POS_X = 60
    TEAMMATE_POS_Y = 61
    OPPOSING_ROBOT_1_POS_X = 62
    OPPOSING_ROBOT_1_POS_Y = 63
    OPPOSING_ROBOT_2_POS_X = 64
    OPPOSING_ROBOT_2_POS_Y = 65
    TOTAL = 66


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


VEXAI_GOAL_TYPES = (
    GoalType.SHORT_1,
    GoalType.SHORT_4,
    GoalType.TALL,
    GoalType.RED_1,
    GoalType.BLUE_2,
)


GOAL_POSITIONS = {
    GoalType.SHORT_1: np.array([-24.0, 48.0], dtype=np.float32),
    GoalType.SHORT_2: np.array([-48.0, 24.0], dtype=np.float32),
    GoalType.SHORT_3: np.array([24.0, -48.0], dtype=np.float32),
    GoalType.SHORT_4: np.array([48.0, -24.0], dtype=np.float32),
    GoalType.TALL: np.array([0.0, 0.0], dtype=np.float32),
    GoalType.RED_1: np.array([-24.0, -48.0], dtype=np.float32),
    GoalType.RED_2: np.array([-48.0, -24.0], dtype=np.float32),
    GoalType.BLUE_1: np.array([24.0, 48.0], dtype=np.float32),
    GoalType.BLUE_2: np.array([48.0, 24.0], dtype=np.float32),
}
VEXAI_GOAL_POSITIONS = {
    GoalType.SHORT_1: np.array([-24.0, 48.0], dtype=np.float32),
    GoalType.SHORT_4: np.array([24.0, -48.0], dtype=np.float32),
    GoalType.TALL: np.array([0.0, 0.0], dtype=np.float32),
    GoalType.RED_1: np.array([-48.0, -24.0], dtype=np.float32),
    GoalType.BLUE_2: np.array([48.0, 24.0], dtype=np.float32),
}
# VEX AI omits the second short Goal but retains its field location for the Pin stack.
VEXAI_SECOND_GOAL_POSITION = np.array([-48.0, 24.0], dtype=np.float32)
GOAL_RADII = {
    goal: 12.0 if goal in {GoalType.RED_1, GoalType.RED_2, GoalType.BLUE_1, GoalType.BLUE_2} else 6.0
    for goal in GoalType
}
TOGGLE_POSITIONS = [
    np.array([0.0, FIELD_HALF], dtype=np.float32),
    np.array([FIELD_HALF, 0.0], dtype=np.float32),
    np.array([0.0, -FIELD_HALF], dtype=np.float32),
    np.array([-FIELD_HALF, 0.0], dtype=np.float32),
]
LOADER_POSITIONS = [
    np.array([-FIELD_HALF, 60.0], dtype=np.float32),
    np.array([FIELD_HALF, 60.0], dtype=np.float32),
    np.array([-FIELD_HALF, -60.0], dtype=np.float32),
    np.array([FIELD_HALF, -60.0], dtype=np.float32),
]
HORIZONTAL_PIN_POSITIONS = [
    (-36.0, 24.0), (0.0, 48.0),
    (36.0, -24.0), (48.0, -48.0),
]
SPECIAL_CUP_POSITION = (-24.0, 24.0)
SPECIAL_PIN_POSITIONS = [
    (-32.0, 24.0), (-16.0, 24.0),
    (-24.0, 32.0), (-24.0, 16.0),
]
MIRRORED_CUP_POSITION = (24.0, -24.0)
MIRRORED_PIN_POSITIONS = [
    (24.0, -32.0), (24.0, -16.0),
    (32.0, -24.0), (16.0, -24.0),
]
UPPER_RIGHT_CUP_POSITION = (48.0, -48.0)
UPPER_RIGHT_PIN_POSITIONS = [
    (40.0, -48.0), (56.0, -48.0),
    (48.0, -40.0), (48.0, -56.0),
]
LOWER_LEFT_CUP_POSITION = (-48.0, 48.0)
LOWER_LEFT_PIN_POSITIONS = [
    (-56.0, 48.0), (-40.0, 48.0),
    (-48.0, 56.0), (-48.0, 40.0),
]
CENTER_SQUARE_CORNER = MIDFIELD_SIZE_INCHES / np.sqrt(2.0)
CENTER_STACK_CUP_POSITIONS = [
    (0.0, CENTER_SQUARE_CORNER),
    (CENTER_SQUARE_CORNER, 0.0),
    (0.0, -CENTER_SQUARE_CORNER),
    (-CENTER_SQUARE_CORNER, 0.0),
]
CENTER_STACK_PIN_POSITIONS = [
    (0.0, CENTER_SQUARE_CORNER),
    (CENTER_SQUARE_CORNER, 0.0),
    (0.0, -CENTER_SQUARE_CORNER),
    (-CENTER_SQUARE_CORNER, 0.0),
]
YELLOW_STACK_CUP_POSITIONS = [
    (24.0, 24.0), (-24.0, -24.0),
    (48.0, 48.0), (-48.0, -48.0),
]
YELLOW_STACK_PIN_POSITIONS = YELLOW_STACK_CUP_POSITIONS.copy()
TOGGLE_STACK_CUP_POSITIONS = [
    (-24.0, FIELD_HALF - 2.4), (-19.2, FIELD_HALF - 2.4), (-14.4, FIELD_HALF - 2.4),
    (14.4, FIELD_HALF - 2.4), (19.2, FIELD_HALF - 2.4), (24.0, FIELD_HALF - 2.4),
    (FIELD_HALF - 2.4, -24.0), (FIELD_HALF - 2.4, -19.2), (FIELD_HALF - 2.4, -14.4),
    (FIELD_HALF - 2.4, 14.4), (FIELD_HALF - 2.4, 19.2), (FIELD_HALF - 2.4, 24.0),
    (-24.0, -FIELD_HALF + 2.4), (-19.2, -FIELD_HALF + 2.4), (-14.4, -FIELD_HALF + 2.4),
    (14.4, -FIELD_HALF + 2.4), (19.2, -FIELD_HALF + 2.4), (24.0, -FIELD_HALF + 2.4),
    (-FIELD_HALF + 2.4, -24.0), (-FIELD_HALF + 2.4, -19.2), (-FIELD_HALF + 2.4, -14.4),
    (-FIELD_HALF + 2.4, 14.4), (-FIELD_HALF + 2.4, 19.2), (-FIELD_HALF + 2.4, 24.0),
]
TOGGLE_STACK_PIN_POSITIONS = [
    (-19.2, FIELD_HALF - 2.4), (19.2, FIELD_HALF - 2.4),
    (FIELD_HALF - 2.4, -19.2), (FIELD_HALF - 2.4, 19.2),
    (-19.2, -FIELD_HALF + 2.4), (19.2, -FIELD_HALF + 2.4),
    (-FIELD_HALF + 2.4, -19.2), (-FIELD_HALF + 2.4, 19.2),
]
RED_YELLOW_PIN_POSITIONS = [
       (-48.0, 24.0), (-48.0, 0.0), (-48.0, -24.0), (0.0, -36.0),
    (-24.0, 48.0), (-24.0, 0.0), (-24.0, -48.0), (-12.0, -12.0),
       (-60.0, 36.0), (-60.0, 12.0), (-60.0, -12.0), (-60.0, -36.0),
]
BLUE_YELLOW_PIN_POSITIONS = [
    (48.0, 24.0), (48.0, 0.0), (48.0, -24.0), (12.0, -12.0),
    (24.0, 48.0), (24.0, 0.0), (0.0, -24.0), (24.0, -48.0),
    (60.0, 36.0), (60.0, 12.0), (60.0, -12.0), (60.0, -36.0),
]
RED_EXTRA_PIN_POSITIONS = [(30.0, 30.0), (30.0, -30.0)]
BLUE_EXTRA_PIN_POSITIONS = [(-30.0, 30.0), (-30.0, -30.0)]
YELLOW_YELLOW_PIN_POSITIONS = [
    *TOGGLE_STACK_PIN_POSITIONS,
    (24.0, 12.0),
    (-36.0, 0.0), (-12.0, 0.0), (12.0, 0.0), (36.0, 0.0),
    (0.0, -12.0),
    (0.0, -24.0),
    *YELLOW_STACK_PIN_POSITIONS,
]
PIN_START_POSITIONS = [
    np.array(position, dtype=np.float32)
    for position in (
        SPECIAL_PIN_POSITIONS
        + MIRRORED_PIN_POSITIONS
        + UPPER_RIGHT_PIN_POSITIONS
        + LOWER_LEFT_PIN_POSITIONS
        + CENTER_STACK_PIN_POSITIONS
        + HORIZONTAL_PIN_POSITIONS
        + RED_YELLOW_PIN_POSITIONS[:-4]
        + BLUE_YELLOW_PIN_POSITIONS[:-4]
        + RED_EXTRA_PIN_POSITIONS
        + BLUE_EXTRA_PIN_POSITIONS
        + YELLOW_YELLOW_PIN_POSITIONS
    )
]
CUP_WALL_OFFSETS = (-63.0, -54.0, -18.0, 18.0, 54.0, 63.0)
CUP_START_POSITIONS = [
    np.array(SPECIAL_CUP_POSITION, dtype=np.float32),
    np.array(MIRRORED_CUP_POSITION, dtype=np.float32),
    np.array(UPPER_RIGHT_CUP_POSITION, dtype=np.float32),
    np.array(LOWER_LEFT_CUP_POSITION, dtype=np.float32),
    *[np.array(position, dtype=np.float32) for position in CENTER_STACK_CUP_POSITIONS],
    *[np.array(position, dtype=np.float32) for position in YELLOW_STACK_CUP_POSITIONS],
    *[np.array(position, dtype=np.float32) for position in TOGGLE_STACK_CUP_POSITIONS],
    *[np.array([x, FIELD_HALF - 6.0], dtype=np.float32) for x in CUP_WALL_OFFSETS],
    *[np.array([FIELD_HALF - 6.0, y], dtype=np.float32) for y in CUP_WALL_OFFSETS],
    *[np.array([x, -FIELD_HALF + 6.0], dtype=np.float32) for x in reversed(CUP_WALL_OFFSETS)],
    *[np.array([-FIELD_HALF + 6.0, y], dtype=np.float32) for y in reversed(CUP_WALL_OFFSETS[:-4])],
]
PIN_COLOR_PAIRS = (
    ("red", "yellow"), ("blue", "yellow"),
    ("blue", "yellow"), ("red", "yellow"),
    ("red", "yellow"), ("blue", "yellow"),
    ("blue", "yellow"), ("red", "yellow"),
    ("red", "yellow"), ("blue", "yellow"),
    ("blue", "yellow"), ("red", "yellow"),
    ("red", "yellow"), ("blue", "yellow"),
    ("blue", "yellow"), ("red", "yellow"),
    ("red", "blue"), ("red", "blue"),
    ("red", "blue"), ("red", "blue"),
    ("red", "yellow"), ("red", "yellow"),
    ("blue", "yellow"), ("blue", "yellow"),
    *[("red", "yellow")] * 8,
    *[("blue", "yellow")] * 8,
    ("red", "yellow"), ("red", "yellow"),
    ("blue", "yellow"), ("blue", "yellow"),
    *[("yellow", "yellow")] * 19,
)
PERMANENT_OBSTACLES = [
    Obstacle(float(position[0]), float(position[1]), GOAL_RADII[goal_type], False)
    for goal_type, position in GOAL_POSITIONS.items()
] + [
    Obstacle(float(p[0]), float(p[1]), 4.0, False) for p in TOGGLE_POSITIONS
]


def _get_game_class(game_name: str):
    # Resolve a registered Override game variant by its normalized name.
    normalized_name = game_name.lower()
    if normalized_name in {"vexai_comp", "override_vexai_comp", "vexai_override_comp"}:
        from .vexai_comp import VexAICompGame
        return VexAICompGame
    if normalized_name in {"vexai_skills", "override_vexai_skills", "vexai_override_skills"}:
        from .vexai_skills import VexAISkillsGame
        return VexAISkillsGame
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
            Robot("red_robot_0", Team.RED, RobotSize.INCH_15,
                np.array([0.0, -FIELD_HALF + 12.0], dtype=np.float32), start_orientation=0.0),
            Robot("red_robot_1", Team.RED, RobotSize.INCH_15,
                np.array([-FIELD_HALF + 7.5, 0.0], dtype=np.float32), start_orientation=90.0),
            Robot("blue_robot_0", Team.BLUE, RobotSize.INCH_15,
                np.array([0.0, FIELD_HALF - 12.0], dtype=np.float32), start_orientation=180.0),
            Robot("blue_robot_1", Team.BLUE, RobotSize.INCH_15,
                np.array([FIELD_HALF - 7.5, 0.0], dtype=np.float32), start_orientation=270.0),
        ]
        super().__init__(robots, communication_mode=communication_mode)
        self.deterministic = bool(deterministic)
        self.use_path_planner_for_simulation = True
        self.path_planner = PathPlanner()
        self._visibility_revision = 0
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
        return len(self.action_values)

    @property
    def goal_types(self) -> Tuple[GoalType, ...]:
        # Return the Goals present in this game's field.
        return tuple(GoalType)

    @property
    def goal_positions(self) -> Dict[GoalType, np.ndarray]:
        # Return the goal coordinates used by this game variant.
        return GOAL_POSITIONS

    @property
    def excluded_pin_positions(self) -> Tuple[Tuple[float, float], ...]:
        # Return field Pin positions omitted from this game's initial layout.
        return ()

    @property
    def excluded_cup_positions(self) -> Tuple[Tuple[float, float], ...]:
        # Return field Cup positions omitted from this game's initial layout.
        return ()

    @property
    def action_values(self) -> Tuple[int, ...]:
        # Expose score actions only for Goals present in this game's field.
        return (
            Actions.PICKUP_PIN.value,
            Actions.PICKUP_CUP.value,
            *(Actions.SCORE_GOAL_1.value + index for index, _ in enumerate(self.goal_types)),
            *(action.value for action in (
                Actions.TOGGLE_QUADRANT,
                Actions.PARK_MIDFIELD,
                Actions.TURN_TOWARD_CENTER,
                Actions.TAKE_FROM_LOADER_TL,
                Actions.TAKE_FROM_LOADER_TR,
                Actions.TAKE_FROM_LOADER_BL,
                Actions.TAKE_FROM_LOADER_BR,
                Actions.IDLE,
                Actions.ORIENT_NEXT_PIN,
                Actions.ORIENT_NEXT_CUP,
            )),
        )

    def _decode_action(self, action: int) -> Optional[Actions]:
        # Translate the public compact action index to the legacy action enum.
        try:
            return Actions(self.action_values[int(action)])
        except (IndexError, TypeError, ValueError):
            return None

    @property
    def fallback_action(self) -> int:
        # Return the safe action used when no action is available.
        return self.action_values.index(Actions.TURN_TOWARD_CENTER.value)

    def get_action_name(self, action: int) -> str:
        # Convert an action value into its enum name.
        selected = self._decode_action(action)
        return selected.name if selected is not None else str(action)

    def reset(self) -> None:
        # Clear the current game state before the environment reinitializes it.
        self.state = None
        self._visibility_revision = 0

    def _object(self, kind: str, position: np.ndarray, team: Optional[str] = None,
                face_up: Optional[bool] = None) -> Dict:
        # Create a field object record for a Pin or Cup.
        return {"kind": kind, "position": np.asarray(position, dtype=np.float32),
                "team": team, "face_up": face_up, "status": ObjectStatus.ON_FIELD,
            "held_by": None, "goal": None, "goal_stack_index": None}

    def get_initial_state(self, randomize: bool = False, seed: Optional[int] = None) -> Dict:
        # Create robots, field objects, Toggles, and available Loaders.
        if seed is not None:
            # Seed NumPy so randomized layouts can be reproduced.
            np.random.seed(seed)
        agents = {}
        for robot in self.robots:
            # Store the simulation state needed for each robot and its tracker.
            agents[robot.name] = {
                "position": robot.start_position.copy().astype(np.float32),
                "orientation": np.array([robot.start_orientation], dtype=np.float32),
                "camera_rotation_offset": float(robot.camera_rotation_offset),
                "team": robot.team.value, "robot_size": robot.size.value,
                "held_pins": 1, "held_cups": 0, "parked": False,
                "held_stack": [],
                "held_pin_order": ["yellow", "yellow"],
                "held_cup_face_up": False,
                "opponent_seen_positions": [None, None],
                "visibility_cache": {},
                "opponent_visibility_cache": None,
                "parked_zone": None, "toggled": [0] * NUM_TOGGLES,
                "inferred_toggle_colors": [None] * NUM_TOGGLES,
                "next_pin_color": None, "next_cup_face_up": None,
                "agent_name": robot.name, "current_action": None,
            }
        objects = []
        for index in range(NUM_PINS):
            # Build each Pin with its position, front/back colors, and orientation.
            position = PIN_START_POSITIONS[index].copy()
            source_position = tuple(float(value) for value in position)
            if randomize:
                position = np.random.uniform(-66, 66, 2).astype(np.float32)
            primary_color, secondary_color = PIN_COLOR_PAIRS[index]
            # Stacked Pins start face-up so their visible color matches the stack.
            yellow_stack_count = len(YELLOW_STACK_PIN_POSITIONS) + len(TOGGLE_STACK_PIN_POSITIONS)
            yellow_cue_positions = {
                tuple(round(float(value), 3) for value in position)
                for position in YELLOW_STACK_PIN_POSITIONS + TOGGLE_STACK_PIN_POSITIONS
            }
            stack_pin = (
                16 <= index < 20
                or index >= NUM_PINS - yellow_stack_count
                or tuple(round(float(value), 3) for value in position) in yellow_cue_positions
            )
            center_pin_face_up = index >= 18 if 16 <= index < 20 else None
            pin = self._object("pin", position, team=primary_color,
                               face_up=center_pin_face_up if center_pin_face_up is not None
                               else (True if stack_pin else (index % 2 == 0)))
            pin["front_color"] = primary_color
            pin["back_color"] = secondary_color
            pin["source_position"] = source_position
            objects.append(pin)
        for index in range(NUM_CUPS):
            # Build Cups and mark the predefined stacks as face-up.
            position = CUP_START_POSITIONS[index].copy()
            source_position = tuple(float(value) for value in position)
            if randomize:
                position = np.random.uniform(-66, 66, 2).astype(np.float32)
            stack_cup = 4 <= index < 36
            cup = self._object("cup", position, face_up=True if stack_cup else (index % 2 == 0))
            cup["source_position"] = source_position
            objects.append(cup)

        preload_candidates = {
            # Select team-colored Pins that can be preloaded onto each robot.
            team: [
                index for index, obj in enumerate(objects)
                if index >= 24
                and obj["kind"] == "pin"
                and obj["status"] == ObjectStatus.ON_FIELD
                and obj.get("front_color") == team
                and obj.get("back_color") == "yellow"
            ]
            for team in ("red", "blue")
        }
        for agent_name, agent_state in agents.items():
            # Give each robot one matching preloaded Pin.
            team = agent_state["team"]
            preload_index = preload_candidates[team].pop(0)
            objects[preload_index].update(status=ObjectStatus.HELD, held_by=agent_name)
            agent_state["held_stack"] = [preload_index]

        black_goal_types = [GoalType.TALL]
        # Keep the fixed center Goal setup separate from movable field objects.
        protected_goal_pin_positions = {
            tuple(round(float(value), 3) for value in position)
            for position in YELLOW_STACK_PIN_POSITIONS + TOGGLE_STACK_PIN_POSITIONS
        }
        yellow_pin_indices = [
            index for index, obj in enumerate(objects)
            if obj["kind"] == "pin"
            and obj.get("front_color") == "yellow"
            and obj.get("back_color") == "yellow"
            and tuple(round(float(value), 3) for value in obj["position"]) not in protected_goal_pin_positions
        ]
        for goal_type, obj_index in zip(black_goal_types, yellow_pin_indices[:len(black_goal_types)]):
            # Place the initial Pin into the center Goal stack.
            objects[obj_index]["status"] = ObjectStatus.SCORED
            objects[obj_index]["goal"] = goal_type.value
            objects[obj_index]["held_by"] = None
            objects[obj_index]["goal_stack_index"] = 0

        protected_yellow_positions = {
            # Preserve Pins that belong to visible field stacks.
            tuple(float(value) for value in position)
            for position in YELLOW_STACK_PIN_POSITIONS + TOGGLE_STACK_PIN_POSITIONS
        }
        removable_yellow_indices = [
            index for index, obj in enumerate(objects)
            if obj["kind"] == "pin"
            and obj["status"] == ObjectStatus.ON_FIELD
            and obj.get("front_color") == "yellow"
            and obj.get("back_color") == "yellow"
            and tuple(float(value) for value in obj["position"]) not in protected_yellow_positions
        ]
        for object_index in sorted(removable_yellow_indices[-6:], reverse=True):
            # Remove unused setup Pins without invalidating earlier indices.
            objects.pop(object_index)

        loader_teams = ("red", "blue", "red", "blue")
        loader_pin_candidates = {
            # Loaders receive Pins matching the alliance on their field side.
            team: [
                index for index, obj in enumerate(objects)
                if index >= 16
                and obj["kind"] == "pin"
                and obj["status"] == ObjectStatus.ON_FIELD
                and obj.get("front_color") == team
                and obj.get("back_color") == "yellow"
            ]
            for team in ("red", "blue")
        }
        visible_cup_positions = {
            # These Cups remain visible on the field instead of moving to Loaders.
            tuple(round(float(value), 3) for value in position)
            for position in (
                [SPECIAL_CUP_POSITION, MIRRORED_CUP_POSITION,
                 UPPER_RIGHT_CUP_POSITION, LOWER_LEFT_CUP_POSITION]
                + CENTER_STACK_CUP_POSITIONS
                + YELLOW_STACK_CUP_POSITIONS
                + TOGGLE_STACK_CUP_POSITIONS
            )
        }
        loader_cup_candidates = [
            index for index, obj in enumerate(objects)
            if obj["kind"] == "cup"
            and obj["status"] == ObjectStatus.ON_FIELD
            and tuple(round(float(value), 3) for value in obj["position"]) not in visible_cup_positions
        ]
        loader_reserves = [5] * len(loader_teams)
        for loader_index, team in enumerate(loader_teams):
            for stack_index in range(5):
                # Pair one Pin and one Cup in each Loader stack position.
                pin_index = loader_pin_candidates[team].pop(0)
                cup_index = loader_cup_candidates.pop(0)
                for object_index in (pin_index, cup_index):
                    objects[object_index].update(
                        status=ObjectStatus.LOADER,
                        held_by=None,
                        loader_index=loader_index,
                        loader_stack=stack_index,
                    )

        extra_loader_cups = (1, 2, 2, 1)
        for loader_index, extra_count in enumerate(extra_loader_cups):
            for extra_index in range(extra_count):
                # Add the remaining yellow Pins needed to match Loader reserves.
                extra_pin = self._object(
                    "pin", LOADER_POSITIONS[loader_index].copy(), team="yellow", face_up=True,
                )
                extra_pin["front_color"] = "yellow"
                extra_pin["back_color"] = "yellow"
                extra_pin.update(
                    status=ObjectStatus.LOADER,
                    held_by=None,
                    loader_index=loader_index,
                    loader_stack=5 + extra_index,
                )
                objects.append(extra_pin)
                loader_reserves[loader_index] += 1

        self.state = {"agents": agents, "objects": objects, "toggles": [None] * NUM_TOGGLES,
                  "loaders": [6] * NUM_TOGGLES, "loader_reserves": loader_reserves,
                  "toggle_holders": [None] * NUM_TOGGLES, "autonomous_winner": None}
        # Force the first observation of this layout to perform a fresh scan.
        self._visibility_revision += 1
        return self.state

    def _visible(self, agent: str, kind: str) -> List[Tuple[float, int]]:
        # Return visible field objects of a type sorted by distance.
        state = self.state["agents"][agent]
        # Include the robot's camera offset when calculating its viewing direction.
        camera = vex_normalize_angle(float(state["orientation"][0]) + state["camera_rotation_offset"])
        cache = state.setdefault("visibility_cache", {}).get(kind)
        if cache is not None:
            moved = np.linalg.norm(state["position"] - cache["position"]) > VISIBILITY_POSITION_THRESHOLD
            rotated = abs(vex_shortest_angular_distance(camera, cache["camera"])) > VISIBILITY_ORIENTATION_THRESHOLD
            if cache["revision"] == self._visibility_revision and not moved and not rotated:
                return cache["visible"]
        visible = []
        for index, obj in enumerate(self.state["objects"]):
            # Ignore objects that are not the requested type or are no longer on the field.
            if obj["kind"] != kind or obj["status"] != ObjectStatus.ON_FIELD:
                continue
            direction = obj["position"] - state["position"]
            distance = float(np.linalg.norm(direction))
            # Keep only objects inside both the camera range and field of view.
            if distance <= 72 and abs(vex_shortest_angular_distance(camera, vex_atan2(direction[0], direction[1]))) <= FOV / 2:
                visible.append((distance, index))
        visible = sorted(visible)
        state.setdefault("visibility_cache", {})[kind] = {
            "position": state["position"].copy(),
            "camera": camera,
            "revision": self._visibility_revision,
            "visible": visible,
        }
        return visible

    def _visible_opponent_robots(self, agent: str, game_time: float = 0.0) -> List[Tuple[float, str, np.ndarray]]:
        # Return opposing robots currently visible to this agent, sorted by distance.
        state = self.state["agents"][agent]
        camera = vex_normalize_angle(float(state["orientation"][0]) + state["camera_rotation_offset"])
        cache = state.get("opponent_visibility_cache")
        if cache is not None:
            moved = np.linalg.norm(state["position"] - cache["position"]) > VISIBILITY_POSITION_THRESHOLD
            rotated = abs(vex_shortest_angular_distance(camera, cache["camera"])) > VISIBILITY_ORIENTATION_THRESHOLD
            elapsed = float(game_time) - cache["game_time"]
            if elapsed < OPPONENT_SCAN_INTERVAL and not moved and not rotated:
                return cache["visible"]
        visible = []
        for other_agent, other_state in self.state["agents"].items():
            if other_agent == agent or other_state["team"] == state["team"]:
                continue
            direction = other_state["position"] - state["position"]
            distance = float(np.linalg.norm(direction))
            if distance <= 72 and abs(vex_shortest_angular_distance(camera, vex_atan2(direction[0], direction[1]))) <= FOV / 2:
                visible.append((distance, other_agent, other_state["position"].copy()))
        visible = sorted(visible, key=lambda item: (item[0], float(item[2][0]), float(item[2][1])))
        state["opponent_visibility_cache"] = {
            "position": state["position"].copy(),
            "camera": camera,
            "game_time": float(game_time),
            "visible": visible,
        }
        return visible

    @staticmethod
    def _is_empty_opponent_memory(memory: List[Optional[np.ndarray]]) -> bool:
        return all(value is None for value in memory)

    def _update_opponent_memory(self, agent: str, game_time: float = 0.0) -> None:
        # Store the last up to two enemy robot positions that were seen this action.
        state = self.state["agents"][agent]
        current_memory = list(state.get("opponent_seen_positions", [None, None]))
        visible = self._visible_opponent_robots(agent, game_time)
        if visible:
            current_memory = [None, None]
            for index, (_, _, position) in enumerate(visible[:2]):
                current_memory[index] = np.asarray(position, dtype=np.float32).copy()
            state["opponent_seen_positions"] = current_memory
        elif self._is_empty_opponent_memory(current_memory):
            state["opponent_seen_positions"] = [None, None]

    def clear_opponent_memory(self, agent: str) -> None:
        # Clear the enemy-robot memory when the current action completes.
        state = self.state["agents"][agent]
        state["opponent_seen_positions"] = [None, None]
        state["opponent_visibility_cache"] = None

    def get_opponent_memory_penalty(self, agent: str, previous_pos: np.ndarray, current_pos: np.ndarray) -> float:
        # If the robot crosses a previously seen opponent location, apply a small penalty.
        state = self.state["agents"][agent]
        memory = state.get("opponent_seen_positions", [None, None])
        previous = np.asarray(previous_pos, dtype=np.float32)
        current = np.asarray(current_pos, dtype=np.float32)
        if np.allclose(previous, current):
            threshold = 12.0
            for seen in memory:
                if seen is None:
                    continue
                if np.linalg.norm(previous - np.asarray(seen, dtype=np.float32)) <= threshold:
                    return 1.0
            return 0.0
        segment = current - previous
        if np.allclose(segment, 0.0):
            return 0.0
        for seen in memory:
            if seen is None:
                continue
            seen_pos = np.asarray(seen, dtype=np.float32)
            v = seen_pos - previous
            t = float(np.clip(np.dot(v, segment) / np.dot(segment, segment), 0.0, 1.0))
            closest = previous + t * segment
            if np.linalg.norm(closest - seen_pos) <= 12.0:
                return 1.0
        return 0.0

    @staticmethod
    def _color_to_index(color: Optional[str]) -> float:
        # Convert a pin color name to its observation-space index.
        color_map = {"red": 1.0, "blue": 2.0, "yellow": 3.0}
        return float(color_map.get(color, -1.0))

    def get_game_observation(self, agent: str, game_time: float = 0.0) -> np.ndarray:
        # Build the agent's partial observation, including tracker fields.
        state = self.state["agents"][agent]
        # Detect Pins and Cups independently because they occupy separate observation slots.
        pin_visible = self._visible(agent, "pin")
        cup_visible = self._visible(agent, "cup")
        # Start with robot state, inventory, and the remaining match time.
        values = [state["position"][0], state["position"][1],
                  vex_normalize_angle(float(state["orientation"][0]) + state["camera_rotation_offset"]),
                  state["held_pins"], state["held_cups"], float(state["parked"]), self.total_time - game_time,
                  len(pin_visible), len(cup_visible)]
        for visible in (pin_visible[:10], cup_visible[:10]):
            # Encode up to ten visible objects and pad unused slots with the sentinel value.
            values.extend([self.state["objects"][i]["position"][0] for _, i in visible])
            values.extend([self.state["objects"][i]["position"][1] for _, i in visible])
            values.extend([-144.0] * (20 - 2 * len(visible)))
        # Encode team-owned Toggles and available Loaders as binary tracker values.
        values.extend(float(toggle == state["team"]) for toggle in self.state["toggles"])
        values.extend(float(count > 0) for count in self.state["loaders"])

        # Encode the orientation and colors of objects currently held by the robot.
        held_pin_order = state.get("held_pin_order", [None, None])
        pin_front = held_pin_order[0] if state["held_pins"] > 0 else None
        pin_back = held_pin_order[1] if state["held_pins"] > 0 else None
        values.extend([
            self._color_to_index(pin_front),
            self._color_to_index(pin_back),
            1.0 if state["held_cups"] > 0 and bool(state.get("held_cup_face_up", False)) else 0.0,
        ])

        teammate = next(
            (
                other for name, other in self.state["agents"].items()
                if name != agent and other["team"] == state["team"]
            ),
            None,
        )
        if teammate is None:
            values.extend([0.0, 0.0])
        else:
            values.extend([float(teammate["position"][0]), float(teammate["position"][1])])

        self._update_opponent_memory(agent, game_time)
        seen_positions = state.get("opponent_seen_positions", [None, None])
        for seen_position in seen_positions[:2]:
            if seen_position is None:
                values.extend([0.0, 0.0])
            else:
                values.extend([float(seen_position[0]), float(seen_position[1])])

        return np.asarray(values, dtype=np.float32)

    def get_game_observation_space(self, agent: str) -> spaces.Space:
        # Return the fixed-size continuous observation space.
        return spaces.Box(-1e10, 1e10, shape=(ObsIndex.TOTAL,), dtype=np.float32)

    def get_game_action_space(self, agent: str) -> spaces.Space:
        # Return the discrete Override action space.
        return spaces.Discrete(self.num_actions)

    @staticmethod
    def _target_next_to_rectangle(
        robot_position: np.ndarray,
        robot_length: float,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        # Return a point one half-robot-length outside the nearest rectangle edge.
        position = np.asarray(robot_position, dtype=np.float32)
        # Clamp the robot position to the rectangle to find the nearest boundary point.
        edge = np.clip(position, lower, upper)
        approach = position - edge
        approach_distance = float(np.linalg.norm(approach))
        if approach_distance == 0.0:
            # If the robot is inside the rectangle, choose the nearest edge direction.
            distances = np.array([
                position[0] - lower[0], upper[0] - position[0],
                position[1] - lower[1], upper[1] - position[1],
            ])
            side = int(np.argmin(distances))
            directions = (
                np.array([-1.0, 0.0]), np.array([1.0, 0.0]),
                np.array([0.0, -1.0]), np.array([0.0, 1.0]),
            )
            approach = directions[side]
            edge = position.copy()
            if side < 2:
                edge[0] = lower[0] if side == 0 else upper[0]
            else:
                edge[1] = lower[1] if side == 2 else upper[1]
        else:
            # Normalize the outward vector before applying the robot clearance.
            approach /= approach_distance
        return edge + approach * (robot_length / 2.0)

    def _loader_wall_pose(self, agent: str, loader_index: int) -> Tuple[np.ndarray, float]:
        # Return the approach pose and wall-facing orientation for a Loader.
        position = LOADER_POSITIONS[loader_index]
        robot_length, _ = self.get_robot_dimensions(agent)
        # Stand just inside the wall and face outward toward the Loader.
        if position[0] < 0.0:
            return np.array([-FIELD_HALF + robot_length / 2.0, position[1]], dtype=np.float32), 270.0
        return np.array([FIELD_HALF - robot_length / 2.0, position[1]], dtype=np.float32), 90.0

    def _target_next_to_goal(self, agent: str, goal: GoalType) -> np.ndarray:
        # Return a point outside the selected Goal along the robot's approach vector.
        state = self.state["agents"][agent]
        goal_position = self.goal_positions[goal]
        approach = state["position"] - goal_position
        approach_distance = float(np.linalg.norm(approach))
        if approach_distance == 0.0:
            # Use a stable default approach if the robot is at the Goal center.
            approach = np.array([0.0, 1.0], dtype=np.float32)
        else:
            approach /= approach_distance
        robot_length, _ = self.get_robot_dimensions(agent)
        # Stop outside the Goal by its radius plus half the robot length.
        return goal_position + approach * (GOAL_RADII[goal] + robot_length / 2.0)

    def _move(self, agent: str, target: np.ndarray, event: ActionEvent,
              final_orientation: Optional[float] = None) -> List[ActionStep]:
        # Create a turn, movement, and event-completion action plan.
        state = self.state["agents"][agent]
        start = state["position"].copy()
        movement = np.asarray(target, dtype=np.float32) - start
        distance = float(np.linalg.norm(movement))
        # Face the travel direction, unless the robot is already at the target.
        orientation = np.array([vex_atan2(movement[0], movement[1])], dtype=np.float32) if distance else state["orientation"].copy()
        duration = distance / max(1.0, float(self.get_robot_speed(agent)))
        target = np.asarray(target, dtype=np.float32)
        if final_orientation is not None:
            # Add a final turn before dispatching the event at the destination.
            final_orient = np.array([final_orientation], dtype=np.float32)
            return [
                ActionStep(DEFAULT_DURATION, start, orientation),
                ActionStep(duration, target, orientation),
                ActionStep(DEFAULT_DURATION, target, final_orient),
                ActionStep(DEFAULT_DURATION, target, final_orient, [event]),
            ]
        # Finish with a stationary step that dispatches the event.
        return [ActionStep(DEFAULT_DURATION, start, orientation), ActionStep(duration, target, orientation),
                ActionStep(DEFAULT_DURATION, target, orientation), ActionStep(DEFAULT_DURATION, target, orientation, [event])]

    def execute_action(self, agent: str, action: int) -> Tuple[List[ActionStep], float]:
        # Translate a high-level action into timed steps and a penalty.
        state = self.state["agents"][agent]
        selected = self._decode_action(action)
        if selected is None:
            # Invalid actions become a short no-op and receive the default penalty.
            return [ActionStep(0.1, state["position"].copy(), state["orientation"].copy())], DEFAULT_PENALTY
        if selected == Actions.IDLE:
            # Idling is free only after the robot has parked.
            return [ActionStep(0.1, state["position"].copy(), state["orientation"].copy())], 0.0 if state["parked"] else DEFAULT_PENALTY
        if selected == Actions.TURN_TOWARD_CENTER:
            # Turn toward the field center while accounting for camera offset.
            angle = vex_atan2(-state["position"][0], -state["position"][1]) - state["camera_rotation_offset"]
            return [ActionStep(DEFAULT_DURATION, state["position"].copy(), np.array([angle], dtype=np.float32), [ActionEvent("turn", {"angle": angle})])], 0.0
        if selected == Actions.ORIENT_NEXT_PIN:
            # Remember the desired Pin color for the next pickup.
            return [ActionStep(
                DEFAULT_DURATION, state["position"].copy(), state["orientation"].copy(),
                [ActionEvent("orient_next", {"kind": "pin", "color": state["team"]})],
            )], 0.0
        if selected == Actions.ORIENT_NEXT_CUP:
            # Remember that the next Cup should be picked up face-up.
            return [ActionStep(
                DEFAULT_DURATION, state["position"].copy(), state["orientation"].copy(),
                [ActionEvent("orient_next", {"kind": "cup", "face_up": True})],
            )], 0.0
        if selected in (Actions.PICKUP_PIN, Actions.PICKUP_CUP):
            # Select the nearest visible object of the requested type.
            kind = "pin" if selected == Actions.PICKUP_PIN else "cup"
            visible = self._visible(agent, kind)
            held_key = f"held_{kind}s"
            capacity = MAX_HELD_PINS if kind == "pin" else MAX_HELD_CUPS
            if not visible or state[held_key] >= capacity:
                return [ActionStep(0.1, state["position"].copy(), state["orientation"].copy())], DEFAULT_PENALTY
            index = visible[0][1]
            # Apply the pickup event after the movement plan reaches the object.
            return self._move(agent, self.state["objects"][index]["position"], ActionEvent("pickup", {"index": index})), 0.0
        if Actions.SCORE_GOAL_1.value <= selected.value <= Actions.SCORE_GOAL_9.value:
            # Score the Pin, Cup, or pair currently held by the robot.
            has_pin = state["held_pins"] > 0
            has_cup = state["held_cups"] > 0
            if not has_pin and not has_cup:
                return [ActionStep(0.1, state["position"].copy(), state["orientation"].copy())], DEFAULT_PENALTY
            goal = self.goal_types[selected.value - Actions.SCORE_GOAL_1.value]
            paired = has_pin and has_cup
            kind = "pin" if has_pin else "cup"
            scoring_kind = kind
            # Enforce the Goal's capacity and alternating Pin/Cup sequence.
            if not self._goal_allows_scoring(
                    goal.value,
                    scoring_kind,
                    pin_count=int(has_pin),
                    cup_count=int(has_cup),
            ):
                return [ActionStep(0.1, state["position"].copy(), state["orientation"].copy())], DEFAULT_PENALTY
            return self._move(
                agent, self._target_next_to_goal(agent, goal),
                ActionEvent("score", {"kind": kind, "goal": goal.value, "paired": paired}),
            ), 0.0
        if selected in (Actions.TAKE_FROM_LOADER_TL, Actions.TAKE_FROM_LOADER_TR,
                        Actions.TAKE_FROM_LOADER_BL, Actions.TAKE_FROM_LOADER_BR):
            # Approach the selected wall Loader, face it, and clear its contents.
            loader_index = selected.value - Actions.TAKE_FROM_LOADER_TL.value
            loader_count = self.state["loaders"][loader_index]
            if loader_count <= 0 or state["held_cups"] >= MAX_HELD_CUPS:
                return [ActionStep(0.1, state["position"].copy(), state["orientation"].copy())], DEFAULT_PENALTY
            loader_target, loader_orientation = self._loader_wall_pose(agent, loader_index)
            event = ActionEvent("clear_loader", {"loader_index": loader_index})
            return self._move(agent, loader_target, event, loader_orientation), 0.0
        if selected == Actions.TOGGLE_QUADRANT:
            # Claim the nearest Toggle and approach its edge before changing it.
            index = int(np.argmin([np.linalg.norm(state["position"] - p) for p in TOGGLE_POSITIONS]))
            holder = self.state["toggle_holders"][index]
            if holder is not None and holder != agent:
                return [ActionStep(0.1, state["position"].copy(), state["orientation"].copy())], DEFAULT_PENALTY
            toggle_position = TOGGLE_POSITIONS[index]
            toggle_half_size = np.array([12.0, 2.0] if toggle_position[0] == 0.0 else [2.0, 12.0])
            robot_length, _ = self.get_robot_dimensions(agent)
            toggle_target = self._target_next_to_rectangle(
                state["position"], robot_length,
                toggle_position - toggle_half_size, toggle_position + toggle_half_size,
            )
            return self._move(agent, toggle_target, ActionEvent("toggle", {"index": index})), 0.0
        if selected == Actions.TOGGLE_QUADRANT:
            # Retain the legacy Toggle movement path for compatibility.
            index = int(np.argmin([np.linalg.norm(state["position"] - p) for p in TOGGLE_POSITIONS]))
            return self._move(agent, TOGGLE_POSITIONS[index], ActionEvent("toggle", {"index": index})), 0.0
        if selected == Actions.PARK_MIDFIELD:
            # Move next to the center Goal to complete the Midfield park.
            return self._move(
                agent, self._target_next_to_goal(agent, GoalType.TALL), ActionEvent("park")
            ), 0.0
        return [ActionStep(0.1, state["position"].copy(), state["orientation"].copy())], DEFAULT_PENALTY

    def update_tracker(self, agent: str, action: int) -> None:
        # Update inferred held-object, parking, and Toggle state after an action.
        state = self.state["agents"][agent]
        selected = self._decode_action(action)
        if selected is None:
            # Do not update tracker state for an undecodable action.
            return
        if selected == Actions.PICKUP_PIN and state["held_pins"] < MAX_HELD_PINS:
            # Reflect a successful Pin pickup in the inferred inventory.
            state["held_pins"] += 1
        elif selected == Actions.PICKUP_CUP and state["held_cups"] < MAX_HELD_CUPS:
            # Reflect a successful Cup pickup in the inferred inventory.
            state["held_cups"] += 1
        elif Actions.SCORE_GOAL_1.value <= selected.value <= Actions.SCORE_GOAL_9.value:
            # Scoring transfers all held objects out of the inferred inventory.
            state["held_pins"] = 0
            state["held_cups"] = 0
        elif selected == Actions.PARK_MIDFIELD:
            # Mark the robot as parked for observations and scoring.
            state["parked"] = True
        elif selected == Actions.TOGGLE_QUADRANT:
            # Record the alliance color inferred for the nearest Toggle.
            toggle_index = int(np.argmin([
                np.linalg.norm(state["position"] - position)
                for position in TOGGLE_POSITIONS
            ]))
            if self.state["toggle_holders"][toggle_index] not in (None, agent):
                return
            toggle_colors = list(state.get("inferred_toggle_colors", [None] * NUM_TOGGLES))
            toggle_colors[toggle_index] = state["team"]
            state["inferred_toggle_colors"] = toggle_colors
        elif selected == Actions.TOGGLE_QUADRANT:
            # Record which robot currently holds the Toggle claim.
            toggle_index = int(np.argmin([
                np.linalg.norm(state["position"] - position)
                for position in TOGGLE_POSITIONS
            ]))
            state["held_toggle_index"] = toggle_index
            self.state["toggle_holders"][toggle_index] = agent

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

        held_pin_order = state.get("held_pin_order", [None, None])
        pin_front = held_pin_order[0] if state["held_pins"] > 0 else None
        pin_back = held_pin_order[1] if state["held_pins"] > 0 else None
        observation[ObsIndex.HELD_PIN_FRONT_COLOR] = self._color_to_index(pin_front)
        observation[ObsIndex.HELD_PIN_BACK_COLOR] = self._color_to_index(pin_back)
        observation[ObsIndex.HELD_CUP_FACE_UP] = 1.0 if state["held_cups"] > 0 and bool(state.get("held_cup_face_up", False)) else 0.0
        return observation

    def _goal_allows_scoring(self, goal_value: str, kind: str,
                             pin_count: int = 1, cup_count: int = 0) -> bool:
        # Goals alternate cup/pin and have separate pin/cup capacities.
        # Count only objects already committed to this Goal.
        scored_objects = [
            obj for obj in self.state["objects"]
            if obj.get("goal") == goal_value and obj["status"] == ObjectStatus.SCORED
        ]
        max_pins = 6 if goal_value == GoalType.TALL.value else 7
        max_cups = 5 if goal_value == GoalType.TALL.value else 6
        # Reject scores that would exceed the per-object-type capacity.
        scored_pins = sum(obj["kind"] == "pin" for obj in scored_objects)
        scored_cups = sum(obj["kind"] == "cup" for obj in scored_objects)
        if scored_pins + pin_count > max_pins or scored_cups + cup_count > max_cups:
            return False
        if not scored_objects:
            # Every Goal must begin with a Pin.
            return kind == "pin"
        # Subsequent objects must alternate between Pins and Cups.
        last_kind = scored_objects[-1]["kind"]
        return last_kind != kind

    def apply_events(self, agent: str, events: List[ActionEvent]) -> None:
        # Apply completed action events to objects, robots, Toggles, and Loaders.
        state = self.state["agents"][agent]
        objects_changed = any(event.type in {"pickup", "score"} for event in events)
        for event in events:
            if event.type == "pickup":
                # Move the selected field object into the robot's inventory.
                obj = self.state["objects"][event.data["index"]]
                held_key = f"held_{obj['kind']}s"
                capacity = MAX_HELD_PINS if obj["kind"] == "pin" else MAX_HELD_CUPS
                if obj["status"] == ObjectStatus.ON_FIELD and state[held_key] < capacity:
                    if obj["kind"] == "pin":
                        # Apply a requested Pin face before storing its visible colors.
                        desired_color = state.get("next_pin_color")
                        if desired_color == obj.get("front_color"):
                            obj["face_up"] = True
                        elif desired_color == obj.get("back_color"):
                            obj["face_up"] = False
                        state["next_pin_color"] = None
                    else:
                        # Apply a requested Cup orientation before pickup.
                        desired_face_up = state.get("next_cup_face_up")
                        if desired_face_up is not None:
                            obj["face_up"] = bool(desired_face_up)
                        state["next_cup_face_up"] = None
                    obj.update(status=ObjectStatus.HELD, held_by=agent)
                    state[held_key] += 1
                    if obj["kind"] == "pin":
                        # Preserve the front/back order visible to the robot tracker.
                        ordered_colors = [obj.get("front_color"), obj.get("back_color")]
                        if not bool(obj.get("face_up", True)):
                            ordered_colors = list(reversed(ordered_colors))
                        state["held_pin_order"] = ordered_colors
                    elif obj["kind"] == "cup":
                        # Preserve the Cup face for the held-object observation.
                        state["held_cup_face_up"] = bool(obj.get("face_up", False))
                    held_indices = [
                        index for index, held_obj in enumerate(self.state["objects"])
                        if held_obj["status"] == ObjectStatus.HELD and held_obj["held_by"] == agent
                    ]
                    for partner in self.state["objects"]:
                        # Pick up a paired object occupying the same field location.
                        same_position = np.allclose(partner["position"], obj["position"])
                        partner_key = f"held_{partner['kind']}s"
                        partner_capacity = MAX_HELD_PINS if partner["kind"] == "pin" else MAX_HELD_CUPS
                        if (partner is not obj and same_position
                                and partner["kind"] != obj["kind"]
                                and partner["status"] == ObjectStatus.ON_FIELD
                                and state[partner_key] < partner_capacity):
                            if partner["kind"] == "pin":
                                desired_color = state.get("next_pin_color")
                                if desired_color == partner.get("front_color"):
                                    partner["face_up"] = True
                                elif desired_color == partner.get("back_color"):
                                    partner["face_up"] = False
                                state["next_pin_color"] = None
                            else:
                                desired_face_up = state.get("next_cup_face_up")
                                if desired_face_up is not None:
                                    partner["face_up"] = bool(desired_face_up)
                                state["next_cup_face_up"] = None
                            partner.update(status=ObjectStatus.HELD, held_by=agent)
                            state[partner_key] += 1
                            if partner["kind"] == "pin":
                                ordered_colors = [partner.get("front_color"), partner.get("back_color")]
                                if not bool(partner.get("face_up", True)):
                                    ordered_colors = list(reversed(ordered_colors))
                                state["held_pin_order"] = ordered_colors
                            elif partner["kind"] == "cup":
                                state["held_cup_face_up"] = bool(partner.get("face_up", False))
                    held_indices = [
                        index for index, held_obj in enumerate(self.state["objects"])
                        if held_obj["status"] == ObjectStatus.HELD and held_obj["held_by"] == agent
                    ]
                    state["held_stack"] = sorted(
                        held_indices,
                        key=lambda index: 0 if self.state["objects"][index]["kind"] == "cup" else 1,
                    )
            elif event.type == "score":
                # Transfer held objects into the selected Goal stack.
                kind = event.data["kind"]
                goal_value = event.data["goal"]
                scored_kinds = {kind}
                if event.data.get("paired") and state["held_pins"] > 0 and state["held_cups"] > 0:
                    # A paired score places both the Pin and Cup together.
                    scored_kinds = {"pin", "cup"}
                if not self._goal_allows_scoring(
                        goal_value,
                        kind,
                        pin_count=int("pin" in scored_kinds),
                        cup_count=int("cup" in scored_kinds),
                ):
                    state[f"held_{kind}s"] = max(0, state[f"held_{kind}s"])
                    continue
                next_stack_index = sum(
                    # Append new objects after all existing objects in this Goal.
                    1 for obj in self.state["objects"]
                    if obj["status"] == ObjectStatus.SCORED
                    and obj.get("goal") == goal_value
                )
                for obj in self.state["objects"]:
                    # Only score objects held by this robot and requested by the event.
                    if (obj["status"] == ObjectStatus.HELD and obj["held_by"] == agent
                            and obj["kind"] in scored_kinds):
                        obj.update(
                            status=ObjectStatus.SCORED,
                            held_by=None,
                            goal=goal_value,
                            goal_stack_index=next_stack_index,
                        )
                        next_stack_index += 1
                for scored_kind in scored_kinds:
                    state[f"held_{scored_kind}s"] = 0
                state["held_pin_order"] = [None, None]
                state["held_cup_face_up"] = False
                state["held_stack"] = [
                    index for index in state.get("held_stack", [])
                    if self.state["objects"][index]["status"] == ObjectStatus.HELD
                ]
            elif event.type == "orient_next":
                # Store orientation preferences until the next pickup event.
                if event.data["kind"] == "pin":
                    state["next_pin_color"] = event.data["color"]
                else:
                    state["next_cup_face_up"] = bool(event.data["face_up"])
            elif event.type == "toggle":
                # Change a Toggle only when another robot does not own it.
                toggle_index = int(event.data["index"])
                if self.state["toggle_holders"][toggle_index] not in (None, agent):
                    continue
                self.state["toggles"][toggle_index] = state["team"]
                toggle_colors = list(state.get("inferred_toggle_colors", [None] * NUM_TOGGLES))
                toggle_colors[toggle_index] = state["team"]
                state["inferred_toggle_colors"] = toggle_colors
            elif event.type == "hold_toggle":
                # Record temporary Toggle ownership for action validation.
                toggle_index = int(event.data["index"])
                self.state["toggle_holders"][toggle_index] = agent
                state["held_toggle_index"] = toggle_index
            elif event.type == "clear_loader":
                # Clear one Loader and restore any reserved refill count.
                loader_index = int(event.data["loader_index"])
                if self.state["loaders"][loader_index] > 0:
                    state["held_cups"] = min(
                        MAX_HELD_CUPS, state["held_cups"] + 1
                    )
                    self.state["loaders"][loader_index] = 0
                    reserve_count = self.state.get("loader_reserves", [0] * NUM_TOGGLES)[loader_index]
                    if reserve_count > 0:
                        self.state["loaders"][loader_index] = reserve_count
                        self.state["loader_reserves"][loader_index] = 0
            elif event.type == "park":
                state.update(parked=True, parked_zone="midfield")
            elif event.type == "turn":
                state["orientation"] = np.array([event.data["angle"]], dtype=np.float32)
        if objects_changed:
            # Invalidate cached field visibility after object status changes.
            self._visibility_revision += 1

    def _score_pin_color(self, pin: Dict, goal_value: str) -> Optional[Tuple[str, int]]:
        """Return the alliance and value of the pin half facing away from its goal."""
        if not pin.get("face_up", True):
            # A face-down Pin does not expose a scoring color.
            return None
        color = pin.get("front_color")
        if color not in {"red", "blue", "yellow"}:
            return None
        if color == "yellow":
            # Yellow halves score for the color of the Goal they occupy.
            goal_color = goal_value.split("_", 1)[0]
            if goal_color not in {"red", "blue"}:
                return None
            return goal_color, 10
        # A red or blue half scores directly for its alliance.
        return color, 5

    def _score_goal_pins(self) -> Dict[str, int]:
        """Score exposed pin halves, walking each stack outward from its goal."""
        scores = {"red": 0, "blue": 0}
        goal_values = {goal.value for goal in GoalType}
        for goal_value in goal_values:
            # Sort each Goal from its base outward so cover rules are deterministic.
            stack = sorted(
                (
                    obj for obj in self.state["objects"]
                    if obj["status"] == ObjectStatus.SCORED
                    and obj.get("goal") == goal_value
                ),
                key=lambda obj: (
                    obj.get("goal_stack_index") is None,
                    obj.get("goal_stack_index") or 0,
                ),
            )
            for index, obj in enumerate(stack):
                if obj["kind"] != "pin":
                    continue
                # The goal hides the lower half of the first pin. A black cup
                # underside hides the upper half of the pin immediately below it.
                if index == 0 and not obj.get("face_up", True):
                    # The Goal itself hides the lower half of its first Pin.
                    continue
                if index + 1 < len(stack):
                    cover = stack[index + 1]
                    if cover["kind"] == "cup" and cover.get("face_up", True):
                        # A face-up Cup hides the Pin half immediately below it.
                        continue
                scored = self._score_pin_color(obj, goal_value)
                if scored is not None:
                    alliance, value = scored
                    scores[alliance] += value
        return scores

    def compute_score(self) -> Dict[str, int]:
        # Calculate alliance scores from exposed scored pin halves, parking, and bonuses.
        scores = {"red": 0, "blue": 0}
        # Add exposed Pin-half values from every Goal.
        pin_scores = self._score_goal_pins()
        for alliance, value in pin_scores.items():
            scores[alliance] += value
        for agent in self.state["agents"].values():
            # Award the Midfield parking bonus to each parked robot's alliance.
            if agent.get("parked_zone") == "midfield":
                scores[agent["team"]] += 8
        if self.state.get("autonomous_winner") in scores:
            # Apply the autonomous bonus to the recorded winning alliance.
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
        selected = self._decode_action(action)
        if selected is None:
            return False
        if selected == Actions.PICKUP_PIN and observation[ObsIndex.HELD_PINS] >= MAX_HELD_PINS:
            # Do not pick up another Pin when the Pin capacity is full.
            return False
        if selected == Actions.PICKUP_CUP and observation[ObsIndex.HELD_CUPS] >= MAX_HELD_CUPS:
            # Do not pick up another Cup when the Cup capacity is full.
            return False
        if Actions.SCORE_GOAL_1.value <= selected.value <= Actions.SCORE_GOAL_9.value and (
                observation[ObsIndex.HELD_PINS] <= 0
                and observation[ObsIndex.HELD_CUPS] <= 0):
            # A scoring action requires at least one held object.
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
        return [
            Obstacle(
                float(self.goal_positions[goal_type][0]),
                float(self.goal_positions[goal_type][1]),
                GOAL_RADII[goal_type],
                False,
            )
            for goal_type in self.goal_types
        ] + [
            Obstacle(float(p[0]), float(p[1]), 4.0, False) for p in TOGGLE_POSITIONS
        ]

    def render_field_markings(self, ax: Any) -> None:
        # Render the square Midfield, diagonal Autonomous Lines, and Load Zones.
        import matplotlib.patches as patches

        field_half = self.field_size_inches / 2
        # Draw the outer field boundary.
        ax.set_facecolor("#d7d7d7")
        ax.add_patch(patches.Rectangle(
            (-field_half, -field_half), self.field_size_inches, self.field_size_inches,
            fill=False, edgecolor="black", linewidth=1.5,
        ))

        midfield_half = MIDFIELD_SIZE_INCHES / 2
        # Draw the rotated square that marks Midfield.
        ax.add_patch(patches.Rectangle(
            (-midfield_half, -midfield_half), MIDFIELD_SIZE_INCHES, MIDFIELD_SIZE_INCHES,
            fill=False, edgecolor="white", linewidth=2.5, angle=45,
            rotation_point="center", zorder=1,
        ))

        square_side_midpoint = midfield_half / np.sqrt(2.0)
        # Connect each field corner to the corresponding Midfield corner.
        for field_x, field_y, square_x, square_y in (
            (-field_half, field_half, -square_side_midpoint, square_side_midpoint),
            (field_half, field_half, square_side_midpoint, square_side_midpoint),
            (-field_half, -field_half, -square_side_midpoint, -square_side_midpoint),
            (field_half, -field_half, square_side_midpoint, -square_side_midpoint),
        ):
            ax.plot(
                [field_x, square_x], [field_y, square_y],
                color="white", linewidth=2.0, zorder=1,
            )

        zone_inset = 12.0
        zone_length = 24.0
        # Draw the colored alliance Load Zones at both field ends.
        for x, color in ((-field_half, "red"), (field_half, "blue")):
            inner_x = x + zone_inset if x < 0 else x - zone_inset
            for y in (field_half, -field_half):
                inner_y = y - zone_length if y > 0 else y + zone_length
                ax.plot(
                    [x, inner_x], [inner_y, inner_y],
                    color=color, linewidth=2.0, zorder=1,
                )
                ax.plot(
                    [inner_x, inner_x], [inner_y, y],
                    color=color, linewidth=2.0, zorder=1,
                )

    def camera_fov_degrees(self) -> float:
        # Return the robot camera's field of view in degrees.
        return FOV

    def camera_range_inches(self) -> float:
        # Return the robot camera's maximum detection range in inches.
        return 72.0

    def split_action(self, action: int, observation: np.ndarray, robot: Robot) -> List[str]:
        # Convert a high-level action into controller command strings.
        selected = self._decode_action(action)
        if selected is None:
            return ["WAIT;0.5"]
        if selected == Actions.IDLE:
            # Keep the physical robot still while it idles.
            return ["WAIT;0.5"]
        if selected == Actions.TURN_TOWARD_CENTER:
            # Use the shared controller command to face the field center.
            return ["TURN_TO_POINT;(0.0,0.0);40"]
        if Actions.TAKE_FROM_LOADER_TL.value <= selected.value <= Actions.TAKE_FROM_LOADER_BR.value:
            # Convert a Loader action into approach, turn, and clear commands.
            loader_index = selected.value - Actions.TAKE_FROM_LOADER_TL.value
            loader_target, _ = self._loader_wall_pose(robot.name, loader_index)
            wall_x = -FIELD_HALF if LOADER_POSITIONS[loader_index][0] < 0.0 else FIELD_HALF
            return [
                f"FOLLOW;({loader_target[0]:.1f}, {loader_target[1]:.1f});50",
                f"TURN_TO_POINT;({wall_x:.1f}, {loader_target[1]:.1f});30",
                "CLEAR_LOADER",
            ]
        # Actions without a direct controller sequence remain a short wait.
        return ["WAIT;0.5"]

    def action_to_name(self, action: int) -> str:
        # Return the display name for an action value.
        return self.get_action_name(action)

    def render_game_elements(self, ax: Any) -> None:
        # Draw Goals, Toggles, and visible field objects on a Matplotlib axis.
        import matplotlib.patches as patches

        goal_colors = {
            GoalType.RED_1: "red", GoalType.RED_2: "red",
            GoalType.BLUE_1: "blue", GoalType.BLUE_2: "blue",
        }
        for goal_index, goal_type in enumerate(self.goal_types, start=1):
            # Draw each Goal and its numbered display label.
            position = self.goal_positions[goal_type]
            ax.add_patch(patches.RegularPolygon(
                position, numVertices=8, radius=5.0,
                orientation=np.pi / 8,
                fill=False, edgecolor=goal_colors.get(goal_type, "black"),
                linewidth=2.0, zorder=3,
            ))
            ax.text(position[0], position[1] + 8.5, str(goal_index),
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="black", zorder=6)

        scored_pins_by_goal: Dict[str, List[Dict]] = {goal.value: [] for goal in GoalType}
        # Group scored Pins so they can be drawn in their Goal stacks.
        for obj in self.state["objects"]:
            if obj["status"] == ObjectStatus.SCORED and obj["kind"] == "pin" and obj.get("goal"):
                scored_pins_by_goal.setdefault(obj["goal"], []).append(obj)

        for goal_type in self.goal_types:
            position = self.goal_positions[goal_type]
            # Draw each scored Pin as two colored halves.
            pins = scored_pins_by_goal.get(goal_type.value, [])
            for offset_index, obj in enumerate(pins):
                pin_front = obj.get("front_color") or (obj["team"] or "yellow")
                pin_back = obj.get("back_color") or pin_front
                pin_x = position[0]
                pin_y = position[1] + offset_index * 2.2
                upper_color = pin_front if obj.get("face_up", True) else pin_back
                lower_color = pin_back if obj.get("face_up", True) else pin_front
                ax.add_patch(patches.Wedge(
                    (pin_x, pin_y), 2.1, 0.0, 180.0,
                    facecolor=upper_color, edgecolor="black", linewidth=0.6, zorder=4,
                ))
                ax.add_patch(patches.Wedge(
                    (pin_x, pin_y), 2.1, 180.0, 360.0,
                    facecolor=lower_color, edgecolor="black", linewidth=0.6, zorder=4,
                ))
                ax.add_patch(patches.Circle(
                    (pin_x, pin_y), 2.1, fill=False,
                    edgecolor="black", linewidth=0.7, zorder=5,
                ))

        for index, position in enumerate(TOGGLE_POSITIONS):
            # Draw Toggles using their current alliance color.
            toggle_color = self.state["toggles"][index] or "yellow"
            if position[0] == 0:
                toggle_xy = (position[0] - 12.0, position[1] - 2.0)
                toggle_width, toggle_height = 24.0, 4.0
            else:
                toggle_xy = (position[0] - 2.0, position[1] - 12.0)
                toggle_width, toggle_height = 4.0, 24.0
            ax.add_patch(patches.Rectangle(
                toggle_xy, toggle_width, toggle_height,
                facecolor=toggle_color, edgecolor="black", linewidth=1.0, zorder=3,
            ))

        for index, position in enumerate(LOADER_POSITIONS):
            # Draw each Loader and its remaining object count.
            loader_count = self.state["loaders"][index]
            loader_color = "#f4df00"
            if position[0] < 0:
                loader_xy = (position[0], position[1] - 3.0)
                loader_width, loader_height = 4.0, 6.0
            else:
                loader_xy = (position[0] - 4.0, position[1] - 3.0)
                loader_width, loader_height = 4.0, 6.0
            ax.add_patch(patches.Rectangle(
                loader_xy,
                loader_width, loader_height,
                fill=False, edgecolor=loader_color, linewidth=2.0, zorder=3,
            ))
            ax.text(
                position[0], position[1], str(loader_count),
                ha="center", va="center", fontsize=7, color=loader_color, zorder=4,
            )
        field_pin_offsets: Dict[Tuple[float, float], int] = {}
        for obj in self.state["objects"]:
            if obj["status"] == ObjectStatus.ON_FIELD:
                draw_position = obj["position"]
                if obj["kind"] == "pin":
                    position_key = tuple(
                        round(float(value), 3) for value in obj["position"]
                    )
                    pin_offset = field_pin_offsets.get(position_key, 0)
                    field_pin_offsets[position_key] = pin_offset + 1
                    if pin_offset:
                        draw_position = obj["position"].copy()
                        draw_position[1] += pin_offset * 2.2
                if obj["kind"] == "cup":
                    # Draw Cups as two-sided circles and outline a stacked Pin.
                    cup_radius = 2.4
                    upper_color = "#d9d9d9" if obj["face_up"] else "#666666"
                    lower_color = "#666666" if obj["face_up"] else "#d9d9d9"
                    has_stacked_pin = any(
                        other["kind"] == "pin"
                        and other["status"] == ObjectStatus.ON_FIELD
                        and np.allclose(other["position"], obj["position"])
                        for other in self.state["objects"]
                    )
                    ax.add_patch(patches.Wedge(
                        draw_position, cup_radius, 0.0, 180.0,
                        facecolor=upper_color, edgecolor="black", linewidth=0.7, zorder=6,
                    ))
                    ax.add_patch(patches.Wedge(
                        draw_position, cup_radius, 180.0, 360.0,
                        facecolor=lower_color, edgecolor="black", linewidth=0.7, zorder=6,
                    ))
                    ax.add_patch(patches.Circle(
                        draw_position, cup_radius, fill=False,
                        edgecolor="black", linewidth=0.8, zorder=7,
                    ))
                    if has_stacked_pin:
                        ax.add_patch(patches.Circle(
                            draw_position, cup_radius + 0.6, fill=False,
                            edgecolor="#f4df00", linewidth=1.0, zorder=5,
                        ))
                else:
                    # Draw Pins as colored halves and outline a stacked Cup.
                    pin_front = obj.get("front_color") or (obj["team"] or "yellow")
                    pin_back = obj.get("back_color") or pin_front
                    upper_color = pin_front if obj.get("face_up", True) else pin_back
                    lower_color = pin_back if obj.get("face_up", True) else pin_front
                    has_stacked_cup = any(
                        other["kind"] == "cup"
                        and other["status"] == ObjectStatus.ON_FIELD
                        and np.allclose(other["position"], obj["position"])
                        for other in self.state["objects"]
                    )
                    pin_radius = 1.7 if has_stacked_cup else 2.4
                    ax.add_patch(patches.Wedge(
                        draw_position, pin_radius, 0.0, 180.0,
                        facecolor=upper_color, edgecolor="black", linewidth=0.7, zorder=6,
                    ))
                    ax.add_patch(patches.Wedge(
                        draw_position, pin_radius, 180.0, 360.0,
                        facecolor=lower_color, edgecolor="black", linewidth=0.7, zorder=6,
                    ))
                    ax.add_patch(patches.Circle(
                        draw_position, pin_radius, fill=False,
                        edgecolor="black", linewidth=0.8, zorder=7,
                    ))

    def render_info_panel(self, ax_info: Any, agents: List[str] = None, actions: Optional[Dict] = None,
                          rewards: Optional[Dict] = None, num_steps: int = 0,
                          agent_times: Optional[Dict[str, float]] = None,
                          action_time_remaining: Optional[Dict[str, float]] = None) -> None:
        # Draw agent holdings and current alliance scores in the info panel.
        import matplotlib.patches as patches

        ax_info.axis("off")
        ax_info.set_frame_on(False)
        ax_info.patch.set_visible(False)
        ax_info.set_xlim(0.0, 1.2)
        ax_info.set_ylim(0.0, 1.0)
        ax_info.text(0.05, 0.95, "Override", fontweight="bold", va="top")

        goal_colors = {
            GoalType.RED_1: "red", GoalType.RED_2: "red",
            GoalType.BLUE_1: "blue", GoalType.BLUE_2: "blue",
        }

        y = 0.72
        robot_rows = []
        for agent in agents or self.state["agents"]:
            # Build one display row containing alliance, inventory, and action state.
            state = self.state["agents"][agent]
            current_action = state.get("current_action")
            try:
                current_action_name = self.action_to_name(int(current_action)) if current_action is not None else "IDLE"
            except Exception:
                current_action_name = str(current_action) if current_action is not None else "IDLE"
            robot_rows.append(
                (
                    agent,
                    f"{state['team']} | ID {agent} | P{state['held_pins']} C{state['held_cups']}",
                    f"Action: {current_action_name}",
                    y,
                )
            )
            y -= 0.06

        for _, id_label, action_label, row_y in robot_rows:
            # Render robot rows above the score summary.
            ax_info.text(0.05, row_y, id_label, va="top")
            ax_info.text(0.05, row_y - 0.02, action_label, va="top", fontweight="bold")
        ax_info.text(0.05, y - 0.03, str(self.compute_score()), va="top")

        ax_info.text(0.45, 0.16, "Goals", fontsize=10, fontweight="bold", va="bottom")
        goal_order = self.goal_types
        goal_counts = {goal.value: 0 for goal in goal_order}
        goal_colors_by_goal = {goal.value: [] for goal in goal_order}
        for obj in self.state["objects"]:
            # Count scored Pins and retain their visible colors for the Goal diagram.
            if obj["status"] == ObjectStatus.SCORED and obj.get("goal"):
                goal_value = obj["goal"]
                if obj["kind"] == "pin":
                    goal_counts[goal_value] = goal_counts.get(goal_value, 0) + 1
                    goal_colors_by_goal.setdefault(goal_value, []).append((obj.get("front_color") or obj.get("back_color") or "yellow"))

        goal_base_x = 0.08
        goal_y = 0.02
        goal_spacing = 0.09
        for index, goal_type in enumerate(goal_order, start=1):
            # Draw each numbered Goal and its scored Pin markers.
            goal_value = goal_type.value
            count = goal_counts.get(goal_value, 0)
            x = goal_base_x + index * goal_spacing
            ax_info.add_patch(patches.RegularPolygon(
                (x, goal_y), numVertices=8, radius=0.025,
                orientation=np.pi / 8,
                fill=False, edgecolor=goal_colors.get(goal_type, "black"),
                linewidth=1.5, zorder=3,
            ))
            ax_info.text(x, goal_y + 0.035, str(index), fontsize=7,
                         ha="center", va="center", color="black", fontweight="bold")
            for offset_index in range(count):
                pin_x = x + (offset_index % 3 - 1) * 0.022
                pin_y = goal_y + 0.028 + (offset_index // 3) * 0.018
                pin_color = goal_colors_by_goal.get(goal_value, ["yellow"])[offset_index % len(goal_colors_by_goal.get(goal_value, ["yellow"]))]
                upper_color = pin_color
                lower_color = "#666666" if pin_color == "yellow" else pin_color
                ax_info.add_patch(patches.Wedge(
                    (pin_x, pin_y), 0.013, 0.0, 180.0,
                    facecolor=upper_color, edgecolor="black", linewidth=0.5,
                ))
                ax_info.add_patch(patches.Wedge(
                    (pin_x, pin_y), 0.013, 180.0, 360.0,
                    facecolor=lower_color, edgecolor="black", linewidth=0.5,
                ))
            ax_info.text(x + 0.03, goal_y, f"{count}", fontsize=7, va="center", ha="left")



class VexUOverrideGame(OverrideGame):
    # VEX U Override competition variant.
    pass


Override = VexUOverrideGame
