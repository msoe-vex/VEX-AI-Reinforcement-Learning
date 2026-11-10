# VEX U implementation of the 2026-2027 VEX V5 Override game.

from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from vex_core.base_game import ActionEvent, ActionStep, VexGame
from vex_core.config import CommunicationOption
from vex_core.path_planner import Obstacle, PathPlanner
from vex_core.robot import Robot, RobotSize, Team
from vex_core.utils import (
    relative_polar,
    vex_atan2,
    vex_normalize_angle,
    vex_shortest_angular_distance,
)

FIELD_SIZE_INCHES = 144.0
FIELD_HALF = FIELD_SIZE_INCHES / 2.0
MIDFIELD_SIZE_INCHES = 24.0
NUM_CUPS = 48
NUM_PINS = 63
MAX_OBSERVED_PINS = NUM_PINS
PIN_OBJECT_FEATURES = 5
CUP_OBJECT_FEATURES = 4
ROBOT_POSITION_FEATURES = 3
MAX_VISIBLE_OBJECTS = 10
MAX_NEAREST_STACKS = 3
STACK_OBJECT_FEATURES = 7
NUM_GOALS = 9
GOAL_POLAR_FEATURES = 3
GOAL_PIN_COUNT_FEATURES = 3
GOAL_TOP_FEATURES = 6
GOAL_HEIGHT_FEATURES = 1
GOAL_FEATURES = (
    GOAL_POLAR_FEATURES
    + GOAL_PIN_COUNT_FEATURES
    + GOAL_TOP_FEATURES
    + GOAL_HEIGHT_FEATURES
)
NUM_TOGGLES = 4
MATCH_TIME = 120.0
AUTONOMOUS_TIME = 15.0
DRIVER_TIME = 105.0
FOV = 90.0
DEFAULT_DURATION = 0.5
DEFAULT_PENALTY = 1.0
OPPONENT_GOAL_PENALTY = 200.0
MAX_HELD_PINS = 1
MAX_HELD_CUPS = 1
VISIBILITY_POSITION_THRESHOLD = 3.0
VISIBILITY_ORIENTATION_THRESHOLD = 15.0
OPPONENT_SCAN_INTERVAL = 0.25
GROUND_OBJECT_HEIGHT = 1.5
STANDING_CUP_HEIGHT = 3.0
STACK_HEIGHT_INCREMENT = 3.25


class Actions(Enum):
    PICKUP_PIN_RED_YELLOW = 0
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
    FLIP_TOGGLE_1 = 21
    FLIP_TOGGLE_2 = 22
    FLIP_TOGGLE_3 = 23
    FLIP_TOGGLE_4 = 24
    PARK_MIDFIELD = 12
    TURN_TOWARD_CENTER = 13
    TAKE_FROM_LOADER_1 = 14
    TAKE_FROM_LOADER_2 = 15
    IDLE = 18
    PICKUP_PIN_BLUE_YELLOW = 25
    PICKUP_PIN_RED_BLUE = 26
    PICKUP_PIN_YELLOW_YELLOW = 27


GOAL_SCORE_ACTIONS = (
    Actions.SCORE_GOAL_1,
    Actions.SCORE_GOAL_2,
    Actions.SCORE_GOAL_3,
    Actions.SCORE_GOAL_4,
    Actions.SCORE_GOAL_5,
    Actions.SCORE_GOAL_6,
    Actions.SCORE_GOAL_7,
    Actions.SCORE_GOAL_8,
    Actions.SCORE_GOAL_9,
)
LOADER_ACTIONS = (
    Actions.TAKE_FROM_LOADER_1,
    Actions.TAKE_FROM_LOADER_2,
)
TOGGLE_ACTIONS = (
    Actions.FLIP_TOGGLE_1,
    Actions.FLIP_TOGGLE_2,
    Actions.FLIP_TOGGLE_3,
    Actions.FLIP_TOGGLE_4,
)


@dataclass(frozen=True)
class ActionDescriptor:
    """Semantic definition for one compact action-space entry."""

    action: Actions
    name: str
    kind: str
    target: Optional[int] = None


class ObjectStatus:
    ON_FIELD = 0
    HELD = 1
    SCORED = 2
    LOADER = 3


class ObsIndex:
    OBJECT_DISTANCE_OFFSET = 0
    OBJECT_SIN_THETA_OFFSET = 1
    OBJECT_COS_THETA_OFFSET = 2
    OBJECT_VISIBLE_OFFSET = 3
    PIN_COLOR_OFFSET = 4
    GOAL_DISTANCE_OFFSET = 0
    GOAL_SIN_THETA_OFFSET = 1
    GOAL_COS_THETA_OFFSET = 2
    GOAL_FRIENDLY_PIN_COUNT_OFFSET = 3
    GOAL_OPPONENT_PIN_COUNT_OFFSET = 4
    GOAL_NEUTRAL_PIN_COUNT_OFFSET = 5
    GOAL_TOP_EMPTY = 0
    GOAL_TOP_BLACK_CUP = 1
    GOAL_TOP_CLEAR_CUP = 2
    GOAL_TOP_FRIENDLY_PIN = 3
    GOAL_TOP_OPPONENT_PIN = 4
    GOAL_TOP_NEUTRAL_PIN = 5
    SELF_ORIENT = 0
    HELD_PINS = 1
    HELD_CUPS = 2
    PARKED = 3
    TIME_REMAINING = 4
    PIN_COUNT = 5
    CUP_COUNT = 6
    # Pin slots add an alliance-relative color after the four shared features.
    # Coordinates are populated only for the ten closest visible objects.
    PIN_POSITIONS = 7
    CUP_POSITIONS = PIN_POSITIONS + MAX_OBSERVED_PINS * PIN_OBJECT_FEATURES
    TOGGLES = CUP_POSITIONS + NUM_CUPS * CUP_OBJECT_FEATURES
    LOADERS = TOGGLES + NUM_TOGGLES
    LOADER_OWNERS = LOADERS + NUM_TOGGLES
    HELD_PIN_FRONT_COLOR = LOADER_OWNERS + NUM_TOGGLES
    HELD_PIN_BACK_COLOR = HELD_PIN_FRONT_COLOR + 1
    HELD_CUP_FACE_UP = HELD_PIN_BACK_COLOR + 1
    TEAMMATE_POSITION = HELD_CUP_FACE_UP + 1
    STACKS = TEAMMATE_POSITION + ROBOT_POSITION_FEATURES
    GOALS = STACKS + MAX_NEAREST_STACKS * STACK_OBJECT_FEATURES
    TOTAL = GOALS + NUM_GOALS * GOAL_FEATURES


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


GOAL_ALLIANCES = {
    GoalType.RED_1: "red",
    GoalType.RED_2: "red",
    GoalType.BLUE_1: "blue",
    GoalType.BLUE_2: "blue",
}


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
GOAL_RADII = {goal: 6.0 for goal in GoalType}
GOAL_STRUCTURE_RADIUS = 5.0
TOGGLE_RADIUS = 4.0
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
LOADER_TEAMS = ("red", "blue", "red", "blue")
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
    (-26.4, FIELD_HALF - 2.4), (-21.6, FIELD_HALF - 2.4),
    (21.6, FIELD_HALF - 2.4), (26.4, FIELD_HALF - 2.4),
    (FIELD_HALF - 2.4, -26.4), (FIELD_HALF - 2.4, -21.6),
    (FIELD_HALF - 2.4, 21.6), (FIELD_HALF - 2.4, 26.4),
    (-26.4, -FIELD_HALF + 2.4), (-21.6, -FIELD_HALF + 2.4),
    (21.6, -FIELD_HALF + 2.4), (26.4, -FIELD_HALF + 2.4),
    (-FIELD_HALF + 2.4, -26.4), (-FIELD_HALF + 2.4, -21.6),
    (-FIELD_HALF + 2.4, 21.6), (-FIELD_HALF + 2.4, 26.4),
]
TOGGLE_STACK_PIN_POSITIONS = [
    (-26.4, FIELD_HALF - 2.4), (26.4, FIELD_HALF - 2.4),
    (FIELD_HALF - 2.4, -26.4), (FIELD_HALF - 2.4, 26.4),
    (-26.4, -FIELD_HALF + 2.4), (26.4, -FIELD_HALF + 2.4),
    (-FIELD_HALF + 2.4, -26.4), (-FIELD_HALF + 2.4, 26.4),
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
PIN_COLOR_COMBINATIONS = (
    ("red", "yellow"),
    ("blue", "yellow"),
    ("red", "blue"),
    ("yellow", "yellow"),
)
PIN_PICKUP_ACTIONS = (
    Actions.PICKUP_PIN_RED_YELLOW,
    Actions.PICKUP_PIN_BLUE_YELLOW,
    Actions.PICKUP_PIN_RED_BLUE,
    Actions.PICKUP_PIN_YELLOW_YELLOW,
)
PERMANENT_OBSTACLES = [
    Obstacle(float(position[0]), float(position[1]), GOAL_RADII[goal_type], False)
    for goal_type, position in GOAL_POSITIONS.items()
] + [
    Obstacle(float(p[0]), float(p[1]), TOGGLE_RADIUS, False) for p in TOGGLE_POSITIONS
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
        self._action_descriptors = self._build_action_descriptors()
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
        return len(self._action_descriptors)

    @property
    def goal_types(self) -> Tuple[GoalType, ...]:
        # Return the Goals present in this game's field.
        return tuple(GoalType)

    @property
    def goal_action_map(self) -> Dict[Actions, GoalType]:
        """Map each active Goal's score action to its Goal explicitly."""
        if len(self.goal_types) > len(GOAL_SCORE_ACTIONS):
            raise ValueError(
                f"At most {len(GOAL_SCORE_ACTIONS)} scoreable Goals are supported"
            )
        return dict(zip(GOAL_SCORE_ACTIONS, self.goal_types))

    @property
    def loader_action_map(self) -> Dict[Actions, int]:
        return {action: index for index, action in enumerate(LOADER_ACTIONS)}

    @property
    def loader_positions(self) -> List[np.ndarray]:
        return LOADER_POSITIONS

    def _loader_index_for_action(
        self, agent: str, action: Optional[Actions]
    ) -> Optional[int]:
        action_index = self.loader_action_map.get(action) if action is not None else None
        if action_index is None:
            return None
        friendly_loader_indices = tuple(
            index for index, team in enumerate(LOADER_TEAMS)
            if team == self.get_team_for_agent(agent)
        )
        if len(friendly_loader_indices) != len(LOADER_ACTIONS):
            raise ValueError(
                f"Expected {len(LOADER_ACTIONS)} friendly Loaders for {agent}"
            )
        return friendly_loader_indices[action_index]

    @property
    def toggle_action_map(self) -> Dict[Actions, int]:
        return {action: index for index, action in enumerate(TOGGLE_ACTIONS)}

    @property
    def toggle_positions(self) -> List[np.ndarray]:
        return TOGGLE_POSITIONS

    def _goal_for_action(self, action: Optional[Actions]) -> Optional[GoalType]:
        return self.goal_action_map.get(action) if action is not None else None

    def _goal_objects(self, goal: GoalType) -> List[Dict]:
        goal_position = self.goal_positions[goal]
        return [
            obj for obj in self.state["objects"]
            if obj["kind"] in {"pin", "cup"}
            and obj["status"] in {ObjectStatus.ON_FIELD, ObjectStatus.SCORED}
            and np.linalg.norm(obj["position"] - goal_position) <= GOAL_RADII[goal]
        ]

    def _goal_stack_height(self, goal: GoalType) -> float:
        """Return the highest item currently occupying a Goal."""
        objects = self._goal_objects(goal)
        return max((float(obj.get("z", 0.0)) for obj in objects), default=0.0)

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

    def _build_action_descriptors(self) -> Tuple[ActionDescriptor, ...]:
        descriptors = [
            ActionDescriptor(
                action,
                action.name,
                "pickup_pin",
                index,
            )
            for index, action in enumerate(PIN_PICKUP_ACTIONS)
        ]
        descriptors.append(
            ActionDescriptor(Actions.PICKUP_CUP, "PICKUP_CUP", "pickup_cup")
        )
        descriptors.extend(
            ActionDescriptor(action, action.name, "score_goal", index)
            for index, action in enumerate(self.goal_action_map)
        )
        descriptors.extend(
            ActionDescriptor(action, action.name, "toggle", index)
            for index, action in enumerate(TOGGLE_ACTIONS)
        )
        descriptors.extend([
            ActionDescriptor(Actions.PARK_MIDFIELD, "PARK_MIDFIELD", "park"),
            ActionDescriptor(
                Actions.TURN_TOWARD_CENTER, "TURN_TOWARD_CENTER", "turn_center"
            ),
        ])
        descriptors.extend(
            ActionDescriptor(action, action.name, "take_loader", index)
            for index, action in enumerate(LOADER_ACTIONS)
        )
        descriptors.append(ActionDescriptor(Actions.IDLE, "IDLE", "idle"))
        return tuple(descriptors)

    @property
    def action_values(self) -> Tuple[int, ...]:
        # Expose the legacy action values in the compact policy order.
        return tuple(descriptor.action.value for descriptor in self._action_descriptors)

    def action_descriptor(self, action: int) -> Optional[ActionDescriptor]:
        try:
            index = int(action)
        except (IndexError, TypeError, ValueError):
            return None
        if index < 0 or index >= len(self._action_descriptors):
            return None
        return self._action_descriptors[index]

    def _decode_action(self, action: int) -> Optional[Actions]:
        descriptor = self.action_descriptor(action)
        return descriptor.action if descriptor is not None else None

    @property
    def fallback_action(self) -> int:
        # Return the safe action used when no action is available.
        return next(
            index for index, descriptor in enumerate(self._action_descriptors)
            if descriptor.kind == "turn_center"
        )

    def get_action_name(self, action: int) -> str:
        # Convert an action value into its enum name.
        descriptor = self.action_descriptor(action)
        return descriptor.name if descriptor is not None else str(action)

    def reset(self) -> None:
        # Clear the current game state before the environment reinitializes it.
        self.state = None
        self._visibility_revision = 0

    def _object(self, kind: str, position: np.ndarray, team: Optional[str] = None,
                face_up: Optional[bool] = None) -> Dict:
        # Create a field object record for a Pin or Cup.
        return {"kind": kind, "position": np.asarray(position, dtype=np.float32),
                "z": self._base_object_height(kind, face_up),
                "unstacked": False,
                "team": team, "face_up": face_up, "status": ObjectStatus.ON_FIELD,
            "held_by": None, "goal": None, "goal_stack_index": None}

    @staticmethod
    def _base_object_height(kind: str, face_up: Optional[bool]) -> float:
        """Return the height of an unstacked Pin or Cup in inches."""
        if kind == "cup" and face_up:
            return STANDING_CUP_HEIGHT
        return GROUND_OBJECT_HEIGHT

    def _refresh_object_heights(self) -> None:
        """Recompute object z coordinates from their current stack placement."""
        stack_groups: Dict[Tuple, List[Dict]] = {}
        for obj in self.state["objects"]:
            if obj["status"] == ObjectStatus.SCORED and obj.get("goal") is not None:
                key = ("goal", obj["goal"])
                stack_index = int(obj.get("goal_stack_index") or 0)
            elif obj["status"] == ObjectStatus.LOADER:
                key = ("loader", obj.get("loader_index", -1))
                stack_index = int(obj.get("loader_stack", 0))
            elif obj["status"] == ObjectStatus.HELD:
                key = ("held", obj.get("held_by"))
                stack_index = 0
            else:
                position = np.asarray(obj["position"], dtype=np.float32)
                if obj.get("unstacked", False):
                    key = ("unstacked", id(obj))
                    stack_groups.setdefault(key, []).append((-1, obj))
                    continue
                nearest_goal = next(
                    (
                        goal.value for goal in self.goal_types
                        if np.linalg.norm(position - self.goal_positions[goal]) <= GOAL_RADII[goal]
                    ),
                    None,
                )
                if nearest_goal is not None:
                    key = ("goal", nearest_goal)
                else:
                    key = (
                        "field",
                        round(float(position[0]), 3),
                        round(float(position[1]), 3),
                    )
                stack_index = -1

            stack_groups.setdefault(key, []).append((stack_index, obj))

        for key, entries in stack_groups.items():
            if key[0] == "held":
                entries.sort(
                    key=lambda item: (
                        0 if item[1]["kind"] == "cup" else 1,
                        item[1].get("kind", ""),
                    )
                )
            elif key[0] == "field":
                entries.sort(
                    key=lambda item: (
                        0 if item[1]["kind"] == "cup" else 1,
                        item[1].get("kind", ""),
                    )
                )
            else:
                entries.sort(key=lambda item: (item[0], item[1].get("kind", "")))

            stack_height = 0.0
            for index, (_, obj) in enumerate(entries):
                if index == 0:
                    stack_height = self._base_object_height(
                        obj["kind"], obj.get("face_up")
                    )
                else:
                    stack_height += STACK_HEIGHT_INCREMENT
                obj["z"] = stack_height

    def get_initial_state(
        self,
        randomize: bool = False,
        seed: Optional[int] = None,
    ) -> Dict:
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
            objects[obj_index]["position"] = GOAL_POSITIONS[goal_type].copy()
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

        loader_teams = LOADER_TEAMS
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
                    "pin", self.loader_positions[loader_index].copy(), team="yellow", face_up=True,
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
        if randomize:
            for obj in objects:
                if obj["status"] == ObjectStatus.ON_FIELD:
                    obj["position"] = np.random.uniform(-66, 66, 2).astype(np.float32)
        self._refresh_object_heights()
        # Force the first observation of this layout to perform a fresh scan.
        self._visibility_revision += 1
        return self.state

    def _visible(
        self,
        agent: str,
        kind: str,
        include_goal_objects: bool = False,
    ) -> List[Tuple[float, int]]:
        # Return visible field objects of a type sorted by distance.
        state = self.state["agents"][agent]
        # Include the robot's camera offset when calculating its viewing direction.
        camera = vex_normalize_angle(float(state["orientation"][0]) + state["camera_rotation_offset"])
        cache_key = (kind, include_goal_objects)
        cache = state.setdefault("visibility_cache", {}).get(cache_key)
        if cache is not None:
            moved = np.linalg.norm(state["position"] - cache["position"]) > VISIBILITY_POSITION_THRESHOLD
            rotated = abs(vex_shortest_angular_distance(camera, cache["camera"])) > VISIBILITY_ORIENTATION_THRESHOLD
            if cache["revision"] == self._visibility_revision and not moved and not rotated:
                return cache["visible"]
        visible = []
        for index, obj in enumerate(self.state["objects"]):
            allowed_statuses = {ObjectStatus.ON_FIELD}
            if include_goal_objects:
                # Goal-stacked objects remain field-visible at their observed coordinates.
                allowed_statuses.add(ObjectStatus.SCORED)
            if obj["kind"] != kind or obj["status"] not in allowed_statuses:
                continue
            direction = obj["position"] - state["position"]
            distance = float(np.linalg.norm(direction))
            # Keep only objects inside both the camera range and field of view.
            if distance <= 72 and abs(vex_shortest_angular_distance(camera, vex_atan2(direction[0], direction[1]))) <= FOV / 2:
                visible.append((distance, index))
        visible = sorted(visible)
        state.setdefault("visibility_cache", {})[cache_key] = {
            "position": state["position"].copy(),
            "camera": camera,
            "revision": self._visibility_revision,
            "visible": visible,
        }
        return visible

    @staticmethod
    def _object_goal_color(obj: Dict) -> Optional[str]:
        """Return the exposed color for a Pin or Cup's top-facing feature."""
        if obj["kind"] == "cup":
            return "clear" if bool(obj.get("face_up", False)) else "black"
        color = (
            obj.get("front_color")
            if bool(obj.get("face_up", True))
            else obj.get("back_color")
        )
        return color if color in {"red", "blue", "yellow"} else None

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
    def _color_to_alliance_value(color: Optional[str], alliance: str) -> float:
        # Encode an alliance color relative to this robot; yellow/unknown is neutral.
        if color == "yellow" or color not in {"red", "blue"}:
            return 0.0
        return 1.0 if color == alliance else -1.0

    def get_game_observation(self, agent: str, game_time: float = 0.0) -> np.ndarray:
        state = self.state["agents"][agent]
        pin_visible = self._visible(agent, "pin", include_goal_objects=True)
        cup_visible = self._visible(agent, "cup", include_goal_objects=True)
        camera_angle = vex_normalize_angle(
            float(state["orientation"][0]) + state["camera_rotation_offset"]
        )
        values = [
            camera_angle,
            state["held_pins"],
            state["held_cups"],
            float(state["parked"]),
            self.total_time - game_time,
            len(pin_visible),
            len(cup_visible),
        ]

        for kind, visible, max_objects in (
            ("pin", pin_visible, MAX_OBSERVED_PINS),
            ("cup", cup_visible, NUM_CUPS),
        ):
            objects = [
                (index, obj)
                for index, obj in enumerate(self.state["objects"])
                if obj["kind"] == kind
            ]
            if len(objects) > max_objects:
                raise ValueError(
                    f"Observation supports at most {max_objects} {kind}s, "
                    f"but the game state contains {len(objects)}"
                )
            visible_indices = {index for _, index in visible}
            nearest_indices = {
                index for _, index in visible[:MAX_VISIBLE_OBJECTS]
            }
            alliance = self.get_team_for_agent(agent)

            for index, obj in objects:
                if index in nearest_indices:
                    values.extend(relative_polar(
                        obj["position"],
                        state["position"],
                        camera_angle,
                    ))
                else:
                    values.extend([-144.0, 0.0, 0.0])
                values.append(float(index in visible_indices))
                if kind == "pin":
                    color = self._object_goal_color(obj) if index in visible_indices else None
                    values.append(self._color_to_alliance_value(color, alliance))

            for _ in range(max_objects - len(objects)):
                values.extend([-144.0, 0.0, 0.0, 0.0])
                if kind == "pin":
                    values.append(0.0)

        alliance = self.get_team_for_agent(agent)
        values.extend(
            self._color_to_alliance_value(toggle, alliance)
            for toggle in self.state["toggles"]
        )
        values.extend(float(count > 0) for count in self.state["loaders"])
        values.extend(
            1.0 if loader_team == alliance else -1.0
            for loader_team in LOADER_TEAMS
        )

        held_pin_order = state.get("held_pin_order", [None, None])
        values.extend([
            self._color_to_alliance_value(
                held_pin_order[0] if state["held_pins"] > 0 else None,
                alliance,
            ),
            self._color_to_alliance_value(
                held_pin_order[1] if state["held_pins"] > 0 else None,
                alliance,
            ),
            1.0 if state["held_cups"] > 0 and state.get("held_cup_face_up", False) else 0.0,
        ])

        teammate = next(
            (
                other for name, other in self.state["agents"].items()
                if name != agent and other["team"] == state["team"]
            ),
            None,
        )
        if teammate is None:
            values.extend([-144.0, 0.0, 0.0])
        else:
            values.extend(relative_polar(
                teammate["position"], state["position"], camera_angle
            ))

        stack_groups: Dict[Tuple[float, float], List[Dict]] = {}
        for obj in self.state["objects"]:
            if obj["status"] not in {ObjectStatus.ON_FIELD, ObjectStatus.SCORED}:
                continue
            if obj.get("unstacked", False):
                continue
            position = np.asarray(obj["position"], dtype=np.float32)
            goal_position = next(
                (
                    self.goal_positions[goal]
                    for goal in self.goal_types
                    if np.linalg.norm(position - self.goal_positions[goal]) <= GOAL_RADII[goal]
                ),
                None,
            )
            if goal_position is not None:
                position = goal_position
            key = (round(float(position[0]), 3), round(float(position[1]), 3))
            stack_groups.setdefault(key, []).append(obj)

        stack_candidates = []
        for position, stack_objects in stack_groups.items():
            if len(stack_objects) < 2:
                continue
            stack_position = np.asarray(position, dtype=np.float32)
            stack_candidates.append((
                float(np.linalg.norm(stack_position - state["position"])),
                stack_position,
                stack_objects,
            ))
        stack_candidates.sort(key=lambda item: item[0])

        for _, stack_position, stack_objects in stack_candidates[:MAX_NEAREST_STACKS]:
            values.extend(relative_polar(
                stack_position,
                state["position"],
                camera_angle,
            ))
            values.extend([
                max(float(obj["z"]) for obj in stack_objects),
                float(sum(obj["kind"] == "pin" for obj in stack_objects)),
                float(sum(obj["kind"] == "cup" for obj in stack_objects)),
                float(any(
                    obj["kind"] == "pin" and float(obj["z"]) >= 6.25
                    for obj in stack_objects
                )),
            ])

        for _ in range(MAX_NEAREST_STACKS - min(len(stack_candidates), MAX_NEAREST_STACKS)):
            values.extend([-144.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self._update_opponent_memory(agent, game_time)
        opponent = "blue" if alliance == "red" else "red"
        active_goals = tuple(self.goal_types)
        for goal_type in active_goals:
            goal_position = self.goal_positions[goal_type]
            values.extend(relative_polar(
                goal_position, state["position"], camera_angle
            ))
            goal_objects = [
                (index, obj)
                for index, obj in enumerate(self.state["objects"])
                if any(
                    candidate is obj
                    for candidate in self._goal_objects(goal_type)
                )
            ]

            pin_counts = {"friendly": 0.0, "opponent": 0.0, "neutral": 0.0}
            for _, obj in goal_objects:
                if obj["kind"] != "pin":
                    continue
                color = self._object_goal_color(obj)
                if color == alliance:
                    pin_counts["friendly"] += 1.0
                elif color == opponent:
                    pin_counts["opponent"] += 1.0
                else:
                    pin_counts["neutral"] += 1.0
            values.extend(pin_counts.values())

            if goal_objects:
                _, top_object = max(
                    goal_objects,
                    key=lambda item: (
                        item[1].get("goal_stack_index") is None,
                        int(item[1].get("goal_stack_index") or 0),
                        item[0],
                    ),
                )
                if top_object["kind"] == "cup":
                    top_feature = (
                        ObsIndex.GOAL_TOP_BLACK_CUP
                        if self._object_goal_color(top_object) == "black"
                        else ObsIndex.GOAL_TOP_CLEAR_CUP
                    )
                else:
                    top_color = self._object_goal_color(top_object)
                    if top_color == alliance:
                        top_feature = ObsIndex.GOAL_TOP_FRIENDLY_PIN
                    elif top_color == opponent:
                        top_feature = ObsIndex.GOAL_TOP_OPPONENT_PIN
                    else:
                        top_feature = ObsIndex.GOAL_TOP_NEUTRAL_PIN
            else:
                top_feature = ObsIndex.GOAL_TOP_EMPTY
            values.extend(
                1.0 if index == top_feature else 0.0
                for index in range(GOAL_TOP_FEATURES)
            )
            values.append(self._goal_stack_height(goal_type))

        for _ in range(NUM_GOALS - len(active_goals)):
            values.extend([-144.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            values.extend([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            values.append(0.0)

        observation = np.asarray(values, dtype=np.float32)
        if observation.shape != (ObsIndex.TOTAL,):
            raise ValueError(
                f"Override observation has shape {observation.shape}; "
                f"expected {(ObsIndex.TOTAL,)}"
            )
        return observation

    def get_game_observation_space(self, agent: str) -> spaces.Space:
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
        position = self.loader_positions[loader_index]
        robot = self.get_robot_for_agent(agent)
        wall_clearance = float(robot.total_radius) + 0.5 if robot else 0.0
        # Keep the robot's conservative circular footprint inside the field.
        if position[0] < 0.0:
            return np.array([
                -FIELD_HALF + wall_clearance,
                position[1],
            ], dtype=np.float32), 270.0
        return np.array([
            FIELD_HALF - wall_clearance,
            position[1],
        ], dtype=np.float32), 90.0

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

    def _target_next_to_toggle(self, agent: str, index: int) -> Tuple[np.ndarray, float]:
        # Approach from inside the field with the robot's front facing the Toggle.
        toggle_position = self.toggle_positions[index]
        inward = -toggle_position / float(np.linalg.norm(toggle_position))
        robot_length, _ = self.get_robot_dimensions(agent)
        clearance = TOGGLE_RADIUS + robot_length / 2.0
        target = toggle_position + inward * clearance
        facing_angle = vex_atan2(-inward[0], -inward[1])
        return target.astype(np.float32), float(facing_angle)

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

    def _pickup_stack_at(self, index: int) -> List[int]:
        """Return all field Pins and Cups occupying the selected object's stack."""
        selected = self.state["objects"][index]
        if selected["status"] != ObjectStatus.ON_FIELD:
            return [index]
        return [
            object_index for object_index, obj in enumerate(self.state["objects"])
            if obj["status"] == ObjectStatus.ON_FIELD
            and obj["kind"] in {"pin", "cup"}
            and np.allclose(obj["position"], selected["position"])
        ]

    def _pickup_item_index(self, agent: str, descriptor: ActionDescriptor) -> Optional[int]:
        kind = "cup" if descriptor.action == Actions.PICKUP_CUP else "pin"
        visible = self._visible(agent, kind)
        if kind == "pin":
            if descriptor.target is None:
                raise ValueError(
                    f"Pin pickup action {descriptor.name} has no color combination"
                )
            color_pair = PIN_COLOR_COMBINATIONS[descriptor.target]
            visible = [
                item
                for item in visible
                if (
                    self.state["objects"][item[1]].get("front_color"),
                    self.state["objects"][item[1]].get("back_color"),
                ) == color_pair
            ]
        return visible[0][1] if visible else None

    def execute_action(self, agent: str, action: int) -> Tuple[List[ActionStep], float]:
        # Translate a high-level action into timed steps and a penalty.
        state = self.state["agents"][agent]
        descriptor = self.action_descriptor(action)
        if descriptor is None:
            # Invalid actions become a short no-op and receive the default penalty.
            return [ActionStep(0.1, state["position"].copy(), state["orientation"].copy())], DEFAULT_PENALTY
        selected = descriptor.action
        if descriptor.kind not in ("idle", "park"):
            # Leaving the park zone removes the robot's parking status and bonus.
            state["parked"] = False
            state["parked_zone"] = None
        if descriptor.kind == "idle":
            # Idling is free only after the robot has parked.
            return [ActionStep(0.1, state["position"].copy(), state["orientation"].copy())], 0.0 if state["parked"] else DEFAULT_PENALTY
        if descriptor.kind == "turn_center":
            # Turn toward the field center while accounting for camera offset.
            angle = vex_atan2(-state["position"][0], -state["position"][1]) - state["camera_rotation_offset"]
            return [ActionStep(DEFAULT_DURATION, state["position"].copy(), np.array([angle], dtype=np.float32), [ActionEvent("turn", {"angle": angle})])], 0.0
        if descriptor.kind in ("pickup_pin", "pickup_cup"):
            # Select the nearest visible object of the requested type.
            kind = "pin" if descriptor.kind == "pickup_pin" else "cup"
            visible = self._visible(agent, kind)
            if descriptor.kind == "pickup_pin":
                if descriptor.target is None:
                    raise ValueError(
                        f"Pin pickup action {descriptor.name} has no color combination"
                    )
                color_pair = PIN_COLOR_COMBINATIONS[descriptor.target]
                visible = [
                    item
                    for item in visible
                    if (
                        self.state["objects"][item[1]].get("front_color"),
                        self.state["objects"][item[1]].get("back_color"),
                    ) == color_pair
                ]
            held_key = f"held_{kind}s"
            capacity = MAX_HELD_PINS if kind == "pin" else MAX_HELD_CUPS
            if not visible or state[held_key] >= capacity:
                index = visible[0][1] if visible else None
                stack_indices = self._pickup_stack_at(index) if index is not None else []
                if len(stack_indices) <= 1 or not (state["held_pins"] or state["held_cups"]):
                    return [ActionStep(0.1, state["position"].copy(), state["orientation"].copy())], DEFAULT_PENALTY
            index = visible[0][1]
            stack_indices = self._pickup_stack_at(index)
            stack_pickup = len(stack_indices) > 1
            if stack_pickup:
                delta = self.state["objects"][index]["position"] - state["position"]
                reverse_heading = vex_atan2(float(delta[0]), float(delta[1])) + 180.0
                event = ActionEvent(
                    "pickup",
                    {"index": index, "stack_indices": stack_indices, "replace_held": True},
                )
                state["_pending_stack_pickup"] = True
                return self._move(
                    agent,
                    self.state["objects"][index]["position"],
                    event,
                    reverse_heading,
                ), 0.0
            # Apply the pickup event after the movement plan reaches the object.
            return self._move(agent, self.state["objects"][index]["position"], ActionEvent("pickup", {"index": index})), 0.0
        goal = self._goal_for_action(selected)
        if goal is not None:
            # Score the Pin, Cup, or pair currently held by the robot.
            has_pin = state["held_pins"] > 0
            has_cup = state["held_cups"] > 0
            if not has_pin and not has_cup:
                return [ActionStep(0.1, state["position"].copy(), state["orientation"].copy())], DEFAULT_PENALTY
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
            penalty = (
                OPPONENT_GOAL_PENALTY
                if GOAL_ALLIANCES.get(goal) not in (None, state["team"])
                else 0.0
            )
            return self._move(
                agent, self._target_next_to_goal(agent, goal),
                ActionEvent("score", {"kind": kind, "goal": goal.value, "paired": paired}),
            ), penalty
        if selected in self.loader_action_map:
            # Approach the selected wall Loader, face it, and clear its contents.
            loader_index = self._loader_index_for_action(agent, selected)
            if loader_index is None:
                raise ValueError(f"Unable to resolve Loader action {selected}")
            loader_count = self.state["loaders"][loader_index]
            if loader_count <= 0 or (
                    state["held_pins"] >= MAX_HELD_PINS
                    and state["held_cups"] >= MAX_HELD_CUPS
            ):
                return [ActionStep(0.1, state["position"].copy(), state["orientation"].copy())], DEFAULT_PENALTY
            loader_target, loader_orientation = self._loader_wall_pose(agent, loader_index)
            event = ActionEvent("clear_loader", {"loader_index": loader_index})
            return self._move(agent, loader_target, event, loader_orientation), 0.0
        if selected in self.toggle_action_map:
            # Approach the selected Toggle from inside the field before changing it.
            index = self.toggle_action_map[selected]
            holder = self.state["toggle_holders"][index]
            if holder is not None and holder != agent:
                return [ActionStep(0.1, state["position"].copy(), state["orientation"].copy())], DEFAULT_PENALTY
            toggle_target, toggle_orientation = self._target_next_to_toggle(agent, index)
            return self._move(
                agent,
                toggle_target,
                ActionEvent("toggle", {"index": index}),
                toggle_orientation,
            ), 0.0
        if descriptor.kind == "park":
            # Move next to the center Goal to complete the Midfield park.
            return self._move(
                agent, self._target_next_to_goal(agent, GoalType.TALL), ActionEvent("park")
            ), 0.0
        return [ActionStep(0.1, state["position"].copy(), state["orientation"].copy())], DEFAULT_PENALTY

    def update_tracker(self, agent: str, action: int) -> None:
        # Update inferred held-object, parking, and Toggle state after an action.
        state = self.state["agents"][agent]
        descriptor = self.action_descriptor(action)
        if descriptor is None:
            # Do not update tracker state for an undecodable action.
            return
        selected = descriptor.action
        if descriptor.kind not in ("idle", "park"):
            # Keep inferred parking state synchronized when the robot leaves the zone.
            state["parked"] = False
            state["parked_zone"] = None
        if descriptor.kind in ("pickup_pin", "pickup_cup") and state.pop("_pending_stack_pickup", False):
            state["held_pins"] = 1
            state["held_cups"] = 1
        elif descriptor.kind == "pickup_pin" and state["held_pins"] < MAX_HELD_PINS:
            # Reflect a successful Pin pickup in the inferred inventory.
            state["held_pins"] += 1
        elif descriptor.kind == "pickup_cup" and state["held_cups"] < MAX_HELD_CUPS:
            # Reflect a successful Cup pickup in the inferred inventory.
            state["held_cups"] += 1
        elif self._goal_for_action(selected) is not None:
            # Scoring transfers all held objects out of the inferred inventory.
            state["held_pins"] = 0
            state["held_cups"] = 0
        elif descriptor.kind == "park":
            # Mark the robot as parked for observations and scoring.
            state["parked"] = True
        elif selected in self.loader_action_map:
            # Reflect the Pin/Cup stack supplied by a Loader when possible.
            loader_index = self._loader_index_for_action(agent, selected)
            if loader_index is None:
                raise ValueError(f"Unable to resolve Loader action {selected}")
            if self.state["loaders"][loader_index] > 0 and state["held_pins"] < MAX_HELD_PINS:
                state["held_pins"] += 1
            if self.state["loaders"][loader_index] > 0 and state["held_cups"] < MAX_HELD_CUPS:
                state["held_cups"] += 1
        elif selected in self.toggle_action_map:
            # Record the alliance color inferred for the selected Toggle.
            toggle_index = self.toggle_action_map[selected]
            if self.state["toggle_holders"][toggle_index] not in (None, agent):
                return
            toggle_colors = list(state.get("inferred_toggle_colors", [None] * NUM_TOGGLES))
            toggle_colors[toggle_index] = self.get_team_for_agent(agent)
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
                observation[ObsIndex.TOGGLES + index] = (
                    self._color_to_alliance_value(
                        color, self.get_team_for_agent(agent)
                    )
                )

        held_pin_order = state.get("held_pin_order", [None, None])
        pin_front = held_pin_order[0] if state["held_pins"] > 0 else None
        pin_back = held_pin_order[1] if state["held_pins"] > 0 else None
        alliance = self.get_team_for_agent(agent)
        observation[ObsIndex.HELD_PIN_FRONT_COLOR] = self._color_to_alliance_value(
            pin_front, alliance
        )
        observation[ObsIndex.HELD_PIN_BACK_COLOR] = self._color_to_alliance_value(
            pin_back, alliance
        )
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
                # Move the selected field object, or its entire stack, into inventory.
                if event.data.get("replace_held"):
                    drop_position = state["position"].copy()
                    for held_obj in self.state["objects"]:
                        if held_obj["status"] == ObjectStatus.HELD and held_obj["held_by"] == agent:
                            held_obj.update(status=ObjectStatus.ON_FIELD, held_by=None,
                                            position=drop_position.copy())
                            held_obj["unstacked"] = True
                    state["held_pins"] = 0
                    state["held_cups"] = 0
                    state["held_pin_order"] = [None, None]
                    state["held_cup_face_up"] = False
                    state["held_stack"] = []

                pickup_indices = event.data.get("stack_indices", [event.data["index"]])
                for pickup_index in pickup_indices:
                    obj = self.state["objects"][int(pickup_index)]
                    held_key = f"held_{obj['kind']}s"
                    capacity = MAX_HELD_PINS if obj["kind"] == "pin" else MAX_HELD_CUPS
                    if obj["status"] != ObjectStatus.ON_FIELD or state[held_key] >= capacity:
                        continue
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
                            position=self.goal_positions[GoalType(goal_value)].copy(),
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
                # Transfer one Pin/Cup stack from the Loader whenever capacity allows.
                loader_index = int(event.data["loader_index"])
                if self.state["loaders"][loader_index] > 0:
                    loader_objects = sorted(
                        (
                            (index, obj) for index, obj in enumerate(self.state["objects"])
                            if obj["status"] == ObjectStatus.LOADER
                            and obj.get("loader_index") == loader_index
                        ),
                        key=lambda item: item[1].get("loader_stack", 0),
                    )
                    for kind, capacity in (
                        ("pin", MAX_HELD_PINS),
                        ("cup", MAX_HELD_CUPS),
                    ):
                        if state[f"held_{kind}s"] >= capacity:
                            continue
                        matching = next(
                            (
                                (index, obj) for index, obj in loader_objects
                                if obj["kind"] == kind
                            ),
                            None,
                        )
                        if matching is None:
                            continue
                        object_index, obj = matching
                        obj.update(status=ObjectStatus.HELD, held_by=agent)
                        state[f"held_{kind}s"] += 1
                        if kind == "pin":
                            ordered_colors = [obj.get("front_color"), obj.get("back_color")]
                            if not bool(obj.get("face_up", True)):
                                ordered_colors.reverse()
                            state["held_pin_order"] = ordered_colors
                        else:
                            state["held_cup_face_up"] = bool(obj.get("face_up", False))
                    held_indices = [
                        index for index, held_obj in enumerate(self.state["objects"])
                        if held_obj["status"] == ObjectStatus.HELD and held_obj["held_by"] == agent
                    ]
                    state["held_stack"] = sorted(
                        held_indices,
                        key=lambda index: 0 if self.state["objects"][index]["kind"] == "cup" else 1,
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
            self._refresh_object_heights()
            # Invalidate cached field visibility after object status changes.
            self._visibility_revision += 1

    def _yellow_pin_award_color(self, goal_value: str) -> Optional[str]:
        """Return the alliance that receives yellow-pin points in a Goal."""
        if GoalType(goal_value) == GoalType.TALL:
            parked_counts = {
                alliance: sum(
                    agent.get("team") == alliance
                    and agent.get("parked", False)
                    and agent.get("parked_zone") == "midfield"
                    for agent in self.state["agents"].values()
                )
                for alliance in ("red", "blue")
            }
            if parked_counts["red"] == parked_counts["blue"]:
                return None
            return max(parked_counts, key=parked_counts.get)

        goal_type = GoalType(goal_value)
        goal_position = self.goal_positions[goal_type]
        distances = [
            float(np.linalg.norm(goal_position - toggle_position))
            for toggle_position in self.toggle_positions
        ]
        nearest_distance = min(distances)
        nearest_toggles = [
            index for index, distance in enumerate(distances)
            if np.isclose(distance, nearest_distance)
        ]
        if len(nearest_toggles) != 1:
            return None
        toggle_color = self.state["toggles"][nearest_toggles[0]]
        return toggle_color if toggle_color in {"red", "blue"} else None

    def _score_pin_color(
        self,
        pin: Dict,
        goal_value: str,
        visible_top_half: bool = True,
    ) -> Optional[Tuple[str, int]]:
        """Return the alliance and value of the exposed half of a Pin."""
        face_up = bool(pin.get("face_up", True))
        color = (
            pin.get("front_color")
            if face_up == visible_top_half
            else pin.get("back_color")
        )
        if color not in {"red", "blue", "yellow"}:
            return None
        if color == "yellow":
            # Center Goal Pins use the midfield parking majority; other yellow
            # Pins use the alliance controlling the Goal's quadrant.
            award_color = self._yellow_pin_award_color(goal_value)
            if award_color is None:
                return None
            return award_color, 10
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
                # The Goal hides the bottom half of the first Pin; a face-up Cup
                # hides the top half of the Pin immediately beneath it.
                bottom_covered = index == 0
                top_covered = (
                    index + 1 < len(stack)
                    and stack[index + 1]["kind"] == "cup"
                    and stack[index + 1].get("face_up", True)
                )
                if bottom_covered and top_covered:
                    # Both halves are blocked, so this Pin has no exposed scoring face.
                    continue
                scored = self._score_pin_color(
                    obj,
                    goal_value,
                    visible_top_half=not top_covered,
                )
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
            if agent.get("parked", False) and agent.get("parked_zone") == "midfield":
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
        state = self.state["agents"][agent]
        descriptor = self.action_descriptor(action)
        if descriptor is None:
            return False
        selected = descriptor.action
        if descriptor.kind == "pickup_pin" and state["held_pins"] >= MAX_HELD_PINS:
            return False
        if descriptor.kind == "pickup_cup" and state["held_cups"] >= MAX_HELD_CUPS:
            return False
        if self._goal_for_action(selected) is not None and (
            observation[ObsIndex.HELD_PINS] <= 0
            and observation[ObsIndex.HELD_CUPS] <= 0
        ):
            # A scoring action requires at least one held object.
            return False
        if descriptor.kind == "idle" and not state["parked"]:
            return False
        if selected in self.loader_action_map:
            loader_index = self._loader_index_for_action(agent, selected)
            if loader_index is None:
                return False
            if observation[ObsIndex.LOADERS + loader_index] < 1:
                return False
        return not (descriptor.kind == "park" and state["parked"])

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
            Obstacle(float(p[0]), float(p[1]), TOGGLE_RADIUS, False) for p in self.toggle_positions
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
        descriptor = self.action_descriptor(action)
        if descriptor is None:
            return ["WAIT;0.5"]

        if descriptor.kind == "pickup_item":
            item_index = self._pickup_item_index(robot.name, descriptor)
            if item_index is None:
                return ["WAIT;0.5"]
            state = self.state["agents"].get(robot.name)
            start_position = (
                np.asarray(state["position"], dtype=np.float32)
                if state is not None
                else np.asarray(robot.start_position, dtype=np.float32)
            )
            item_position = self.state["objects"][item_index]["position"]
            path, _, _, _ = self.path_planner.Solve(
                start_position,
                item_position,
                self.get_permanent_obstacles(),
                robot,
                optimize=False,
            )
            path_text = ",".join(
                f"({float(point[0]):.1f}, {float(point[1]):.1f})"
                for point in path
            )
            reverse = int(len(self._pickup_stack_at(item_index)) > 1)
            return [f"PICKUP_ITEM;{path_text};{reverse}"]

        if descriptor.kind == "score_goal":
            goal = self._goal_for_action(descriptor.action)
            if goal is None:
                return ["WAIT;0.5"]
            state = self.state["agents"].get(robot.name)
            start_position = (
                np.asarray(state["position"], dtype=np.float32)
                if state is not None
                else np.asarray(robot.start_position, dtype=np.float32)
            )
            target_position = self._target_next_to_goal(robot.name, goal)
            path, _, _, _ = self.path_planner.Solve(
                start_position,
                target_position,
                self.get_permanent_obstacles(),
                robot,
                optimize=False,
            )
            path_text = ",".join(
                f"({float(point[0]):.1f}, {float(point[1]):.1f})"
                for point in path
            )
            return [
                f"SCORE;{path_text};{self._goal_stack_height(goal):.2f}"
            ]

        if descriptor.kind == "toggle":
            toggle_index = descriptor.target
            if toggle_index is None or toggle_index >= len(self.toggle_positions):
                return ["WAIT;0.5"]
            state = self.state["agents"].get(robot.name)
            start_position = (
                np.asarray(state["position"], dtype=np.float32)
                if state is not None
                else np.asarray(robot.start_position, dtype=np.float32)
            )
            target_position, _ = self._target_next_to_toggle(robot.name, toggle_index)
            path, _, _, _ = self.path_planner.Solve(
                start_position,
                target_position,
                self.get_permanent_obstacles(),
                robot,
                optimize=False,
            )
            path_text = ",".join(
                f"({float(point[0]):.1f}, {float(point[1]):.1f})"
                for point in path
            )
            toggle_color = self.state["toggles"][toggle_index] or "yellow"
            toggle_color_value = self._color_to_alliance_value(
                toggle_color,
                robot.team.value,
            )
            return [f"TOGGLE;{path_text};{toggle_color_value:.0f}"]

        if descriptor.kind == "park":
            state = self.state["agents"].get(robot.name)
            start_position = (
                np.asarray(state["position"], dtype=np.float32)
                if state is not None
                else np.asarray(robot.start_position, dtype=np.float32)
            )
            target_position = self._target_next_to_goal(robot.name, GoalType.TALL)
            path, _, _, _ = self.path_planner.Solve(
                start_position,
                target_position,
                self.get_permanent_obstacles(),
                robot,
                optimize=False,
            )
            path_text = ",".join(
                f"({float(point[0]):.1f}, {float(point[1]):.1f})"
                for point in path
            )
            return [f"PARK;{path_text}"]

        if descriptor.kind == "idle":
            # Keep the physical robot still while it idles.
            return ["WAIT;0.5"]

        if descriptor.kind == "turn_center":
            # Use the shared controller command to face the field center.
            return ["TURN_TO_POINT;(0.0,0.0);40"]

        if descriptor.kind == "take_loader":
            # Convert a Loader action into a collision-aware approach path.
            loader_index = self._loader_index_for_action(robot.name, descriptor.action)
            if loader_index is None:
                return ["WAIT;0.5"]
            loader_target, _ = self._loader_wall_pose(robot.name, loader_index)
            state = self.state["agents"].get(robot.name)
            start_position = (
                np.asarray(state["position"], dtype=np.float32)
                if state is not None
                else np.asarray(robot.start_position, dtype=np.float32)
            )
            path, _, _, _ = self.path_planner.Solve(
                start_position,
                loader_target,
                self.get_permanent_obstacles(),
                robot,
                optimize=False,
            )
            path_text = ",".join(
                f"({float(point[0]):.1f}, {float(point[1]):.1f})"
                for point in path
            )
            return [f"LOADER;{path_text}"]

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
                position, numVertices=8, radius=GOAL_STRUCTURE_RADIUS,
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

        for index, position in enumerate(self.toggle_positions):
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

        for index, position in enumerate(self.loader_positions):
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
