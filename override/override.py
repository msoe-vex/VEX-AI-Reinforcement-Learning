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


class Actions(Enum):
    PICKUP_PIN = 0
    PICKUP_CUP = 1
    SCORE_PIN = 2
    SCORE_CUP = 3
    #Score held item(s) on goal in front of robot, if allowed by goal rules.
    #Drive to goal 1
    #Drive to goal 2
    #Drive to goal 3
    #ect.
    TOGGLE_QUADRANT = 4
    PARK_MIDFIELD = 5
    TURN_TOWARD_CENTER = 6
    TAKE_FROM_LOADER_TL = 7
    TAKE_FROM_LOADER_TR = 8
    TAKE_FROM_LOADER_BL = 9
    TAKE_FROM_LOADER_BR = 10
    IDLE = 11
    ORIENT_NEXT_PIN = 12
    ORIENT_NEXT_CUP = 13


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
    GoalType.SHORT_1: np.array([48.0, 24.0], dtype=np.float32),
    GoalType.SHORT_2: np.array([24.0, 48.0], dtype=np.float32),
    GoalType.SHORT_3: np.array([-48.0, -24.0], dtype=np.float32),
    GoalType.SHORT_4: np.array([-24.0, -48.0], dtype=np.float32),
    GoalType.TALL: np.array([0.0, 0.0], dtype=np.float32),
    GoalType.RED_1: np.array([-48.0, 24.0], dtype=np.float32),
    GoalType.RED_2: np.array([-24.0, 48.0], dtype=np.float32),
    GoalType.BLUE_1: np.array([48.0, -24.0], dtype=np.float32),
    GoalType.BLUE_2: np.array([24.0, -48.0], dtype=np.float32),
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
    Obstacle(float(position[0]), float(position[1]), 6.0, False)
    for goal_type, position in GOAL_POSITIONS.items()
    if goal_type not in {GoalType.RED_1, GoalType.RED_2, GoalType.BLUE_1, GoalType.BLUE_2}
] + [
    Obstacle(float(position[0]), float(position[1]), 12.0, False)
    for goal_type, position in GOAL_POSITIONS.items()
    if goal_type in {GoalType.RED_1, GoalType.RED_2, GoalType.BLUE_1, GoalType.BLUE_2}
] + [
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
            Robot("red_robot_0", Team.RED, RobotSize.INCH_24,
                np.array([0.0, -FIELD_HALF + 12.0], dtype=np.float32), start_orientation=0.0),
            Robot("red_robot_1", Team.RED, RobotSize.INCH_15,
                np.array([-FIELD_HALF + 7.5, 0.0], dtype=np.float32), start_orientation=90.0),
            Robot("blue_robot_0", Team.BLUE, RobotSize.INCH_24,
                np.array([0.0, FIELD_HALF - 12.0], dtype=np.float32), start_orientation=180.0),
            Robot("blue_robot_1", Team.BLUE, RobotSize.INCH_15,
                np.array([FIELD_HALF - 7.5, 0.0], dtype=np.float32), start_orientation=270.0),
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

    def _object(self, kind: str, position: np.ndarray, team: Optional[str] = None,
                face_up: Optional[bool] = None) -> Dict:
        # Create a field object record for a Pin or Cup.
        return {"kind": kind, "position": np.asarray(position, dtype=np.float32),
                "team": team, "face_up": face_up, "status": ObjectStatus.ON_FIELD,
            "held_by": None, "goal": None, "goal_stack_index": None}

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
                "held_stack": [],
                "parked_zone": None, "toggled": [0] * NUM_TOGGLES,
                "inferred_toggle_colors": [None] * NUM_TOGGLES,
                "next_pin_color": None, "next_cup_face_up": None,
                "agent_name": robot.name, "current_action": None,
            }
        objects = []
        for index in range(NUM_PINS):
            position = PIN_START_POSITIONS[index].copy()
            if randomize:
                position = np.random.uniform(-66, 66, 2).astype(np.float32)
            primary_color, secondary_color = PIN_COLOR_PAIRS[index]
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
            objects.append(pin)
        for index in range(NUM_CUPS):
            position = CUP_START_POSITIONS[index].copy()
            if randomize:
                position = np.random.uniform(-66, 66, 2).astype(np.float32)
            stack_cup = 4 <= index < 36
            objects.append(self._object("cup", position, face_up=True if stack_cup else (index % 2 == 0)))

        preload_candidates = {
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
            team = agent_state["team"]
            preload_index = preload_candidates[team].pop(0)
            objects[preload_index].update(status=ObjectStatus.HELD, held_by=agent_name)
            agent_state["held_stack"] = [preload_index]

        black_goal_types = [GoalType.TALL]
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
            objects[obj_index]["status"] = ObjectStatus.SCORED
            objects[obj_index]["goal"] = goal_type.value
            objects[obj_index]["held_by"] = None
            objects[obj_index]["goal_stack_index"] = 0

        protected_yellow_positions = {
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
            objects.pop(object_index)

        loader_teams = ("red", "blue", "red", "blue")
        loader_pin_candidates = {
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
                  "autonomous_winner": None}
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
        if selected == Actions.ORIENT_NEXT_PIN:
            return [ActionStep(
                DEFAULT_DURATION, state["position"].copy(), state["orientation"].copy(),
                [ActionEvent("orient_next", {"kind": "pin", "color": state["team"]})],
            )], 0.0
        if selected == Actions.ORIENT_NEXT_CUP:
            return [ActionStep(
                DEFAULT_DURATION, state["position"].copy(), state["orientation"].copy(),
                [ActionEvent("orient_next", {"kind": "cup", "face_up": True})],
            )], 0.0
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
            paired = state["held_pins"] > 0 and state["held_cups"] > 0
            scoring_kind = "pin" if paired else kind
            if not self._goal_allows_scoring(goal.value, scoring_kind):
                return [ActionStep(0.1, state["position"].copy(), state["orientation"].copy())], DEFAULT_PENALTY
            return self._move(
                agent, GOAL_POSITIONS[goal],
                ActionEvent("score", {"kind": kind, "goal": goal.value, "paired": paired}),
            ), 0.0
        if selected in (Actions.TAKE_FROM_LOADER_TL, Actions.TAKE_FROM_LOADER_TR,
                        Actions.TAKE_FROM_LOADER_BL, Actions.TAKE_FROM_LOADER_BR):
            loader_index = selected.value - Actions.TAKE_FROM_LOADER_TL.value
            loader_count = self.state["loaders"][loader_index]
            if loader_count <= 0 or state["held_cups"] >= MAX_HELD_CUPS:
                return [ActionStep(0.1, state["position"].copy(), state["orientation"].copy())], DEFAULT_PENALTY
            loader_position = LOADER_POSITIONS[loader_index]
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

    def _goal_allows_scoring(self, goal_value: str, kind: str) -> bool:
        # Goals must begin with a pin and then alternate cup/pin for each additional score.
        scored_objects = [
            obj for obj in self.state["objects"]
            if obj.get("goal") == goal_value and obj["status"] == ObjectStatus.SCORED
        ]
        if not scored_objects:
            return kind == "pin"
        last_kind = scored_objects[-1]["kind"]
        return last_kind != kind

    def apply_events(self, agent: str, events: List[ActionEvent]) -> None:
        # Apply completed action events to objects, robots, Toggles, and Loaders.
        state = self.state["agents"][agent]
        for event in events:
            if event.type == "pickup":
                obj = self.state["objects"][event.data["index"]]
                held_key = f"held_{obj['kind']}s"
                capacity = MAX_HELD_PINS if obj["kind"] == "pin" else MAX_HELD_CUPS
                if obj["status"] == ObjectStatus.ON_FIELD and state[held_key] < capacity:
                    if obj["kind"] == "pin":
                        desired_color = state.get("next_pin_color")
                        if desired_color == obj.get("front_color"):
                            obj["face_up"] = True
                        elif desired_color == obj.get("back_color"):
                            obj["face_up"] = False
                        state["next_pin_color"] = None
                    else:
                        desired_face_up = state.get("next_cup_face_up")
                        if desired_face_up is not None:
                            obj["face_up"] = bool(desired_face_up)
                        state["next_cup_face_up"] = None
                    obj.update(status=ObjectStatus.HELD, held_by=agent)
                    state[held_key] += 1
                    held_indices = [
                        index for index, held_obj in enumerate(self.state["objects"])
                        if held_obj["status"] == ObjectStatus.HELD and held_obj["held_by"] == agent
                    ]
                    for partner in self.state["objects"]:
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
                    held_indices = [
                        index for index, held_obj in enumerate(self.state["objects"])
                        if held_obj["status"] == ObjectStatus.HELD and held_obj["held_by"] == agent
                    ]
                    state["held_stack"] = sorted(
                        held_indices,
                        key=lambda index: 0 if self.state["objects"][index]["kind"] == "cup" else 1,
                    )
            elif event.type == "score":
                kind = event.data["kind"]
                goal_value = event.data["goal"]
                if not self._goal_allows_scoring(goal_value, kind):
                    state[f"held_{kind}s"] = max(0, state[f"held_{kind}s"])
                    continue
                scored_kinds = {kind}
                if event.data.get("paired") and state["held_pins"] > 0 and state["held_cups"] > 0:
                    scored_kinds = {"pin", "cup"}
                next_stack_index = sum(
                    1 for obj in self.state["objects"]
                    if obj["status"] == ObjectStatus.SCORED
                    and obj.get("goal") == goal_value
                )
                for obj in self.state["objects"]:
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
                state["held_stack"] = [
                    index for index in state.get("held_stack", [])
                    if self.state["objects"][index]["status"] == ObjectStatus.HELD
                ]
            elif event.type == "orient_next":
                if event.data["kind"] == "pin":
                    state["next_pin_color"] = event.data["color"]
                else:
                    state["next_cup_face_up"] = bool(event.data["face_up"])
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
                    reserve_count = self.state.get("loader_reserves", [0] * NUM_TOGGLES)[loader_index]
                    if reserve_count > 0:
                        self.state["loaders"][loader_index] = reserve_count
                        self.state["loader_reserves"][loader_index] = 0
            elif event.type == "park":
                state.update(parked=True, parked_zone="midfield")
            elif event.type == "turn":
                state["orientation"] = np.array([event.data["angle"]], dtype=np.float32)

    def _score_pin_color(self, pin: Dict, goal_value: str) -> Optional[Tuple[str, int]]:
        """Return the alliance and value of the pin half facing away from its goal."""
        if not pin.get("face_up", True):
            return None
        color = pin.get("front_color")
        if color not in {"red", "blue", "yellow"}:
            return None
        if color == "yellow":
            goal_color = goal_value.split("_", 1)[0]
            if goal_color not in {"red", "blue"}:
                return None
            return goal_color, 10
        return color, 5

    def _score_goal_pins(self) -> Dict[str, int]:
        """Score exposed pin halves, walking each stack outward from its goal."""
        scores = {"red": 0, "blue": 0}
        goal_values = {goal.value for goal in GoalType}
        for goal_value in goal_values:
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
                    continue
                if index + 1 < len(stack):
                    cover = stack[index + 1]
                    if cover["kind"] == "cup" and cover.get("face_up", True):
                        continue
                scored = self._score_pin_color(obj, goal_value)
                if scored is not None:
                    alliance, value = scored
                    scores[alliance] += value
        return scores

    def compute_score(self) -> Dict[str, int]:
        # Calculate alliance scores from exposed scored pin halves, parking, and bonuses.
        scores = {"red": 0, "blue": 0}
        pin_scores = self._score_goal_pins()
        for alliance, value in pin_scores.items():
            scores[alliance] += value
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

    def render_field_markings(self, ax: Any) -> None:
        # Render the square Midfield, diagonal Autonomous Lines, and Load Zones.
        import matplotlib.patches as patches

        field_half = self.field_size_inches / 2
        ax.set_facecolor("#d7d7d7")
        ax.add_patch(patches.Rectangle(
            (-field_half, -field_half), self.field_size_inches, self.field_size_inches,
            fill=False, edgecolor="black", linewidth=1.5,
        ))

        midfield_half = MIDFIELD_SIZE_INCHES / 2
        ax.add_patch(patches.Rectangle(
            (-midfield_half, -midfield_half), MIDFIELD_SIZE_INCHES, MIDFIELD_SIZE_INCHES,
            fill=False, edgecolor="white", linewidth=2.5, angle=45,
            rotation_point="center", zorder=1,
        ))

        square_side_midpoint = midfield_half / np.sqrt(2.0)
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
        return FOV

    def camera_range_inches(self) -> float:
        return 72.0

    def split_action(self, action: int, observation: np.ndarray, robot: Robot) -> List[str]:
        # Convert a high-level action into controller command strings.
        if action == Actions.IDLE.value:
            return ["WAIT;0.5"]
        if action == Actions.TURN_TOWARD_CENTER.value:
            return ["TURN_TO_POINT;(0.0,0.0);40"]
        if Actions.TAKE_FROM_LOADER_TL.value <= action <= Actions.TAKE_FROM_LOADER_BR.value:
            loader_index = action - Actions.TAKE_FROM_LOADER_TL.value
            loader_position = LOADER_POSITIONS[loader_index]
            return [
                f"FOLLOW;({loader_position[0]:.1f}, {loader_position[1]:.1f});50",
                "CLEAR_LOADER",
            ]
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
        goal_label_order = [
            GoalType.SHORT_1, GoalType.SHORT_2, GoalType.SHORT_3, GoalType.SHORT_4,
            GoalType.TALL,
            GoalType.RED_1, GoalType.RED_2,
            GoalType.BLUE_1, GoalType.BLUE_2,
        ]
        for goal_index, goal_type in enumerate(goal_label_order, start=1):
            position = GOAL_POSITIONS[goal_type]
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
        for obj in self.state["objects"]:
            if obj["status"] == ObjectStatus.SCORED and obj["kind"] == "pin" and obj.get("goal"):
                scored_pins_by_goal.setdefault(obj["goal"], []).append(obj)

        for goal_type, position in GOAL_POSITIONS.items():
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
        for obj in self.state["objects"]:
            if obj["status"] == ObjectStatus.ON_FIELD:
                if obj["kind"] == "cup":
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
                        obj["position"], cup_radius, 0.0, 180.0,
                        facecolor=upper_color, edgecolor="black", linewidth=0.7, zorder=6,
                    ))
                    ax.add_patch(patches.Wedge(
                        obj["position"], cup_radius, 180.0, 360.0,
                        facecolor=lower_color, edgecolor="black", linewidth=0.7, zorder=6,
                    ))
                    ax.add_patch(patches.Circle(
                        obj["position"], cup_radius, fill=False,
                        edgecolor="black", linewidth=0.8, zorder=7,
                    ))
                    if has_stacked_pin:
                        ax.add_patch(patches.Circle(
                            obj["position"], cup_radius + 0.6, fill=False,
                            edgecolor="#f4df00", linewidth=1.0, zorder=5,
                        ))
                else:
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
                        obj["position"], pin_radius, 0.0, 180.0,
                        facecolor=upper_color, edgecolor="black", linewidth=0.7, zorder=6,
                    ))
                    ax.add_patch(patches.Wedge(
                        obj["position"], pin_radius, 180.0, 360.0,
                        facecolor=lower_color, edgecolor="black", linewidth=0.7, zorder=6,
                    ))
                    ax.add_patch(patches.Circle(
                        obj["position"], pin_radius, fill=False,
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
        for agent in agents or self.state["agents"]:
            state = self.state["agents"][agent]
            ax_info.text(0.05, y, f"{agent}: {state['team']} P{state['held_pins']} C{state['held_cups']}", va="top")
            y -= 0.06
        ax_info.text(0.05, y, str(self.compute_score()), va="top")

        ax_info.text(0.45, 0.16, "Goals", fontsize=10, fontweight="bold", va="bottom")
        goal_order = [
            GoalType.SHORT_1, GoalType.SHORT_2, GoalType.SHORT_3, GoalType.SHORT_4,
            GoalType.TALL,
            GoalType.RED_1, GoalType.RED_2,
            GoalType.BLUE_1, GoalType.BLUE_2,
        ]
        goal_counts = {goal.value: 0 for goal in goal_order}
        goal_colors_by_goal = {goal.value: [] for goal in goal_order}
        for obj in self.state["objects"]:
            if obj["status"] == ObjectStatus.SCORED and obj.get("goal"):
                goal_value = obj["goal"]
                if obj["kind"] == "pin":
                    goal_counts[goal_value] = goal_counts.get(goal_value, 0) + 1
                    goal_colors_by_goal.setdefault(goal_value, []).append((obj.get("front_color") or obj.get("back_color") or "yellow"))

        goal_base_x = 0.08
        goal_y = 0.02
        goal_spacing = 0.09
        for index, goal_type in enumerate(goal_order, start=1):
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
