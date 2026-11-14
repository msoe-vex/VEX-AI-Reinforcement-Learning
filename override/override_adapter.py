"""
Override Game Adapter
Concrete implementation of BaseGameAdapter for the 2026-2027 VEX V5 Override game.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from vex_core.game_adapter import BaseGameAdapter
from vex_core.factory import GameAdapterFactory
from vex_core.converters import (
    json_to_numpy,
    numpy_to_json,
    get_default_observation_dict,
    OBS_TOTAL,
    NUM_TOGGLES,
    NUM_GOALS,
)
from override.override import (
    Actions,
    GoalType,
    GOAL_POSITIONS,
    TOGGLE_POSITIONS,
    LOADER_POSITIONS,
    GOAL_SCORE_ACTIONS,
    TOGGLE_ACTIONS,
    LOADER_ACTIONS,
    MAX_HELD_PINS,
    MAX_HELD_CUPS,
)


class OverrideGameAdapter(BaseGameAdapter):
    """
    Adapter bridging physical runtime state with Override reinforcement learning models.
    """

    GOAL_ACTION_TO_POS = {
        Actions.SCORE_GOAL_1.value: GOAL_POSITIONS[GoalType.SHORT_1],
        Actions.SCORE_GOAL_2.value: GOAL_POSITIONS[GoalType.SHORT_2],
        Actions.SCORE_GOAL_3.value: GOAL_POSITIONS[GoalType.SHORT_3],
        Actions.SCORE_GOAL_4.value: GOAL_POSITIONS[GoalType.SHORT_4],
        Actions.SCORE_GOAL_5.value: GOAL_POSITIONS[GoalType.TALL],
        Actions.SCORE_GOAL_6.value: GOAL_POSITIONS[GoalType.RED_1],
        Actions.SCORE_GOAL_7.value: GOAL_POSITIONS[GoalType.RED_2],
        Actions.SCORE_GOAL_8.value: GOAL_POSITIONS[GoalType.BLUE_1],
        Actions.SCORE_GOAL_9.value: GOAL_POSITIONS[GoalType.BLUE_2],
    }

    TOGGLE_ACTION_TO_POS = {
        Actions.FLIP_TOGGLE_1.value: TOGGLE_POSITIONS[0],
        Actions.FLIP_TOGGLE_2.value: TOGGLE_POSITIONS[1],
        Actions.FLIP_TOGGLE_3.value: TOGGLE_POSITIONS[2],
        Actions.FLIP_TOGGLE_4.value: TOGGLE_POSITIONS[3],
    }

    LOADER_ACTION_TO_POS = {
        Actions.TAKE_FROM_LOADER_1.value: LOADER_POSITIONS[0],
        Actions.TAKE_FROM_LOADER_2.value: LOADER_POSITIONS[1],
    }

    def __init__(self, game: Optional[Any] = None, agent_name: str = "robot_0", team: str = "red"):
        super().__init__(game=game, agent_name=agent_name, team=team)

    @property
    def game_name(self) -> str:
        return "override"

    @property
    def observation_schema(self) -> Dict[str, Any]:
        """JSON Schema dictionary defining the Override observation structure."""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "OverrideObservation",
            "type": "object",
            "required": [
                "match_time_remaining",
                "self_robot",
                "teammate_robot",
                "pins",
                "cups",
                "toggles",
                "loaders",
                "loader_owners",
                "stacks",
                "goals",
            ],
            "properties": {
                "match_time_remaining": {"type": "number", "minimum": 0.0, "maximum": 120.0},
                "self_robot": {
                    "type": "object",
                    "required": ["x", "y", "theta", "linear_speed", "held_pins", "held_cups", "parked"],
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "theta": {"type": "number"},
                        "linear_speed": {"type": "number"},
                        "held_pins": {"type": "integer", "minimum": 0, "maximum": 1},
                        "held_cups": {"type": "integer", "minimum": 0, "maximum": 1},
                        "held_pin_front_color": {"type": "string", "enum": ["none", "friendly", "opponent", "neutral"]},
                        "held_pin_back_color": {"type": "string", "enum": ["none", "friendly", "opponent", "neutral"]},
                        "held_cup_face_up": {"type": "boolean"},
                        "parked": {"type": "boolean"},
                    },
                },
                "teammate_robot": {
                    "type": "object",
                    "required": ["distance", "sin_theta", "cos_theta", "visible"],
                },
                "pins": {"type": "array"},
                "cups": {"type": "array"},
                "toggles": {"type": "array", "items": {"type": "integer"}},
                "loaders": {"type": "array", "items": {"type": "integer"}},
                "loader_owners": {"type": "array", "items": {"type": "integer"}},
                "stacks": {"type": "array"},
                "goals": {"type": "array"},
            },
        }

    # =========================================================================
    # Serialization & State Conversion
    # =========================================================================

    def observation_dict_to_numpy(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        return json_to_numpy(obs_dict)

    def numpy_to_observation_dict(self, obs_array: np.ndarray) -> Dict[str, Any]:
        return numpy_to_json(obs_array)

    def get_initial_state_dict(self) -> Dict[str, Any]:
        return get_default_observation_dict()

    # =========================================================================
    # Action Decomposition & Translation
    # =========================================================================

    def decompose_action(
        self,
        action_idx: int,
        current_state: Dict[str, Any],
    ) -> List[str]:
        """
        Decomposes high-level discrete actions into executable V5 sub-actions.
        """
        self_bot = current_state.get("self_robot", {})
        cx = float(self_bot.get("x", 0.0))
        cy = float(self_bot.get("y", 0.0))

        # 1. Scoring Actions (Goal 1 to 9)
        if action_idx in self.GOAL_ACTION_TO_POS:
            pos = self.GOAL_ACTION_TO_POS[action_idx]
            gx, gy = float(pos[0]), float(pos[1])
            # Determine goal height
            goal_idx = list(self.GOAL_ACTION_TO_POS.keys()).index(action_idx)
            goals = current_state.get("goals", [])
            height = float(goals[goal_idx].get("stack_height", 0.0)) if goal_idx < len(goals) else 0.0
            return [
                f"TURN_TO_POINT;({gx:.1f},{gy:.1f});80",
                f"FOLLOW;({gx:.1f},{gy:.1f});90;false",
                f"SCORE;({gx:.1f},{gy:.1f});{height:.1f}",
            ]

        # 2. Toggle Actions (1 to 4)
        if action_idx in self.TOGGLE_ACTION_TO_POS:
            pos = self.TOGGLE_ACTION_TO_POS[action_idx]
            tx, ty = float(pos[0]), float(pos[1])
            return [
                f"TURN_TO_POINT;({tx:.1f},{ty:.1f});80",
                f"FOLLOW;({tx:.1f},{ty:.1f});90;false",
                f"TOGGLE;({tx:.1f},{ty:.1f});1.0",
            ]

        # 3. Loader Actions
        if action_idx in self.LOADER_ACTION_TO_POS:
            pos = self.LOADER_ACTION_TO_POS[action_idx]
            lx, ly = float(pos[0]), float(pos[1])
            return [
                f"TURN_TO_POINT;({lx:.1f},{ly:.1f});80",
                f"FOLLOW;({lx:.1f},{ly:.1f});90;false",
                f"LOADER;({lx:.1f},{ly:.1f});false",
            ]

        # 4. Pickup Pin (Red/Yellow or generic)
        if action_idx in (
            Actions.PICKUP_PIN_RED_YELLOW.value,
            Actions.PICKUP_PIN_BLUE_YELLOW.value,
            Actions.PICKUP_PIN_RED_BLUE.value,
            Actions.PICKUP_PIN_YELLOW_YELLOW.value,
        ):
            pins = current_state.get("pins", [])
            target_x, target_y = cx, cy
            for p in pins:
                if p.get("visible", False) and float(p.get("distance", -144.0)) > 0:
                    dist = float(p.get("distance", 0.0))
                    sin_t = float(p.get("sin_theta", 0.0))
                    cos_t = float(p.get("cos_theta", 1.0))
                    # Estimate coordinate
                    target_x = cx + dist * sin_t
                    target_y = cy + dist * cos_t
                    break
            return [
                f"TURN_TO_POINT;({target_x:.1f},{target_y:.1f});80",
                f"FOLLOW;({target_x:.1f},{target_y:.1f});90;false",
                f"PICKUP_ITEM;({target_x:.1f},{target_y:.1f});false",
            ]

        # 5. Pickup Cup
        if action_idx == Actions.PICKUP_CUP.value:
            cups = current_state.get("cups", [])
            target_x, target_y = cx, cy
            for c in cups:
                if c.get("visible", False) and float(c.get("distance", -144.0)) > 0:
                    dist = float(c.get("distance", 0.0))
                    sin_t = float(c.get("sin_theta", 0.0))
                    cos_t = float(c.get("cos_theta", 1.0))
                    target_x = cx + dist * sin_t
                    target_y = cy + dist * cos_t
                    break
            return [
                f"TURN_TO_POINT;({target_x:.1f},{target_y:.1f});80",
                f"FOLLOW;({target_x:.1f},{target_y:.1f});90;false",
                f"PICKUP_ITEM;({target_x:.1f},{target_y:.1f});false",
            ]

        # 6. Park Midfield
        if action_idx == Actions.PARK_MIDFIELD.value:
            return [
                "TURN_TO_POINT;(0.0,0.0);80",
                "FOLLOW;(0.0,0.0);90;false",
                "PARK;(0.0,0.0)",
            ]

        # 7. Turn toward center
        if action_idx == Actions.TURN_TOWARD_CENTER.value:
            return ["TURN_TO_POINT;(0.0,0.0);80"]

        # 8. Idle / Default
        return ["WAIT;0.5"]

    # =========================================================================
    # State Transition Handlers
    # =========================================================================

    def on_action_success(self, action_idx: int, state: Dict[str, Any]) -> None:
        """
        Updates state dictionary on successful action execution.
        """
        self_bot = state.setdefault("self_robot", {})

        # Leaving parking zone on non-parking actions
        if action_idx not in (Actions.PARK_MIDFIELD.value, Actions.IDLE.value):
            self_bot["parked"] = False

        # Scoring clears held items
        if action_idx in self.GOAL_ACTION_TO_POS:
            self_bot["held_pins"] = 0
            self_bot["held_cups"] = 0
            self_bot["held_pin_front_color"] = "none"
            self_bot["held_pin_back_color"] = "none"
            self_bot["held_cup_face_up"] = False

        # Pickup Pin
        elif action_idx in (
            Actions.PICKUP_PIN_RED_YELLOW.value,
            Actions.PICKUP_PIN_BLUE_YELLOW.value,
            Actions.PICKUP_PIN_RED_BLUE.value,
            Actions.PICKUP_PIN_YELLOW_YELLOW.value,
        ):
            self_bot["held_pins"] = min(MAX_HELD_PINS, self_bot.get("held_pins", 0) + 1)
            self_bot["held_pin_front_color"] = "friendly"

        # Pickup Cup
        elif action_idx == Actions.PICKUP_CUP.value:
            self_bot["held_cups"] = min(MAX_HELD_CUPS, self_bot.get("held_cups", 0) + 1)
            self_bot["held_cup_face_up"] = True

        # Park
        elif action_idx == Actions.PARK_MIDFIELD.value:
            self_bot["parked"] = True

        # Toggle flip
        elif action_idx in self.TOGGLE_ACTION_TO_POS:
            tog_idx = list(self.TOGGLE_ACTION_TO_POS.keys()).index(action_idx)
            toggles = state.setdefault("toggles", [0, 0, 0, 0])
            if tog_idx < len(toggles):
                toggles[tog_idx] = 1  # 1 for friendly

        # Loader
        elif action_idx in self.LOADER_ACTION_TO_POS:
            ldr_idx = list(self.LOADER_ACTION_TO_POS.keys()).index(action_idx)
            loaders = state.setdefault("loaders", [0, 0, 0, 0])
            if ldr_idx < len(loaders) and loaders[ldr_idx] > 0:
                loaders[ldr_idx] -= 1
            self_bot["held_pins"] = min(MAX_HELD_PINS, self_bot.get("held_pins", 0) + 1)
            self_bot["held_cups"] = min(MAX_HELD_CUPS, self_bot.get("held_cups", 0) + 1)

    def on_action_error(self, action_idx: int, error_code: int, state: Dict[str, Any]) -> None:
        """
        Handles action failure by recording error state.
        """
        state["last_error"] = {
            "action_idx": action_idx,
            "error_code": error_code,
        }

    # =========================================================================
    # Intra-Option Step Verification
    # =========================================================================

    def validate_subaction_precondition(
        self,
        action_idx: int,
        subaction_index: int,
        subaction_cmd: str,
        current_state: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        self_bot = current_state.get("self_robot", {})

        # Scoring requires at least one held item before the SCORE command
        if subaction_cmd.startswith("SCORE"):
            held_pins = int(self_bot.get("held_pins", 0))
            held_cups = int(self_bot.get("held_cups", 0))
            if held_pins <= 0 and held_cups <= 0:
                return False, "NoElementsHeldForScoring"

        # Pickup item requires available inventory capacity
        if subaction_cmd.startswith("PICKUP_ITEM"):
            held_pins = int(self_bot.get("held_pins", 0))
            held_cups = int(self_bot.get("held_cups", 0))
            if held_pins >= MAX_HELD_PINS and held_cups >= MAX_HELD_CUPS:
                return False, "InventoryCapacityFull"

        return True, None


# Automatically register Override with the factory
GameAdapterFactory.register("override", OverrideGameAdapter)
