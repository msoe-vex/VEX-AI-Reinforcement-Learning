"""VEX AI Competition variant of Override.

The field geometry and game elements are inherited from :mod:`override`.
The explicit name keeps this variant separate from Pushback's VEX AI game.
"""

from typing import Dict, Optional

import numpy as np

from vex_core.config import CommunicationOption
from vex_core.robot import Robot, RobotSize, Team

from .override import (
    Actions,
    GoalType,
    ObjectStatus,
    OverrideGame,
    VEXAI_GOAL_POSITIONS,
    VEXAI_GOAL_TYPES,
    MIRRORED_PIN_POSITIONS,
    LOADER_POSITIONS,
    MIRRORED_CUP_POSITION,
    SPECIAL_CUP_POSITION,
    SPECIAL_PIN_POSITIONS,
    UPPER_RIGHT_CUP_POSITION,
    UPPER_RIGHT_PIN_POSITIONS,
    LOWER_LEFT_CUP_POSITION,
    LOWER_LEFT_PIN_POSITIONS,
)


def remove_field_clusters(
    state: Dict,
    pin_positions: tuple,
    cup_positions: tuple,
) -> Dict:
    """Remove designated source clusters after shared setup is complete."""
    pin_positions = set(pin_positions)
    cup_positions = set(cup_positions)
    state["objects"] = [
        obj for obj in state["objects"]
        if not (
            (obj["kind"] == "pin" and obj.get("source_position") in pin_positions)
            or (obj["kind"] == "cup" and obj.get("source_position") in cup_positions)
        )
    ]
    return state


def configure_vexai_objects(state: Dict, include_blue_preloads: bool) -> Dict:
    """Assign the exact VEX AI inventory across goals, robots, loaders, and field."""
    for obj in state["objects"]:
        obj.update(
            status=ObjectStatus.ON_FIELD,
            held_by=None,
            goal=None,
            goal_stack_index=None,
        )
        for key in ("loader_index", "loader_stack"):
            obj.pop(key, None)

    pins_by_color = {}
    for obj in state["objects"]:
        if obj["kind"] == "pin":
            pins_by_color.setdefault(
                (obj["front_color"], obj["back_color"]), []
            ).append(obj)

    cups = [obj for obj in state["objects"] if obj["kind"] == "cup"]

    cluster_source_positions = {
        tuple(round(float(value), 3) for value in position)
        for position in SPECIAL_PIN_POSITIONS + MIRRORED_PIN_POSITIONS
    }

    def take_pin(
        front: str,
        back: str,
        count: int,
        excluded_positions: set = frozenset(),
    ) -> list:
        candidates = pins_by_color[(front, back)]
        candidates = [
            obj for obj in candidates
            if tuple(round(float(value), 3) for value in obj.get(
                "source_position", obj["position"]
            ))
            not in excluded_positions
        ]
        if len(candidates) < count:
            raise ValueError(
                f"VEX AI layout needs {count} ({front}, {back}) pins, "
                f"but only {len(candidates)} are available"
            )
        selected = candidates[:count]
        selected_ids = {id(obj) for obj in selected}
        pins_by_color[(front, back)] = [
            obj for obj in pins_by_color[(front, back)]
            if id(obj) not in selected_ids
        ]
        return selected

    def take_cups(count: int) -> list:
        if len(cups) < count:
            raise ValueError(
                f"VEX AI layout needs {count} cups, but only {len(cups)} are available"
            )
        selected = cups[:count]
        del cups[:count]
        return selected

    selected = []

    goal_pins = take_pin("yellow", "yellow", 3)
    for goal, obj in zip(
        (GoalType.SHORT_1, GoalType.SHORT_4, GoalType.TALL),
        goal_pins,
    ):
        obj.update(
            status=ObjectStatus.SCORED,
            goal=goal.value,
            goal_stack_index=0,
        )
        selected.append(obj)

    preload_by_team = {
        "red": take_pin("red", "yellow", 2, cluster_source_positions),
        "blue": take_pin("blue", "yellow", 2, cluster_source_positions)
        if include_blue_preloads else [],
    }
    preloads_by_agent = {}
    agents_by_team = {"red": [], "blue": []}
    for agent_name, agent_state in state["agents"].items():
        agents_by_team[agent_state["team"]].append(agent_name)
    for team, team_agents in agents_by_team.items():
        for agent_name, obj in zip(team_agents, preload_by_team[team]):
            obj.update(status=ObjectStatus.HELD, held_by=agent_name)
            preloads_by_agent[agent_name] = obj
            state["agents"][agent_name]["held_pin_order"] = [
                obj["front_color"], obj["back_color"]
            ]
            selected.append(obj)

    loader_plan = (
        (0, "red"), (1, "blue"), (2, "red"), (3, "blue"),
    )
    loader_pins = {
        "red": take_pin("red", "yellow", 10),
        "blue": take_pin("blue", "yellow", 10),
    }
    mirrored_goal_positions = [
        np.array([-position[1], -position[0]], dtype=np.float32)
        for position in VEXAI_GOAL_POSITIONS.values()
    ]
    cluster_cup_positions = [
        np.asarray(SPECIAL_CUP_POSITION, dtype=np.float32),
        np.asarray(MIRRORED_CUP_POSITION, dtype=np.float32),
    ]
    field_cup_positions = cluster_cup_positions + mirrored_goal_positions
    field_cups = take_cups(len(field_cup_positions))
    for cup, position in zip(field_cups, field_cup_positions):
        cup["position"] = position.copy()
    loader_cups = take_cups(20)
    for loader_index, team in loader_plan:
        for stack_index in range(5):
            pin = loader_pins[team].pop(0)
            cup = loader_cups.pop(0)
            for obj in (pin, cup):
                obj.update(
                    status=ObjectStatus.LOADER,
                    loader_index=loader_index,
                    loader_stack=stack_index,
                )
                obj["position"] = LOADER_POSITIONS[loader_index].copy()
                selected.append(obj)

    red_blue_pins = take_pin("red", "blue", 4)
    cluster_positions = SPECIAL_PIN_POSITIONS + MIRRORED_PIN_POSITIONS
    left_of_downsloping_line = [
        position for position in cluster_positions
        if position[0] + position[1] > 0
    ]
    right_of_downsloping_line = [
        position for position in cluster_positions
        if position[0] + position[1] < 0
    ]
    red_cluster_pins = take_pin("red", "yellow", 4)
    blue_cluster_pins = take_pin("blue", "yellow", 4)
    for pin, position in zip(red_cluster_pins, left_of_downsloping_line):
        pin["position"] = np.asarray(position, dtype=np.float32)
    for pin, position in zip(blue_cluster_pins, right_of_downsloping_line):
        pin["position"] = np.asarray(position, dtype=np.float32)
    cluster_pins = red_cluster_pins + blue_cluster_pins

    mirrored_goal_stack = take_pin("yellow", "yellow", 5)
    for pin, position in zip(mirrored_goal_stack, mirrored_goal_positions):
        pin["position"] = position.copy()
        pin["face_up"] = True

    field_pins = (
        take_pin("red", "yellow", 3)
        + take_pin("blue", "yellow", 3)
        + red_blue_pins
        + cluster_pins
        + mirrored_goal_stack
        + take_pin("yellow", "yellow", 3)
    )
    field_cups.extend(take_cups(21))
    selected.extend(field_pins)
    selected.extend(field_cups)

    state["objects"] = selected
    for agent_name, obj in preloads_by_agent.items():
        state["agents"][agent_name]["held_stack"] = [
            index for index, candidate in enumerate(selected) if candidate is obj
        ]
    state["loaders"] = [5] * 4
    state["loader_reserves"] = [5] * 4
    return state


class VexAICompGame(OverrideGame):
    """Two-robot-per-alliance VEX AI Override match."""

    def __init__(
        self,
        robots: Optional[list] = None,
        communication_mode: CommunicationOption = CommunicationOption.NONE,
        deterministic: bool = True,
    ):
        if robots is None:
            robots = [
                Robot("red_robot_0", Team.RED, RobotSize.INCH_24,
                      np.array([-48.0, 24.0], dtype=np.float32)),
                Robot("red_robot_1", Team.RED, RobotSize.INCH_15,
                      np.array([-48.0, -24.0], dtype=np.float32)),
                Robot("blue_robot_0", Team.BLUE, RobotSize.INCH_24,
                      np.array([48.0, 24.0], dtype=np.float32)),
                Robot("blue_robot_1", Team.BLUE, RobotSize.INCH_15,
                      np.array([48.0, -24.0], dtype=np.float32)),
            ]
        super().__init__(
            robots, communication_mode=communication_mode, deterministic=deterministic
        )

    @property
    def total_time(self) -> float:
        return 120.0

    @property
    def goal_types(self) -> tuple:
        return VEXAI_GOAL_TYPES

    @property
    def goal_positions(self) -> Dict:
        return VEXAI_GOAL_POSITIONS

    @property
    def excluded_pin_positions(self) -> tuple:
        return tuple(UPPER_RIGHT_PIN_POSITIONS + LOWER_LEFT_PIN_POSITIONS)

    @property
    def excluded_cup_positions(self) -> tuple:
        return (UPPER_RIGHT_CUP_POSITION, LOWER_LEFT_CUP_POSITION)

    def get_initial_state(self, randomize: bool = False, seed: Optional[int] = None) -> Dict:
        state = super().get_initial_state(randomize, seed)
        return configure_vexai_objects(state, include_blue_preloads=True)

    def is_valid_action(self, agent: str, action: int, observation: np.ndarray) -> bool:
        """Only 24-inch robots may park in their alliance load zone."""
        if not super().is_valid_action(agent, action, observation):
            return False
        if self._decode_action(action) != Actions.PARK_MIDFIELD:
            return True
        return self.state["agents"][agent].get("robot_size") == RobotSize.INCH_24.value

    def compute_score(self) -> Dict[str, int]:
        """Apply the AI parking restriction while retaining Override scoring."""
        scores = super().compute_score()
        for agent in self.state["agents"].values():
            if agent.get("parked_zone") == "midfield" and agent.get("robot_size") != 24:
                scores[agent["team"]] -= 8
        return scores


__all__ = ["VexAICompGame"]
