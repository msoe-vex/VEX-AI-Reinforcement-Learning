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
    CENTER_STACK_CUP_POSITIONS,
    CENTER_STACK_PIN_POSITIONS,
    TOGGLE_STACK_CUP_POSITIONS,
    TOGGLE_STACK_PIN_POSITIONS,
    YELLOW_STACK_CUP_POSITIONS,
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


def configure_vexai_objects(
    state: Dict,
    include_blue_preloads: bool,
    randomize: bool = False,
) -> Dict:
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
        from_end: bool = False,
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
        selected = candidates[-count:] if from_end else candidates[:count]
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

    def take_cups_at(positions: list) -> list:
        position_keys = {
            tuple(round(float(value), 3) for value in position)
            for position in positions
        }
        selected = [
            cup for cup in cups
            if tuple(round(float(value), 3) for value in cup["source_position"])
            in position_keys
        ]
        if len(selected) != len(positions):
            raise ValueError(
                f"VEX AI layout needs cups at {positions}, "
                f"but only {len(selected)} are available"
            )
        selected_ids = {id(cup) for cup in selected}
        cups[:] = [cup for cup in cups if id(cup) not in selected_ids]
        return selected

    def take_pins_at(front: str, back: str, positions: list) -> list:
        position_keys = {
            tuple(round(float(value), 3) for value in position)
            for position in positions
        }
        candidates = [
            pin for pin in pins_by_color[(front, back)]
            if tuple(round(float(value), 3) for value in pin.get(
                "source_position", pin["position"]
            ))
            in position_keys
        ]
        if len(candidates) != len(positions):
            raise ValueError(
                f"VEX AI layout needs {front}/{back} pins at {positions}, "
                f"but only {len(candidates)} are available"
            )
        selected_ids = {id(pin) for pin in candidates}
        pins_by_color[(front, back)] = [
            pin for pin in pins_by_color[(front, back)]
            if id(pin) not in selected_ids
        ]
        return candidates

    selected = []

    toggle_pins = take_pins_at("yellow", "yellow", TOGGLE_STACK_PIN_POSITIONS)
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
        "blue": take_pin("blue", "yellow", 2, cluster_source_positions, from_end=True)
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
    toggle_cups = take_cups_at([
        np.asarray(position, dtype=np.float32)
        for position in TOGGLE_STACK_CUP_POSITIONS
    ])
    center_cups = take_cups_at([
        np.asarray(position, dtype=np.float32)
        for position in CENTER_STACK_CUP_POSITIONS
    ])
    diagonal_cups = take_cups_at([
        np.asarray(position, dtype=np.float32)
        for position in YELLOW_STACK_CUP_POSITIONS[:2]
    ])
    field_cup_positions = (
        cluster_cup_positions
        + [
            position for position in mirrored_goal_positions
            if not np.allclose(position, np.zeros(2, dtype=np.float32))
        ]
    )
    field_cups = take_cups(len(field_cup_positions))
    for cup, position in zip(field_cups, field_cup_positions):
        cup["position"] = position.copy()
    field_cups.extend(diagonal_cups)
    field_cups.extend(toggle_cups)
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
    for pin, position in zip(red_blue_pins, CENTER_STACK_PIN_POSITIONS):
        pin["position"] = np.asarray(position, dtype=np.float32)
    diagonal_pins = [
        take_pin("red", "yellow", 1)[0],
        take_pin("blue", "yellow", 1)[0],
    ]
    diagonal_pins[0].update(
        position=np.asarray((-24.0, -24.0), dtype=np.float32),
        face_up=False,
        team="yellow",
        front_color="yellow",
        back_color="red",
    )
    diagonal_pins[1].update(
        position=np.asarray((24.0, 24.0), dtype=np.float32),
        face_up=False,
        team="yellow",
        front_color="yellow",
        back_color="blue",
    )
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
    center_pin = next(
        pin for pin in mirrored_goal_stack
        if np.allclose(pin["position"], np.zeros(2, dtype=np.float32))
    )
    previous_tall_pin = goal_pins[2]
    if previous_tall_pin is not center_pin:
        previous_tall_pin.update(
            status=ObjectStatus.ON_FIELD,
            goal=None,
            goal_stack_index=None,
        )
        selected[:] = [
            obj for obj in selected
            if obj is not previous_tall_pin
        ]
        center_pin.update(
            status=ObjectStatus.SCORED,
            goal=GoalType.TALL.value,
            goal_stack_index=0,
        )

    field_pins = (
        red_blue_pins
        + diagonal_pins
        + cluster_pins
        + mirrored_goal_stack
        + toggle_pins
    )
    selected.extend(field_pins)
    selected.extend(field_cups)
    selected.extend(center_cups)

    state["objects"] = selected
    for agent_name, obj in preloads_by_agent.items():
        state["agents"][agent_name]["held_stack"] = [
            index for index, candidate in enumerate(selected) if candidate is obj
        ]
    state["loaders"] = [5] * 4
    state["loader_reserves"] = [5] * 4
    if randomize:
        for obj in selected:
            if obj["status"] == ObjectStatus.ON_FIELD:
                obj["position"] = np.random.uniform(-66, 66, 2).astype(np.float32)
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
                Robot("red_robot_0", Team.RED, RobotSize.INCH_15,
                      np.array([0.0, -64.5], dtype=np.float32), start_orientation=0.0),
                Robot("red_robot_1", Team.RED, RobotSize.INCH_15,
                      np.array([-64.5, 0.0], dtype=np.float32), start_orientation=90.0),
                Robot("blue_robot_0", Team.BLUE, RobotSize.INCH_15,
                      np.array([0.0, 64.5], dtype=np.float32), start_orientation=180.0),
                Robot("blue_robot_1", Team.BLUE, RobotSize.INCH_15,
                      np.array([64.5, 0.0], dtype=np.float32), start_orientation=270.0),
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
        state = super().get_initial_state(False, seed)
        return configure_vexai_objects(
            state,
            include_blue_preloads=True,
            randomize=randomize,
        )

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
