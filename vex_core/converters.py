"""
Bidirectional Observation Converters for VEX AI
Converts between structured JSON observation dicts and 1D NumPy policy vectors.
"""

from typing import Dict, Any, List, Optional
import numpy as np

# Offsets matching override.override.ObsIndex
COLOR_TO_VAL = {"none": 0.0, "friendly": 1.0, "opponent": -1.0, "neutral": 0.5}
VAL_TO_COLOR = {0.0: "none", 1.0: "friendly", -1.0: "opponent", 0.5: "neutral"}

TOP_PIECE_TO_INDEX = {
    "empty": 0,
    "black_cup": 1,
    "clear_cup": 2,
    "friendly_pin": 3,
    "opponent_pin": 4,
    "neutral_pin": 5,
}
INDEX_TO_TOP_PIECE = {v: k for k, v in TOP_PIECE_TO_INDEX.items()}

# Dimension constants from override.override
MAX_OBSERVED_PINS = 63
NUM_CUPS = 48
NUM_TOGGLES = 4
NUM_GOALS = 9
MAX_NEAREST_STACKS = 3
PIN_OBJECT_FEATURES = 5
CUP_OBJECT_FEATURES = 4
ROBOT_POSITION_FEATURES = 3
STACK_OBJECT_FEATURES = 7
GOAL_FEATURES = 13
TOTAL_OBS_DIM = 670

# Offsets
OFFSET_SELF_ORIENT = 0
OFFSET_HELD_PINS = 1
OFFSET_HELD_CUPS = 2
OFFSET_PARKED = 3
OFFSET_TIME_REMAINING = 4
OFFSET_PIN_COUNT = 5
OFFSET_CUP_COUNT = 6
OFFSET_PIN_POSITIONS = 7
OFFSET_CUP_POSITIONS = OFFSET_PIN_POSITIONS + MAX_OBSERVED_PINS * PIN_OBJECT_FEATURES  # 322
OFFSET_TOGGLES = OFFSET_CUP_POSITIONS + NUM_CUPS * CUP_OBJECT_FEATURES  # 514
OFFSET_LOADERS = OFFSET_TOGGLES + NUM_TOGGLES  # 518
OFFSET_LOADER_OWNERS = OFFSET_LOADERS + NUM_TOGGLES  # 522
OFFSET_HELD_PIN_FRONT_COLOR = OFFSET_LOADER_OWNERS + NUM_TOGGLES  # 526
OFFSET_HELD_PIN_BACK_COLOR = OFFSET_HELD_PIN_FRONT_COLOR + 1  # 527
OFFSET_HELD_CUP_FACE_UP = OFFSET_HELD_PIN_BACK_COLOR + 1  # 528
OFFSET_TEAMMATE_POSITION = OFFSET_HELD_CUP_FACE_UP + 1  # 529
OFFSET_STACKS = OFFSET_TEAMMATE_POSITION + ROBOT_POSITION_FEATURES  # 532
OFFSET_GOALS = OFFSET_STACKS + MAX_NEAREST_STACKS * STACK_OBJECT_FEATURES  # 553
OBS_TOTAL = OFFSET_GOALS + NUM_GOALS * GOAL_FEATURES  # 670


def get_default_observation_dict() -> Dict[str, Any]:
    """Returns a schema-compliant initial observation dictionary."""
    return {
        "match_time_remaining": 120.0,
        "self_robot": {
            "x": 0.0,
            "y": 0.0,
            "theta": 0.0,
            "linear_speed": 0.0,
            "held_pins": 0,
            "held_cups": 0,
            "held_pin_front_color": "none",
            "held_pin_back_color": "none",
            "held_cup_face_up": False,
            "parked": False,
        },
        "teammate_robot": {
            "connected": False,
            "distance": -144.0,
            "sin_theta": 0.0,
            "cos_theta": 0.0,
            "visible": False,
        },
        "pins": [],
        "cups": [],
        "toggles": [0, 0, 0, 0],
        "loaders": [0, 0, 0, 0],
        "loader_owners": [1, -1, 1, -1],
        "stacks": [],
        "goals": [
            {
                "distance": -144.0,
                "sin_theta": 0.0,
                "cos_theta": 0.0,
                "friendly_pins": 0,
                "opponent_pins": 0,
                "neutral_pins": 0,
                "top_piece": "empty",
                "stack_height": 0.0,
            }
            for _ in range(NUM_GOALS)
        ],
    }


def json_to_numpy(data: Dict[str, Any], normalize: bool = False) -> np.ndarray:
    """
    Converts a structured observation dictionary to a flat 1D numpy vector (length 670).
    If normalize is True, scales angle to [0, 1] and time to [0, 1].
    """
    obs = np.zeros(OBS_TOTAL, dtype=np.float32)

    # 1. Self Robot Features
    self_bot = data.get("self_robot", {})
    theta = float(self_bot.get("theta", 0.0)) % 360.0
    obs[OFFSET_SELF_ORIENT] = (theta / 360.0) if normalize else theta
    obs[OFFSET_HELD_PINS] = float(self_bot.get("held_pins", 0))
    obs[OFFSET_HELD_CUPS] = float(self_bot.get("held_cups", 0))
    obs[OFFSET_PARKED] = 1.0 if self_bot.get("parked", False) else 0.0

    match_time = float(data.get("match_time_remaining", 120.0))
    obs[OFFSET_TIME_REMAINING] = (match_time / 120.0) if normalize else match_time

    pins_list = data.get("pins", [])
    cups_list = data.get("cups", [])
    obs[OFFSET_PIN_COUNT] = float(len([p for p in pins_list if p.get("visible", False)]))
    obs[OFFSET_CUP_COUNT] = float(len([c for c in cups_list if c.get("visible", False)]))

    # 2. Pins (63 slots, 5 features each: dist, sin, cos, vis, color)
    for idx in range(MAX_OBSERVED_PINS):
        base = OFFSET_PIN_POSITIONS + idx * PIN_OBJECT_FEATURES
        if idx < len(pins_list):
            p = pins_list[idx]
            obs[base + 0] = float(p.get("distance", -144.0))
            obs[base + 1] = float(p.get("sin_theta", 0.0))
            obs[base + 2] = float(p.get("cos_theta", 0.0))
            obs[base + 3] = 1.0 if p.get("visible", False) else 0.0
            obs[base + 4] = COLOR_TO_VAL.get(p.get("color_alliance", "none"), 0.0)
        else:
            obs[base + 0] = -144.0
            obs[base + 1] = 0.0
            obs[base + 2] = 0.0
            obs[base + 3] = 0.0
            obs[base + 4] = 0.0

    # 3. Cups (48 slots, 4 features each: dist, sin, cos, vis)
    for idx in range(NUM_CUPS):
        base = OFFSET_CUP_POSITIONS + idx * CUP_OBJECT_FEATURES
        if idx < len(cups_list):
            c = cups_list[idx]
            obs[base + 0] = float(c.get("distance", -144.0))
            obs[base + 1] = float(c.get("sin_theta", 0.0))
            obs[base + 2] = float(c.get("cos_theta", 0.0))
            obs[base + 3] = 1.0 if c.get("visible", False) else 0.0
        else:
            obs[base + 0] = -144.0
            obs[base + 1] = 0.0
            obs[base + 2] = 0.0
            obs[base + 3] = 0.0

    # 4. Toggles, Loaders, Loader Owners
    toggles = data.get("toggles", [])
    for idx in range(NUM_TOGGLES):
        obs[OFFSET_TOGGLES + idx] = float(toggles[idx]) if idx < len(toggles) else 0.0

    loaders = data.get("loaders", [])
    for idx in range(NUM_TOGGLES):
        obs[OFFSET_LOADERS + idx] = float(loaders[idx]) if idx < len(loaders) else 0.0

    loader_owners = data.get("loader_owners", [])
    for idx in range(NUM_TOGGLES):
        obs[OFFSET_LOADER_OWNERS + idx] = float(loader_owners[idx]) if idx < len(loader_owners) else 0.0

    # 5. Held Object Details
    obs[OFFSET_HELD_PIN_FRONT_COLOR] = COLOR_TO_VAL.get(self_bot.get("held_pin_front_color", "none"), 0.0)
    obs[OFFSET_HELD_PIN_BACK_COLOR] = COLOR_TO_VAL.get(self_bot.get("held_pin_back_color", "none"), 0.0)
    obs[OFFSET_HELD_CUP_FACE_UP] = 1.0 if self_bot.get("held_cup_face_up", False) else 0.0

    # 6. Teammate Robot (3 features: dist, sin, cos)
    tm = data.get("teammate_robot", {})
    obs[OFFSET_TEAMMATE_POSITION + 0] = float(tm.get("distance", -144.0))
    obs[OFFSET_TEAMMATE_POSITION + 1] = float(tm.get("sin_theta", 0.0))
    obs[OFFSET_TEAMMATE_POSITION + 2] = float(tm.get("cos_theta", 0.0))

    # 7. Stacks (3 slots * 7 features each: dist, sin, cos, max_z, pin_count, cup_count, high_pin)
    stacks_list = data.get("stacks", [])
    for idx in range(MAX_NEAREST_STACKS):
        base = OFFSET_STACKS + idx * STACK_OBJECT_FEATURES
        if idx < len(stacks_list):
            s = stacks_list[idx]
            obs[base + 0] = float(s.get("distance", -144.0))
            obs[base + 1] = float(s.get("sin_theta", 0.0))
            obs[base + 2] = float(s.get("cos_theta", 0.0))
            obs[base + 3] = float(s.get("max_z", s.get("height", 0.0)))
            obs[base + 4] = float(s.get("pin_count", 0.0))
            obs[base + 5] = float(s.get("cup_count", 0.0))
            obs[base + 6] = 1.0 if s.get("high_pin", False) else 0.0
        else:
            obs[base + 0] = -144.0
            obs[base + 1] = 0.0
            obs[base + 2] = 0.0
            obs[base + 3] = 0.0
            obs[base + 4] = 0.0
            obs[base + 5] = 0.0
            obs[base + 6] = 0.0

    # 8. Goals (9 slots * 13 features each)
    goals_list = data.get("goals", [])
    for idx in range(NUM_GOALS):
        base = OFFSET_GOALS + idx * GOAL_FEATURES
        if idx < len(goals_list):
            g = goals_list[idx]
            obs[base + 0] = float(g.get("distance", -144.0))
            obs[base + 1] = float(g.get("sin_theta", 0.0))
            obs[base + 2] = float(g.get("cos_theta", 0.0))
            obs[base + 3] = float(g.get("friendly_pins", 0))
            obs[base + 4] = float(g.get("opponent_pins", 0))
            obs[base + 5] = float(g.get("neutral_pins", 0))

            # 6 one-hot top piece indicators
            top_piece_str = g.get("top_piece", "empty")
            top_idx = TOP_PIECE_TO_INDEX.get(top_piece_str, 0)
            for top_i in range(6):
                obs[base + 6 + top_i] = 1.0 if top_i == top_idx else 0.0

            obs[base + 12] = float(g.get("stack_height", 0.0))
        else:
            obs[base + 0] = -144.0
            obs[base + 1] = 0.0
            obs[base + 2] = 0.0
            obs[base + 3] = 0.0
            obs[base + 4] = 0.0
            obs[base + 5] = 0.0
            obs[base + 6] = 1.0  # empty
            for top_i in range(1, 6):
                obs[base + 6 + top_i] = 0.0
            obs[base + 12] = 0.0

    return obs


def numpy_to_json(
    obs: np.ndarray,
    normalize: bool = False,
    include_empty_slots: bool = False,
) -> Dict[str, Any]:
    """
    Converts a flat 1D numpy observation vector (length 670) into a structured dictionary.
    """
    if obs.shape != (OBS_TOTAL,):
        raise ValueError(f"Expected observation vector of shape ({OBS_TOTAL},), got {obs.shape}")

    theta_val = float(obs[OFFSET_SELF_ORIENT])
    theta = (theta_val * 360.0) if normalize else theta_val

    time_val = float(obs[OFFSET_TIME_REMAINING])
    match_time = (time_val * 120.0) if normalize else time_val

    def _val_to_color(val: float) -> str:
        rounded = round(float(val), 1)
        return VAL_TO_COLOR.get(rounded, "none")

    data: Dict[str, Any] = {
        "match_time_remaining": match_time,
        "self_robot": {
            "x": 0.0,
            "y": 0.0,
            "theta": theta,
            "linear_speed": 0.0,
            "held_pins": int(round(float(obs[OFFSET_HELD_PINS]))),
            "held_cups": int(round(float(obs[OFFSET_HELD_CUPS]))),
            "parked": bool(obs[OFFSET_PARKED] > 0.5),
            "held_pin_front_color": _val_to_color(obs[OFFSET_HELD_PIN_FRONT_COLOR]),
            "held_pin_back_color": _val_to_color(obs[OFFSET_HELD_PIN_BACK_COLOR]),
            "held_cup_face_up": bool(obs[OFFSET_HELD_CUP_FACE_UP] > 0.5),
        },
        "teammate_robot": {
            "distance": float(obs[OFFSET_TEAMMATE_POSITION + 0]),
            "sin_theta": float(obs[OFFSET_TEAMMATE_POSITION + 1]),
            "cos_theta": float(obs[OFFSET_TEAMMATE_POSITION + 2]),
            "visible": bool(obs[OFFSET_TEAMMATE_POSITION + 0] > -100.0),
        },
        "pins": [],
        "cups": [],
        "toggles": [int(round(float(obs[OFFSET_TOGGLES + i]))) for i in range(NUM_TOGGLES)],
        "loaders": [int(round(float(obs[OFFSET_LOADERS + i]))) for i in range(NUM_TOGGLES)],
        "loader_owners": [int(round(float(obs[OFFSET_LOADER_OWNERS + i]))) for i in range(NUM_TOGGLES)],
        "stacks": [],
        "goals": [],
    }

    # Pins
    for idx in range(MAX_OBSERVED_PINS):
        base = OFFSET_PIN_POSITIONS + idx * PIN_OBJECT_FEATURES
        dist = float(obs[base + 0])
        vis = bool(obs[base + 3] > 0.5)
        if include_empty_slots or vis or dist > -100.0:
            data["pins"].append({
                "distance": dist,
                "sin_theta": float(obs[base + 1]),
                "cos_theta": float(obs[base + 2]),
                "visible": vis,
                "color_alliance": _val_to_color(obs[base + 4]),
            })

    # Cups
    for idx in range(NUM_CUPS):
        base = OFFSET_CUP_POSITIONS + idx * CUP_OBJECT_FEATURES
        dist = float(obs[base + 0])
        vis = bool(obs[base + 3] > 0.5)
        if include_empty_slots or vis or dist > -100.0:
            data["cups"].append({
                "distance": dist,
                "sin_theta": float(obs[base + 1]),
                "cos_theta": float(obs[base + 2]),
                "visible": vis,
            })

    # Stacks
    for idx in range(MAX_NEAREST_STACKS):
        base = OFFSET_STACKS + idx * STACK_OBJECT_FEATURES
        dist = float(obs[base + 0])
        if include_empty_slots or dist > -100.0:
            data["stacks"].append({
                "distance": dist,
                "sin_theta": float(obs[base + 1]),
                "cos_theta": float(obs[base + 2]),
                "max_z": float(obs[base + 3]),
                "pin_count": int(round(float(obs[base + 4]))),
                "cup_count": int(round(float(obs[base + 5]))),
                "high_pin": bool(obs[base + 6] > 0.5),
            })

    # Goals
    for idx in range(NUM_GOALS):
        base = OFFSET_GOALS + idx * GOAL_FEATURES
        top_onehot = obs[base + 6 : base + 12]
        top_idx = int(np.argmax(top_onehot))
        data["goals"].append({
            "distance": float(obs[base + 0]),
            "sin_theta": float(obs[base + 1]),
            "cos_theta": float(obs[base + 2]),
            "friendly_pins": int(round(float(obs[base + 3]))),
            "opponent_pins": int(round(float(obs[base + 4]))),
            "neutral_pins": int(round(float(obs[base + 5]))),
            "top_piece": INDEX_TO_TOP_PIECE.get(top_idx, "empty"),
            "stack_height": float(obs[base + 12]),
        })

    return data
