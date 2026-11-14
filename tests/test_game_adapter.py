"""
Unit tests for Generic Game Adapter and Bidirectional State Serialization (TASK-02).
"""

import sys
import os
import unittest
import numpy as np

# Ensure root of VEX-AI-Reinforcement-Learning is on sys.path
_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_dir not in sys.path:
    sys.path.insert(0, _repo_dir)

from vex_core.factory import GameAdapterFactory
from vex_core.game_adapter import BaseGameAdapter
from vex_core.converters import json_to_numpy, numpy_to_json, OBS_TOTAL
from override.override_adapter import OverrideGameAdapter
from override.override import Actions


class TestGameAdapterAndSerialization(unittest.TestCase):
    def setUp(self):
        self.adapter = GameAdapterFactory.create("override", team="red")

    def test_factory_registration(self):
        """Test that Override is registered with GameAdapterFactory."""
        available = GameAdapterFactory.get_available_games()
        self.assertIn("override", available)
        self.assertIsInstance(self.adapter, OverrideGameAdapter)
        self.assertIsInstance(self.adapter, BaseGameAdapter)
        self.assertEqual(self.adapter.game_name, "override")

    def test_initial_state_dict(self):
        """Test that get_initial_state_dict returns a valid schema dict."""
        state = self.adapter.get_initial_state_dict()
        self.assertIn("match_time_remaining", state)
        self.assertIn("self_robot", state)
        self.assertIn("pins", state)
        self.assertIn("cups", state)
        self.assertIn("toggles", state)
        self.assertIn("loaders", state)
        self.assertIn("goals", state)
        self.assertEqual(len(state["goals"]), 9)
        self.assertEqual(len(state["toggles"]), 4)

    def test_dict_to_numpy_shape_and_type(self):
        """Test that observation_dict_to_numpy produces float32 vector of length 670."""
        state = self.adapter.get_initial_state_dict()
        obs = self.adapter.observation_dict_to_numpy(state)
        self.assertEqual(obs.shape, (OBS_TOTAL,))
        self.assertEqual(obs.dtype, np.float32)

    def test_round_trip_lossless(self):
        """Test that dict -> numpy -> dict -> numpy is lossless."""
        state = self.adapter.get_initial_state_dict()
        state["match_time_remaining"] = 75.0
        state["self_robot"]["theta"] = 180.0
        state["self_robot"]["held_pins"] = 1
        state["self_robot"]["held_cups"] = 0
        state["self_robot"]["held_pin_front_color"] = "friendly"
        state["self_robot"]["parked"] = True
        state["toggles"] = [1, -1, 0, 1]
        state["loaders"] = [2, 0, 3, 1]
        state["loader_owners"] = [1, -1, 1, -1]
        state["teammate_robot"]["distance"] = 45.0
        state["teammate_robot"]["sin_theta"] = 0.5
        state["teammate_robot"]["cos_theta"] = 0.866
        state["pins"] = [
            {"distance": 25.0, "sin_theta": 0.2, "cos_theta": 0.9, "visible": True, "color_alliance": "friendly"},
            {"distance": 35.0, "sin_theta": -0.4, "cos_theta": 0.7, "visible": True, "color_alliance": "opponent"},
        ]
        state["cups"] = [
            {"distance": 18.0, "sin_theta": 0.1, "cos_theta": 0.95, "visible": True},
        ]
        state["goals"][0] = {
            "distance": 30.0,
            "sin_theta": 0.6,
            "cos_theta": 0.8,
            "friendly_pins": 2,
            "opponent_pins": 1,
            "neutral_pins": 0,
            "top_piece": "black_cup",
            "stack_height": 7.5,
        }

        # First conversion
        vec1 = self.adapter.observation_dict_to_numpy(state)
        # Convert to dict
        dict1 = self.adapter.numpy_to_observation_dict(vec1)
        # Second conversion
        vec2 = self.adapter.observation_dict_to_numpy(dict1)

        np.testing.assert_allclose(vec1, vec2, atol=1e-5)

    def test_write_back_to_dict(self):
        """Test write_back_to_dict correctly overlays updates without data loss."""
        state = self.adapter.get_initial_state_dict()
        updates = {
            "self_robot": {"x": 24.0, "y": -12.0, "theta": 90.0},
            "inferred_goal_owner": {"goal_0": "friendly"},
        }
        self.adapter.write_back_to_dict(state, updates)
        self.assertEqual(state["self_robot"]["x"], 24.0)
        self.assertEqual(state["self_robot"]["y"], -12.0)
        self.assertEqual(state["self_robot"]["theta"], 90.0)
        # Verify existing fields were not erased
        self.assertIn("held_pins", state["self_robot"])
        self.assertEqual(state["inferred_goal_owner"]["goal_0"], "friendly")

    def test_decompose_action(self):
        """Test decomposition into runnable V5 sub-actions."""
        state = self.adapter.get_initial_state_dict()
        state["self_robot"]["x"] = 0.0
        state["self_robot"]["y"] = 0.0

        # Score Goal 1
        subactions = self.adapter.decompose_action(Actions.SCORE_GOAL_1.value, state)
        self.assertTrue(len(subactions) >= 2)
        self.assertTrue(any("TURN_TO_POINT" in cmd for cmd in subactions))
        self.assertTrue(any("FOLLOW" in cmd for cmd in subactions))
        self.assertTrue(any("SCORE" in cmd for cmd in subactions))

        # Park Midfield
        park_subs = self.adapter.decompose_action(Actions.PARK_MIDFIELD.value, state)
        self.assertTrue(any("PARK" in cmd for cmd in park_subs))

        # Toggle 1
        tog_subs = self.adapter.decompose_action(Actions.FLIP_TOGGLE_1.value, state)
        self.assertTrue(any("TOGGLE" in cmd for cmd in tog_subs))

        # Idle
        idle_subs = self.adapter.decompose_action(Actions.IDLE.value, state)
        self.assertEqual(idle_subs, ["WAIT;0.5"])

    def test_action_success_lifecycle(self):
        """Test state transitions on action success."""
        state = self.adapter.get_initial_state_dict()
        state["self_robot"]["held_pins"] = 1
        state["self_robot"]["held_cups"] = 1

        # Scoring clears held items
        self.adapter.on_action_success(Actions.SCORE_GOAL_1.value, state)
        self.assertEqual(state["self_robot"]["held_pins"], 0)
        self.assertEqual(state["self_robot"]["held_cups"], 0)

        # Pickup pin increments held_pins
        self.adapter.on_action_success(Actions.PICKUP_PIN_RED_YELLOW.value, state)
        self.assertEqual(state["self_robot"]["held_pins"], 1)

        # Park sets parked = True
        self.adapter.on_action_success(Actions.PARK_MIDFIELD.value, state)
        self.assertTrue(state["self_robot"]["parked"])

        # Non-parking action resets parked = False
        self.adapter.on_action_success(Actions.FLIP_TOGGLE_1.value, state)
        self.assertFalse(state["self_robot"]["parked"])

    def test_action_error_handling(self):
        """Test that action errors are recorded in state."""
        state = self.adapter.get_initial_state_dict()
        self.adapter.on_action_error(Actions.SCORE_GOAL_1.value, 102, state)
        self.assertIn("last_error", state)
        self.assertEqual(state["last_error"]["action_idx"], Actions.SCORE_GOAL_1.value)
        self.assertEqual(state["last_error"]["error_code"], 102)

    def test_validate_subaction_precondition(self):
        """Test option-critic intra-step precondition checks."""
        state = self.adapter.get_initial_state_dict()
        state["self_robot"]["held_pins"] = 0
        state["self_robot"]["held_cups"] = 0

        # Cannot execute SCORE with 0 held objects
        ok, reason = self.adapter.validate_subaction_precondition(
            Actions.SCORE_GOAL_1.value, 2, "SCORE;(-24.0,48.0);0.0", state
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "NoElementsHeldForScoring")

        # Can execute after picking up an item
        state["self_robot"]["held_pins"] = 1
        ok, reason = self.adapter.validate_subaction_precondition(
            Actions.SCORE_GOAL_1.value, 2, "SCORE;(-24.0,48.0);0.0", state
        )
        self.assertTrue(ok)
        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
