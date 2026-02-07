
import unittest
import numpy as np
from unittest.mock import MagicMock
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vex_core.base_env import VexMultiAgentEnv, DELTA_T
from vex_core.base_game import VexGame, Robot, RobotSize, Team

class MockGame(VexGame):
    def __init__(self):
        super().__init__()
        # self.possible_agents is likely a property in Base, so we override it at class level or mocked proeprty
        pass
        
    @property
    def possible_agents(self):
        return ["agent_0"]

    def reset(self):
        # Prevent base class from clearing state if it does so
        pass

        # Initialize state with required keys
        self.state = {
            "agents": {
                "agent_0": {
                    "position": np.array([0.0, 0.0]),
                    "orientation": np.array([0.0]),
                    # "gameTime": 0.0, # Removed
                    "action_skipped": False
                }
            }
        }
        
    @property
    def total_time(self):
        return self.total_time_value
        
    @property
    def field_size_inches(self):
        return 144.0
        
    @property
    def num_actions(self):
        return 2

    @property
    def fallback_action(self):
        return 0

    def action_to_name(self, action):
        return str(action)
        
    def get_permanent_obstacles(self):
        return []
        
    def get_team_for_agent(self, agent):
        return Team.RED
        
    def render_game_elements(self, ax):
        pass
        
    def render_info_panel(self, ax_info, agents, actions, rewards, num_moves, agent_times=None):
        pass
        
    def update_tracker(self, tracker):
        pass
        
    def update_observation_from_tracker(self, obs, agent, tracker):
        return obs
        
    def split_action(self, action):
        return action
        
    def _get_robot_dimensions(self, agent_name):
        return 10.0, 10.0
        
    def get_initial_state(self, randomize=False, seed=None):
        return self.state
        
    def execute_action(self, agent, action):
        # Action 0: Move to (10, 0), takes 1.0s
        if action == 0:
            self.state["agents"][agent]["position"] = np.array([10.0, 0.0])
            return 1.0, 0.0
        return 0.1, 0.0
        
    def compute_score(self):
        return 0
        
    def compute_reward(self, agent, initial_scores, new_scores, penalty):
        return 0.0
        
    def is_agent_terminated(self, agent, game_time=0.0):
        return False
        
    def get_observation(self, agent, game_time=0.0):
        return np.zeros(10)
    
    def observation_space(self, agent):
        return MagicMock()
        
    def action_space(self, agent):
        return MagicMock()
    
    # Required abstract methods from VexGame that might be missing?
    # Checked base_game.py in mind (not shown fully) but these seem usually it.
    def get_game(cls, game_name):
        return MockGame()
    
    @staticmethod
    def get_game_static(game_name):
         return MockGame()
         
    def _get_scoring_config(self):
        pass
    def _get_agents_config(self):
        pass
    def _get_robot_configs(self):
        pass
    def _get_initial_blocks(self, randomize, seed):
        pass

class TestDiscreteTime(unittest.TestCase):
    def test_busy_logic_and_interpolation(self):
        game = MockGame()
        env = VexMultiAgentEnv(game=game)
        env.reset()
        
        # Step 1: Execute action 0 (Duration 1.0s, delta_t=0.1 -> 10 ticks)
        obs, rewards, terms, truncs, infos = env.step({"agent_0": 0})
        
        agent_state = env.environment_state["agents"]["agent_0"]
        
        # Check busy state
        self.assertTrue("agent_0" in env.busy_state)
        # Should be 10 ticks (1.0 / 0.1)
        self.assertEqual(env.busy_state["agent_0"]["remaining_ticks"], 10)
        
        # Position should be at start (0,0) because we interpolate
        np.testing.assert_array_almost_equal(agent_state["position"], [0.0, 0.0])
        
        # Step 2: Next Tick
        # Busy check reduces ticks to 9.
        # Alpha = (10 - 9) / 10 = 0.1
        # Target Delta = (10, 0) - (0, 0) = (10, 0)
        # Pos = 0 + 10 * 0.1 = 1.0
        env.step({}) 
        
        self.assertEqual(env.busy_state["agent_0"]["remaining_ticks"], 9)
        np.testing.assert_array_almost_equal(agent_state["position"], [1.0, 0.0])
        
        # Check orientation: Should point along X axis (0 rads) from (0,0) to (10,0)
        # 0.0 rads.
        self.assertAlmostEqual(agent_state["orientation"][0], 0.0)
        
        # Step 3: Fast forward
        for _ in range(8):
            env.step({})
            
        # Remaining should be 1
        self.assertEqual(env.busy_state["agent_0"]["remaining_ticks"], 1)
        np.testing.assert_array_almost_equal(agent_state["position"], [9.0, 0.0])
        
        # Step 4: Final tick
        env.step({})
        # Should be done
        self.assertFalse("agent_0" in env.busy_state)
        np.testing.assert_array_almost_equal(agent_state["position"], [10.0, 0.0])

if __name__ == "__main__":
    unittest.main()
