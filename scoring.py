"""
VEX Push Back - Scoring Module

Implements scoring rules for all competition modes.
Scoring varies between Skills (1pt/block) and Competition (3pts/block).
"""

from typing import Dict, List
import numpy as np

try:
    from .game_modes import CompetitionMode, ModeConfig, get_mode_config, Team
    from .field import GoalType, GOALS, PARK_ZONES
    from .goals import GoalManager, BlockStatus
except ImportError:
    from game_modes import CompetitionMode, ModeConfig, get_mode_config, Team
    from field import GoalType, GOALS, PARK_ZONES
    from goals import GoalManager, BlockStatus


class ScoreCalculator:
    """
    Calculates scores for VEX Push Back matches.
    
    Scoring rules vary by competition mode:
    - Skills: 1pt per block, 5pt control zones, 15pt parking
    - Competition: 3pts per block, 10pt control zones, 8-30pt parking
    """
    
    def __init__(self, mode: CompetitionMode):
        """
        Initialize score calculator for a specific mode.
        
        Args:
            mode: Competition mode to use for scoring rules
        """
        self.mode = mode
        self.config = get_mode_config(mode)
    
    def calculate_total_score(
        self,
        goal_manager: GoalManager,
        loaders: List[int],
        agents: Dict,
        blocks: List[Dict],
    ) -> int:
        """
        Calculate the total score for a match state.
        
        Args:
            goal_manager: GoalManager with current goal states
            loaders: List of block counts per loader [4 elements]
            agents: Dictionary of agent states
            blocks: List of block dictionaries with positions and statuses
            
        Returns:
            Total score for the match
        """
        score = 0
        
        # 1. Points for blocks in goals
        score += self._score_blocks_in_goals(goal_manager)
        
        # 2. Points for control zones
        score += self._score_control_zones(goal_manager)
        
        # 3. Points for cleared loaders
        score += self._score_cleared_loaders(loaders)
        
        # 4. Points for cleared park zones
        score += self._score_cleared_park_zones(blocks)
        
        # 5. Points for parked robots
        score += self._score_parked_robots(agents)
        
        return score
    
    def _score_blocks_in_goals(self, goal_manager: GoalManager) -> int:
        """Calculate points from blocks scored in goals."""
        total_blocks = sum(goal_manager.get_goal_counts().values())
        return total_blocks * self.config.block_points
    
    def _score_control_zones(self, goal_manager: GoalManager) -> int:
        """Calculate points from controlling goal zones."""
        score = 0
        counts = goal_manager.get_goal_counts()
        
        # Long goals (need 3+ blocks for control)
        for goal_type in [GoalType.LONG_1, GoalType.LONG_2]:
            goal_info = GOALS[goal_type]
            if counts[goal_type] >= goal_info.control_threshold:
                score += self.config.control_zone_long
        
        # Center goals - upper and lower have different point values
        goal_info_upper = GOALS[GoalType.CENTER_UPPER]
        if counts[GoalType.CENTER_UPPER] >= goal_info_upper.control_threshold:
            score += self.config.control_zone_center_upper
        
        goal_info_lower = GOALS[GoalType.CENTER_LOWER]
        if counts[GoalType.CENTER_LOWER] >= goal_info_lower.control_threshold:
            score += self.config.control_zone_center_lower
        
        return score
    
    def _score_cleared_loaders(self, loaders: List[int]) -> int:
        """Calculate points from cleared loaders."""
        cleared_count = sum(1 for count in loaders if count == 0)
        return cleared_count * self.config.cleared_loader
    
    def _score_cleared_park_zones(self, blocks: List[Dict]) -> int:
        """Calculate points from park zones with no blocks."""
        score = 0
        
        for team_name, park_zone in PARK_ZONES.items():
            zone_clear = True
            for block in blocks:
                if block["status"] == BlockStatus.ON_FIELD:
                    if park_zone.contains(block["position"]):
                        zone_clear = False
                        break
            
            if zone_clear:
                score += self.config.cleared_park_zone
        
        return score
    
    def _score_parked_robots(self, agents: Dict) -> int:
        """Calculate points from parked robots."""
        if not agents:
            return 0
        
        # Count parked robots per team
        parked_by_team = {"red": 0, "blue": 0}
        
        for agent_name, agent_state in agents.items():
            if agent_state.get("parked", False):
                if "red" in agent_name:
                    parked_by_team["red"] += 1
                else:
                    parked_by_team["blue"] += 1
        
        score = 0
        
        # In skills mode, just count parked robots
        if self.config.is_skills:
            total_parked = sum(parked_by_team.values())
            score = total_parked * self.config.park_single
        else:
            # Competition mode: different points for 1 vs 2 robots
            for team, count in parked_by_team.items():
                if count >= 2:
                    score += self.config.park_double
                elif count == 1:
                    score += self.config.park_single
        
        return score
    
    def calculate_team_scores(
        self,
        goal_manager: GoalManager,
        loaders: List[int],
        agents: Dict,
        blocks: List[Dict],
    ) -> Dict[str, int]:
        """
        Calculate scores per team for competition modes.
        
        In skills modes, returns a single 'red' team score.
        
        Args:
            goal_manager: GoalManager with current goal states
            loaders: List of block counts per loader
            agents: Dictionary of agent states
            blocks: List of block dictionaries
            
        Returns:
            Dictionary mapping team names to scores
        """
        # For skills mode, return single score
        total = self.calculate_total_score(goal_manager, loaders, agents, blocks)
        
        if self.config.is_skills:
            return {"red": total}
        
        # For competition, would need to track block ownership
        # For now, return total as shared (blocks aren't colored by team in current impl)
        return {"red": total, "blue": 0}


def compute_instant_reward(
    own_score_before: int,
    own_score_after: int,
    opponent_score_before: int,
    opponent_score_after: int,
    penalty: float = 0.0
) -> float:
    """
    Compute the reward for a single step based on team score changes.
    
    Args:
        own_score_before: Agent's team score before the action
        own_score_after: Agent's team score after the action
        opponent_score_before: Opponent's team score before the action
        opponent_score_after: Opponent's team score after the action
        penalty: Any penalty applied (positive value to subtract)
        
    Returns:
        Reward value: (own_delta) - (opponent_delta) - penalty
    """
    own_delta = own_score_after - own_score_before
    opponent_delta = opponent_score_after - opponent_score_before
    return own_delta - opponent_delta - penalty
