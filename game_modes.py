"""
VEX Push Back - Game Modes Configuration

Defines competition modes and their specific rules for:
- VEX U Competition (2v2 teamplay)
- VEX U Skills (single robot)
- VEX AI Skills (single AI robot)
- VEX AI Competition (2v2 AI teamplay)
"""

from enum import Enum
from dataclasses import dataclass


class CompetitionMode(Enum):
    """Four supported VEX competition modes."""
    VEX_U_COMPETITION = "vex_u_competition"
    VEX_U_SKILLS = "vex_u_skills"
    VEX_AI_SKILLS = "vex_ai_skills"
    VEX_AI_COMPETITION = "vex_ai_competition"


class Team(Enum):
    """Robot team assignment."""
    RED = "red"
    BLUE = "blue"


@dataclass
class ModeConfig:
    """Configuration for a specific competition mode."""
    auto_time: float        # Autonomous period duration in seconds
    driver_time: float      # Driver-controlled period duration in seconds
    total_time: float       # Total match time
    robots_per_alliance: int  # Number of robots per alliance (or total for skills)
    is_skills: bool         # Whether this is a skills challenge
    start_same_side: bool   # True = robots start same side, False = opposite sides
    block_points: int       # Points per block scored
    control_zone_long: int  # Points for controlling long goal
    control_zone_center: int  # Points for controlling center goal
    park_single: int        # Points for parking one robot
    park_double: int        # Points for parking both robots
    cleared_loader: int     # Points for clearing a loader
    cleared_park_zone: int  # Points for clearing park zone of blocks


# Mode-specific configurations
MODE_CONFIGS = {
    CompetitionMode.VEX_U_COMPETITION: ModeConfig(
        auto_time=15.0,
        driver_time=105.0,  # 1 minute 45 seconds
        total_time=120.0,
        robots_per_alliance=2,
        is_skills=False,
        start_same_side=True,  # Each alliance starts on their side
        block_points=3,
        control_zone_long=10,
        control_zone_center=10,
        park_single=8,
        park_double=30,
        cleared_loader=5,
        cleared_park_zone=5,
    ),
    CompetitionMode.VEX_U_SKILLS: ModeConfig(
        auto_time=0.0,  # No separate auto phase in skills
        driver_time=60.0,
        total_time=60.0,
        robots_per_alliance=2,  # 2 robots working together
        is_skills=True,
        start_same_side=True,  # Both robots start on red (same) side
        block_points=1,
        control_zone_long=5,
        control_zone_center=10,
        park_single=15,
        park_double=15,  # Same as single in skills
        cleared_loader=5,
        cleared_park_zone=5,
    ),
    CompetitionMode.VEX_AI_SKILLS: ModeConfig(
        auto_time=0.0,
        driver_time=60.0,
        total_time=60.0,
        robots_per_alliance=2,  # 2 robots working together
        is_skills=True,
        start_same_side=False,  # Robots start on OPPOSITE sides
        block_points=1,
        control_zone_long=5,
        control_zone_center=10,
        park_single=15,
        park_double=15,
        cleared_loader=5,
        cleared_park_zone=5,
    ),
    CompetitionMode.VEX_AI_COMPETITION: ModeConfig(
        auto_time=15.0,
        driver_time=105.0,
        total_time=120.0,
        robots_per_alliance=2,
        is_skills=False,
        start_same_side=True,
        block_points=3,
        control_zone_long=10,
        control_zone_center=10,
        park_single=8,
        park_double=30,
        cleared_loader=5,
        cleared_park_zone=5,
    ),
}


def get_mode_config(mode: CompetitionMode) -> ModeConfig:
    """Get the configuration for a specific competition mode."""
    return MODE_CONFIGS[mode]


def get_agents_for_mode(mode: CompetitionMode) -> list:
    """
    Get the list of agent names for a specific competition mode.
    
    Skills modes:
    - VEX U Skills: 2 red robots (both start same side)
    - VEX AI Skills: 1 red + 1 blue robot (start opposite sides)
    
    Competition modes:
    - 2 red + 2 blue robots (opposing alliances)
    """
    config = get_mode_config(mode)
    agents = []
    
    if config.is_skills:
        if config.start_same_side:
            # VEX U Skills: Both robots are red, start same side
            for i in range(config.robots_per_alliance):
                agents.append(f"red_robot_{i}")
        else:
            # VEX AI Skills: One red, one blue, start opposite sides
            agents.append("red_robot_0")
            agents.append("blue_robot_0")
    else:
        # Competition: Red vs Blue alliances
        for i in range(config.robots_per_alliance):
            agents.append(f"red_robot_{i}")
        for i in range(config.robots_per_alliance):
            agents.append(f"blue_robot_{i}")
    
    return agents


def get_team_from_agent(agent_name: str) -> Team:
    """Determine team from agent name."""
    if "red" in agent_name.lower():
        return Team.RED
    elif "blue" in agent_name.lower():
        return Team.BLUE
    return Team.RED  # Default
