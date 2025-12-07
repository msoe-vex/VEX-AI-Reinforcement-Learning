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
    control_zone_center_upper: int  # Points for controlling center goal upper
    control_zone_center_lower: int  # Points for controlling center goal lower
    park_single: int        # Points for parking one robot
    park_double: int        # Points for parking both robots
    cleared_loader: int     # Points for clearing a loader
    cleared_park_zone: int  # Points for clearing park zone of blocks


# Mode-specific configurations
MODE_CONFIGS = {
    CompetitionMode.VEX_U_COMPETITION: ModeConfig(
        auto_time=30.0,  # VEX U has 30 second autonomous
        driver_time=90.0,  # VEX U has 90 second driver (VUT4, VUT5)
        total_time=120.0,
        robots_per_alliance=2,
        is_skills=False,
        start_same_side=True,  # Each alliance starts on their side
        block_points=3,  # SC scoring table
        control_zone_long=10,  # SC scoring table
        control_zone_center_upper=8,  # SC scoring table - Upper center
        control_zone_center_lower=6,  # SC scoring table - Lower center
        park_single=8,  # SC scoring table
        park_double=30,  # SC scoring table
        cleared_loader=5,  # Not mentioned in competition scoring
        cleared_park_zone=5,  # Not mentioned in competition scoring
    ),
    CompetitionMode.VEX_U_SKILLS: ModeConfig(
        auto_time=0.0,  # No separate auto phase in skills
        driver_time=60.0,  # RSC2 - 60 second matches
        total_time=60.0,
        robots_per_alliance=2,  # 2 robots working together (VURS3)
        is_skills=True,
        start_same_side=True,  # Both robots start on red (same) side
        block_points=1,  # RSC2 scoring table
        control_zone_long=5,  # RSC2 scoring table - "filled" control zone
        control_zone_center_upper=10,  # RSC2 scoring table - filled center
        control_zone_center_lower=10,  # RSC2 scoring table - filled center
        park_single=15,  # RSC2 scoring table
        park_double=15,  # Same as single in skills
        cleared_loader=5,  # RSC2 scoring table
        cleared_park_zone=5,  # RSC2 scoring table
    ),
    CompetitionMode.VEX_AI_SKILLS: ModeConfig(
        auto_time=0.0,
        driver_time=60.0,  # 60 second matches per VURC rules
        total_time=60.0,
        robots_per_alliance=2,  # 2 robots working together (VAIRS2)
        is_skills=True,
        start_same_side=False,  # Robots start on OPPOSITE sides (VAIRS4)
        block_points=3,  # VAIRS7 scoring table - majority color blocks
        control_zone_long=0,  # VAIRS7e - Control zones not used in AI Skills
        control_zone_center_upper=0,  # VAIRS7e - Control zones not used
        control_zone_center_lower=0,  # VAIRS7e - Control zones not used
        park_single=5,  # VAIRS7 scoring table
        park_double=5,  # Same in skills
        cleared_loader=0,  # Different system in AI Skills (loader matching)
        cleared_park_zone=0,  # Not used in AI Skills
    ),
    CompetitionMode.VEX_AI_COMPETITION: ModeConfig(
        auto_time=15.0,  # Isolation Period (VAIT1)
        driver_time=105.0,  # Interaction Period (VAIT1)
        total_time=120.0,
        robots_per_alliance=2,
        is_skills=False,
        start_same_side=True,
        block_points=3,  # VAISC scoring table
        control_zone_long=10,  # Control Bonus (VAISC1)
        control_zone_center_upper=10,  # Control Bonus is same for all
        control_zone_center_lower=10,  # Control Bonus is same for all
        park_single=10,  # VAISC2 - Parked 24" Robot (only 24" can park)
        park_double=10,  # Only one robot can park in AI
        cleared_loader=0,  # Not used in AI Competition
        cleared_park_zone=0,  # Not used in AI Competition
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
