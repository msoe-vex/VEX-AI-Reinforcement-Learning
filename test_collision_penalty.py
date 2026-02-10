"""
Test script to verify collision penalty functionality.

Tests that robots are penalized when they collide with each other
after completing their actions.
"""

import numpy as np
from pushback import PushBackGame, Actions
from vex_core.base_game import Robot, RobotSize, Team


def test_collision_detection():
    """Test that collision detection works correctly."""
    
    # Create a game with custom robot positions for testing
    robots = [
        Robot(name="red_robot_0", team=Team.RED, size=RobotSize.INCH_24,
              start_position=np.array([0.0, 0.0], dtype=np.float32)),
        Robot(name="blue_robot_0", team=Team.BLUE, size=RobotSize.INCH_24,
              start_position=np.array([5.0, 0.0], dtype=np.float32)),  # Close to red robot
    ]
    
    game = PushBackGame.get_game("vexai_comp")
    game.robots = robots
    game._robot_map = {r.name: r for r in robots}
    game.get_initial_state()
    
    # Test 1: Check collision detection when robots are close
    print("Test 1: Collision Detection")
    print(f"Red Robot position: {game.state['agents']['red_robot_0']['position']}")
    print(f"Blue Robot position: {game.state['agents']['blue_robot_0']['position']}")
    
    is_colliding = game.check_robot_collision("red_robot_0")
    print(f"Red robot colliding with blue: {is_colliding}")
    assert is_colliding, "Collision should be detected when robots are 5.0 units apart"
    
    # Test 2: Check no collision when robots are far apart
    print("\nTest 2: No Collision at Distance")
    game.state["agents"]["blue_robot_0"]["position"] = np.array([50.0, 50.0], dtype=np.float32)
    print(f"Blue Robot new position: {game.state['agents']['blue_robot_0']['position']}")
    
    is_colliding = game.check_robot_collision("red_robot_0")
    print(f"Red robot colliding with blue: {is_colliding}")
    assert not is_colliding, "Collision should not be detected when robots are far apart"
    
    # Test 3: Check collision penalty is applied
    print("\nTest 3: Collision Penalty in Action")
    game.state["agents"]["blue_robot_0"]["position"] = np.array([5.0, 0.0], dtype=np.float32)
    
    # Execute an action for red robot
    duration, penalty = game.execute_action("red_robot_0", Actions.IDLE.value)
    
    print(f"Action duration: {duration}")
    print(f"Penalty (including collision): {penalty}")
    print(f"Collision penalty value: {game.get_collision_penalty()}")
    
    # The penalty should include both IDLE penalty (DEFAULT_PENALTY) and collision penalty
    expected_penalty = 1.0 + game.get_collision_penalty()  # DEFAULT_PENALTY + collision
    assert penalty == expected_penalty, f"Expected penalty {expected_penalty}, got {penalty}"
    
    print("\n✓ All collision tests passed!")


def test_no_penalty_without_collision():
    """Test that no collision penalty is applied when robots don't collide."""
    
    robots = [
        Robot(name="red_robot_0", team=Team.RED, size=RobotSize.INCH_24,
              start_position=np.array([0.0, 0.0], dtype=np.float32)),
        Robot(name="blue_robot_0", team=Team.BLUE, size=RobotSize.INCH_24,
              start_position=np.array([50.0, 50.0], dtype=np.float32)),  # Far from red robot
    ]
    
    game = PushBackGame.get_game("vexai_comp")
    game.robots = robots
    game._robot_map = {r.name: r for r in robots}
    game.get_initial_state()
    
    print("\nTest 4: No Penalty Without Collision")
    print(f"Red Robot position: {game.state['agents']['red_robot_0']['position']}")
    print(f"Blue Robot position: {game.state['agents']['blue_robot_0']['position']}")
    
    is_colliding = game.check_robot_collision("red_robot_0")
    assert not is_colliding, "No collision should be detected"
    
    # Execute an action
    duration, penalty = game.execute_action("red_robot_0", Actions.IDLE.value)
    
    print(f"Penalty (without collision): {penalty}")
    
    # The penalty should only be DEFAULT_PENALTY (1.0), no collision penalty
    expected_penalty = 1.0  # DEFAULT_PENALTY only
    assert penalty == expected_penalty, f"Expected penalty {expected_penalty}, got {penalty}"
    
    print("✓ No collision penalty test passed!")


def test_different_robot_sizes():
    """Test collision detection with robots of different sizes."""
    
    robots = [
        Robot(name="red_robot_0", team=Team.RED, size=RobotSize.INCH_24,
              start_position=np.array([0.0, 0.0], dtype=np.float32)),
        Robot(name="blue_robot_0", team=Team.BLUE, size=RobotSize.INCH_15,
              start_position=np.array([30.0, 0.0], dtype=np.float32)),
    ]
    
    game = PushBackGame.get_game("vexai_comp")
    game.robots = robots
    game._robot_map = {r.name: r for r in robots}
    game.get_initial_state()
    
    print("\nTest 5: Different Robot Sizes")
    print(f"Red Robot size: {robots[0].size.value}\"")
    print(f"Blue Robot size: {robots[1].size.value}\"")
    print(f"Min collision distance (sum of radii): {24/2 + 15/2} units")
    print(f"Actual distance between robots: 30.0 units")
    
    # With a 24" and 15" robot, min collision distance is 19.5 units
    # They are 30 units apart, so no collision
    is_colliding = game.check_robot_collision("red_robot_0")
    print(f"Red robot colliding: {is_colliding}")
    assert not is_colliding, "No collision at 30 units for 24\" and 15\" robots"
    
    # Move them closer to just at threshold
    game.state["agents"]["blue_robot_0"]["position"] = np.array([19.4, 0.0], dtype=np.float32)
    print(f"\nBlue Robot new position: {game.state['agents']['blue_robot_0']['position']}")
    print(f"New distance: 19.4 units (just under 19.5 threshold)")
    
    is_colliding = game.check_robot_collision("red_robot_0")
    print(f"Red robot colliding: {is_colliding}")
    assert is_colliding, "Collision at 19.4 units for 24\" and 15\" robots"
    
    # Move just past threshold
    game.state["agents"]["blue_robot_0"]["position"] = np.array([19.6, 0.0], dtype=np.float32)
    print(f"\nBlue Robot new position: {game.state['agents']['blue_robot_0']['position']}")
    print(f"New distance: 19.6 units (just over 19.5 threshold)")
    
    is_colliding = game.check_robot_collision("red_robot_0")
    print(f"Red robot colliding: {is_colliding}")
    assert not is_colliding, "No collision at 19.6 units for 24\" and 15\" robots"
    
    print("✓ Different robot sizes test passed!")


if __name__ == "__main__":
    test_collision_detection()
    test_no_penalty_without_collision()
    test_different_robot_sizes()
    print("\n✓✓✓ All collision penalty tests passed! ✓✓✓")
