import math

def vex_normalize_angle(degrees: float) -> float:
    """
    Normalizes a VEX degree angle to the range [0.0, 360.0).
    """
    return degrees % 360.0

def vex_sin(degrees: float) -> float:
    """
    Computes sine for the VEX coordinate system where 0 degrees is North (+Y)
    and positive rotation is clockwise.
    Equivalent to math.sin(radians(degrees)) on the standard standard unit but mapped to VEX.
    
    Standard math uses X-axis as 0, CCW as positive.
    VEX uses Y-axis as 0, CW as positive.
    VEX angle 0 -> Standard angle 90 (pi/2).
    VEX angle 90 -> Standard angle 0.
    So, Standard Angle = 90 - VEX Angle.
    """
    # math.sin(math.radians(90 - degrees)) simplifies mathematically to math.cos(math.radians(degrees))
    return math.cos(math.radians(degrees))

def vex_cos(degrees: float) -> float:
    """
    Computes cosine for the VEX coordinate system where 0 degrees is North (+Y)
    and positive rotation is clockwise.
    
    Standard Angle = 90 - VEX Angle.
    math.cos(math.radians(90 - degrees)) simplifies to math.sin(math.radians(degrees))
    """
    return math.sin(math.radians(degrees))

def vex_atan2(delta_x: float, delta_y: float) -> float:
    """
    Returns the VEX heading in degrees for a given vector (delta_x, delta_y).
    0 is North (+Y), 90 is East (+X).
    """
    # Standard atan2 takes (y, x) and returns CCW from +X.
    # We want CW from +Y.
    # Therefore we swap the arguments to math.atan2(x, y).
    # Since atan2(x, y) computes angle from +Y with CW being positive.
    rad = math.atan2(delta_x, delta_y)
    return vex_normalize_angle(math.degrees(rad))

def relative_polar(
    position,
    origin,
    reference_heading: float,
) -> tuple[float, float, float]:
    """Encode a world position as distance and heading-relative sine/cosine."""
    delta_x = float(position[0]) - float(origin[0])
    delta_y = float(position[1]) - float(origin[1])
    distance = math.hypot(delta_x, delta_y)
    if distance == 0.0:
        return 0.0, 0.0, 1.0

    relative_angle = math.radians(
        vex_shortest_angular_distance(
            reference_heading,
            vex_atan2(delta_x, delta_y),
        )
    )
    return distance, math.sin(relative_angle), math.cos(relative_angle)

def relative_polar_to_position(
    polar: tuple[float, float, float],
    origin,
    reference_heading: float,
) -> tuple[float, float]:
    """Convert distance and heading-relative sine/cosine to a world position."""
    distance, sin_theta, cos_theta = polar
    theta = math.degrees(math.atan2(sin_theta, cos_theta))
    bearing = math.radians(reference_heading + theta)
    return (
        float(origin[0]) + distance * math.sin(bearing),
        float(origin[1]) + distance * math.cos(bearing),
    )

def vex_shortest_angular_distance(current_deg: float, target_deg: float) -> float:
    """
    Computes the shortest rotation from current_deg to target_deg in VEX degrees.
    Returns a value in [-180, 180]. Positive means rotate clockwise.
    """
    diff = vex_normalize_angle(target_deg) - vex_normalize_angle(current_deg)
    
    if diff > 180.0:
        diff -= 360.0
    elif diff < -180.0:
        diff += 360.0
        
    return diff

def vex_to_standard_radians(degrees: float) -> float:
    """
    Converts a VEX degree angle (0=North, +CW) to a standard unit circle 
    radian angle (0=East, +CCW). Useful for external libraries like Matplotlib.
    """
    # Standard Angle = 90 - VEX Angle
    standard_deg = 90.0 - degrees
    return math.radians(standard_deg)
