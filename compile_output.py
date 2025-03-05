import argparse
import numpy as np
import re
import csv

# -----------------------------------------------------------------------------
# Description: Generate the beginning of the file
# -----------------------------------------------------------------------------
def begin_file(file_name="auton1.h"):
    # Translate H file name into different names for within the code
    define_name = file_name.replace("_", "").replace(".", "_").upper()
    class_name_lower = file_name.replace("_", "").split(".")[0]
    class_name = class_name_lower[0].upper() + class_name_lower[1:]

    includes = ["config.h", "vex.h", "BezierCurve.h", "pointPath.h", "robotController.h"]

    # Add include guard
    ret = r'''#ifndef {name}
#define {name}'''.format(name=define_name)

    # Add included H files
    for include in includes:
        ret += "\n#include \"" + include +"\""

    # Start class definition
    ret += r'''
using namespace vex;

class {name} {{
    public:
'''.format(name=class_name)

    return ret

# -----------------------------------------------------------------------------
# Description: Generate the beginning of the run() method
# -----------------------------------------------------------------------------
def begin_run_method(initial_pos=[0, 0], initial_angle=0, \
                     drive_names=["leftDrive", "rightDrive", "centerDrive"], \
                     brake_styles=["brake", "brake", "brake"]):
    # Start run method definition
    ret = r'''    static void run() {
'''

    # Set tracker position
    ret += set_position(x=initial_pos[0], y=initial_pos[1], angle=initial_angle)

    # Set stopping brake styles
    num_drives = len(drive_names)
    for i in range(num_drives):
        ret += "        {drive}.setStopping({brake});\n".format(drive=drive_names[i], brake=brake_styles[i])

    return ret

# -----------------------------------------------------------------------------
# Description: Generate code setting the position for the tracker
# -----------------------------------------------------------------------------
def set_position(x=0, y=0, angle=0, tracker_name="tracker"):
    new_x, new_y = x * 144 / 12 - 72, y * 144 / 12 - 72
    ret = r'''        {tracker}.setPosition({x:.2f}, {y:.2f}, {angle:.2f});
'''.format(tracker=tracker_name, x=new_x, y=new_y, angle=angle)
    return ret

# -----------------------------------------------------------------------------
# Description: Generate commands for an action
# -----------------------------------------------------------------------------
def do_action(action_name, path_name=None, extra_params=None):
    ret = r'''
        // {action_name}'''.format(action_name=action_name)

    # Generate code following a path
    if (action_name == 'FORWARD' or action_name == 'BACKWARD') and path_name is not None:
        # Synchronous or asynchronous?
        asynch = False
        if extra_params is not None and len(extra_params) >= 1:
            # Account for any actions that require asynchronous movement
            if extra_params[0] == 'PICKUP_GOAL':
                asynch = True
            elif extra_params[0] == 'CLIMB':
                asynch = True
        follow_func = 'startFollow' if asynch else 'follow'

        # Direction of the robot?
        reverse = (action_name == 'BACKWARD')

        # Make the code
        ret += r'''
        robotController->setReverse({rev_bool});
        robotController->startHeading({path_name});
        robotController->{follow_func}({path_name});
'''.format(path_name=path_name, rev_bool=str(reverse).lower(), follow_func=follow_func)

    # Used so the robot can see more of the field
    elif action_name == 'TURN_TO':
        ret += '\n'  # Ignored, should be irrelevant for autonomous

    # Generate code to grab a goal once the robot arrives
    elif action_name == 'PICKUP_GOAL':
        ret += r'''
        robotController->waitForPercent(0.95);
        tooth.set(true);
        robotController->waitForCompletion();
'''

    # Generate code to place the goal where the robot is
    elif action_name == 'DROP_GOAL':
        ret += r'''
        tooth.set(false);
'''

    # Generate code to grab a ring
    elif action_name == 'PICKUP_RING':  # TODO: Should the path before this be synchronous or async?
        ret += r'''
        intake.spin(forward, 12, volt);
'''

    # Generate code to place a ring
    elif action_name == 'ADD_RING_TO_GOAL':  # TODO: Should the path before this be synchronous or async?
        ret += r'''
        transfer.spin(forward, 12, volt);
'''

    # Generate code for the robot to climb
    # TODO: Should we get the RL output to generate a path to wherever we need to go to climb before this?
    elif action_name == 'CLIMB' and extra_params is not None and len(extra_params) >= 1:
        ret += r'''
        ladyBrownToPosition(250);
        robotController->follow({path_name});
        ladyBrownToPosition(200);
        robotController->waitForCompletion();
'''.format(path_name=extra_params[0])

    # Currently unsupported action, print so user knows
    else:
        ret += '\n'
        print(f'TODO - ACTION - {action_name} - {path_name} - {extra_params}')

    return ret

# -----------------------------------------------------------------------------
# Description: Generate the end of the run() method
# -----------------------------------------------------------------------------
def end_run_method():
    ret = r'''    }
'''
    return ret

# -----------------------------------------------------------------------------
# Description: Generate the end of the file
# -----------------------------------------------------------------------------
def end_file(file_name="auton1.h"):
    # Translate H file name into different names for within the code
    define_name = file_name.replace(".", "_").upper()

    # End the class definition
    ret = r'''}};

#endif // {name}
'''.format(name=define_name)

    return ret

# -----------------------------------------------------------------------------
# Description: Generate code creating a bezier curve for the robot
# -----------------------------------------------------------------------------
def build_bez_curve(p1, p2, spd):
    ret = ""
    print("TODO - BUILD BEZIER CURVE - INCOMPLETE") # Bez Curves aren't supported
    return ret

# -----------------------------------------------------------------------------
# Description: Generate code creating a point-based path for the robot
# -----------------------------------------------------------------------------
def build_point_path(points, spd_weights, name="PointPath", action="FORWARD"):
    # Start PointPath definition
    # PointPath constructor takes only points
    ret = "    static inline PointPath* {path_name} = new PointPath(".format(path_name=name)

    # Scale points for the robot code
    new_points = [(x * 144 / 12 - 72, y * 144 / 12 - 72) for x, y in points]

    # Build vector for PointPath
    vector = f"{{{', '.join([f'{{{x:.2f}, {y:.2f}}}' for x, y in new_points])}}}"

    # Add vector and finish definition
    other_params = ""
    ret += vector + other_params + ");\n"

    return ret

# -----------------------------------------------------------------------------
# Description: Process assuming the file is a unified action/path file
# -----------------------------------------------------------------------------
def parse_unified(lines):
    # TODO: Will this be the final file format or will revisions be made yet?

    # Generate the beginning of the output file
    output = begin_file()

    # Find initial position and heading
    initial_pos = None
    initial_angle = None
    for line in lines:
        # Get action from line
        fields = [f.strip() for f in line.split(',')]
        fields = [f for f in fields if f != '']
        action_name = fields[0]

        # Extract points
        if action_name == 'FORWARD' or action_name == 'BACKWARD':
            start_idx = 1
            while start_idx <= len(fields) - 2:
                curr_pos = [float(n) for n in fields[start_idx:start_idx+2]]
                if initial_pos is None:
                    initial_pos = curr_pos
                else:
                    if curr_pos[0] != initial_pos[0] or curr_pos[1] != initial_pos[1]:
                        initial_angle_rad = np.arctan2(curr_pos[1] - initial_pos[1], curr_pos[0] - initial_pos[0])
                        initial_angle = initial_angle_rad * 180 / np.pi
                        break
                start_idx += 2
    if initial_pos is None:
        initial_pos = [2.67, 1]
    if initial_angle is None:
        initial_angle = 83

    # Start the run() method
    output += begin_run_method(initial_pos=initial_pos, initial_angle=initial_angle)

    # Iterate through source file's lines
    curr_path_id = 0
    paths = []
    path_actions = []
    for line, next_line in zip(lines, lines[1:] + [None]):
        # Get action from line
        fields = [f.strip() for f in line.split(',')]
        fields = [f for f in fields if f != '']
        action_name = fields[0]

        # Get info from next action (if we can)
        if next_line is not None:
            next_fields = [f.strip() for f in next_line.split(',')]
            next_fields = [f for f in next_fields if f != '']
            next_action_name = next_fields[0]
        else:
            next_action_name = None

        # Do we have a path?
        has_path = (action_name == 'FORWARD' or action_name == 'BACKWARD')

        call_path_id = None
        call_extra_params = None

        # Get path if there is one
        if has_path:
            curr_path_id += 1
            curr_path = []
            for x, y in zip(fields[1::2], fields[2::2]):
                curr_path.append((float(x), float(y)))
            paths.append(curr_path)
            path_actions.append(action_name)
            call_path_id = curr_path_id
            call_extra_params = [next_action_name]
        
        # Pass path name to CLIMB
        elif action_name == 'CLIMB':
            call_extra_params = None if curr_path_id == 0 else [f'Path{curr_path_id}']

        # Get extra parameters if they exist
        elif len(fields) > 1:
            call_extra_params = fields[1:]

        # Add action code to output
        output += do_action(action_name, None if call_path_id is None else f'Path{call_path_id}', \
            call_extra_params)

    # End the run() method
    output += end_run_method()

    # Generate lines defining the paths
    for idx, (path, path_action) in enumerate(zip(paths, path_actions)):
        path_id = idx + 1
        # TODO: spd_weights
        output += build_point_path(path, None, f'Path{path_id}', path_action)

    # Generate the end of the file
    output += end_file()

    return output

#
# Dao Method
#

# Regular expressions to match commands and coordinates
command_pattern = re.compile(r'(\w+),?\s*([\d\s,.-]*)')
coordinate_pattern = re.compile(r'([\d.-]+),([\d.-]+)')

# Function to convert the input commands to the desired format
def convert_to_class_format(csv_file_path):
    output = []
    bezier_curves = []
    bezier_index = 0

    # Start the class definition
    output.append(begin_file())
    output.append(begin_run_method())
    # output.append("class ConvertedClass {")
    # output.append("    public:")
    # output.append("    static void run() {")

    # Read the CSV file
    with open(csv_file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row:
                command_line = ','.join(row)
                match = command_pattern.match(command_line)
                if match:
                    command = match.group(1)
                    coordinates = match.group(2).strip()

                    # Handle FORWARD and BACKWARD commands by generating Bezier curves
                    if command in ["FORWARD", "BACKWARD"]:
                        points = coordinate_pattern.findall(coordinates)
                        curve_name = f"BezierCurve{bezier_index}"
                        # Create a Bezier curve object with the extracted points
                        bezier_curves.append(
                            f"static inline BezierCurve* {curve_name} = new BezierCurve(\"{curve_name}\", "
                            f"{{{', '.join([f'{{{x}, {y}}}' for x, y in points])}}}, 30, 3, 1, "
                            f"{'false' if command == 'FORWARD' else 'true'});"
                        )
                        # Reference the Bezier curve in the run method
                        output.append(f"        robotController->follow({curve_name});")
                        bezier_index += 1

                    # # Handle PICKUP_RING command
                    # elif command == "PICKUP_RING":
                    #     output.append("        robotController->pickupRing();")

                    # # Handle PICKUP_GOAL command
                    # elif command == "PICKUP_GOAL":
                    #     output.append("        robotController->pickupGoal();")

                    # # Handle ADD_RING_TO_GOAL command
                    # elif command == "ADD_RING_TO_GOAL":
                    #     output.append("        robotController->addRingToGoal();")

                    # # Handle ADD_RING_TO_WALL_STAKE command
                    # elif command == "ADD_RING_TO_WALL_STAKE":
                    #     output.append("        robotController->addRingToWallStake();")

                    # # Handle TURN_TO command
                    # elif command == "TURN_TO":
                    #     angle = coordinates.strip()
                    #     output.append(f"        robotController->turnTo({angle});")

                    # # Handle CLIMB command
                    # elif command == "CLIMB":
                    #     output.append("        robotController->climb();")

    # Close the run method
    output.append("    }")
    # Add the Bezier curve definitions to the class
    output.extend(bezier_curves)
    # Close the class definition
    output.append("};")

    # Join all the output lines into a single string
    return '\n'.join(output)

# -----------------------------------------------------------------------------
# Description: Parse arguments, read the file, process, and output into a new file
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Process command line arguments
    parser = argparse.ArgumentParser(description="Translate an RL model's action sequence into C++ autonomous code")
    parser.add_argument('--sequence', type=str, required=True, help='The action sequence to translate')
    parser.add_argument('--output', type=str, required=True, help='The file path for the translated code')
    args = parser.parse_args()

    # Read source file
    with open(args.sequence, 'r') as f:
        lines = f.readlines()

    # Process and convert to source code
    output = parse_unified(lines)

    # Write output string to file
    with open(args.output, 'w') as f:
        # f.write(output)
        f.write(convert_to_class_format(args.sequence))

    # Exit successfully
    print('Translation complete')
    exit(0)
