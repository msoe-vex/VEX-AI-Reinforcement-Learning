import argparse

# -----------------------------------------------------------------------------
# Description: Generate the beginning of the file
# -----------------------------------------------------------------------------
def begin_file(file_name="auton1.h"):
    # Translate H file name into different names for within the code
    define_name = file_name.replace(".", "_").upper()
    class_name_lower = file_name.split(".")[0]
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
def begin_run_method(drive_names=["leftDrive", "rightDrive", "centerDrive"], brake_styles=["brake", "brake", "brake"]):
    # Start run method definition
    ret = r'''    static void run() {
'''

    # Set tracker position
    # TODO: Initial position?
    ret += set_position()

    # Set stopping brake styles
    num_drives = len(drive_names)
    for i in range(num_drives):
        ret += "        {drive}.setStopping({brake});\n".format(drive=drive_names[i], brake=brake_styles[i])

    return ret

# -----------------------------------------------------------------------------
# Description: Generate code setting the position for the tracker
# -----------------------------------------------------------------------------
def set_position(x=0, y=0, angle=0, tracker_name="tracker"):
    ret = r'''        {tracker}.setPosition({x}, {y}, {angle});
'''.format(tracker=tracker_name, x=x, y=y, angle=angle)
    return ret

# -----------------------------------------------------------------------------
# Description: Generate commands for an action
# -----------------------------------------------------------------------------
def do_action(action_name, path_name=None, extra_params=None):
    ret = ''

    # Generate code following a path
    if (action_name == 'FORWARD' or action_name == 'BACKWARD') and path_name is not None:
        reverse = (action_name == 'BACKWARD')
        ret = r'''
        robotController->setReverse({rev_bool});
        robotController->startHeading({path_name});
        robotController->follow({path_name});
'''.format(path_name=path_name, rev_bool=str(reverse).lower())

    elif action_name == 'TURN_TO':
        pass  # Ignored, should be irrelevant for autonomous

    # Currently unsupported action, print so user knows
    else:
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
# Description: Generate the code defining what each path is
# -----------------------------------------------------------------------------
def create_path(path_name, path, path_action):
    # TODO: Seems replaced by build_point_path, remove?
    print(f'TODO - PATH CREATION - {path_name} - {path_action} - {path}')
    return ''

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

    # Start the run() method
    output += begin_run_method()

    # Iterate through source file's lines
    curr_path_id = 0
    paths = []
    path_actions = []
    for line in lines:
        # Get action from line
        fields = [f.strip() for f in line.split(',')]
        fields = [f for f in fields if f != '']
        action_name = fields[0]

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

# -----------------------------------------------------------------------------
# Description: Process assuming the file is a path output
# -----------------------------------------------------------------------------
def parse_path_only(lines):
    # Generate the beginning of the output file
    output = begin_file()

    # Start the run() method
    output += begin_run_method()

    # Single action: Follow path
    output += do_action('FORWARD', 1)

    # Iterate through source file's lines
    path = []
    for line in lines:
        # Get name and value from line
        fields = line.strip().split(', ')
        if len(fields) != 3:
            break # Probably the last line
        x, y, _ = fields
        path.append((float(x), float(y)))

    # End the run() method
    output += end_run_method()

    # Generate lines defining the paths
    # TODO: spd_weights
    output += create_path(path, None, 'Path1', 'FORWARD')

    # Generate the end of the file
    output += end_file()

    return output

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
    if 'endData' in lines[-1]:
        output = parse_path_only(lines)
    else:
        output = parse_unified(lines)

    # Write output string to file
    with open(args.output, 'w') as f:
        f.write(output)

    # Exit successfully
    print('Translation complete')
    exit(0)
