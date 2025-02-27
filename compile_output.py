import argparse

# -----------------------------------------------------------------------------
# Description: Generate the beginning of the file
# -----------------------------------------------------------------------------
def begin_file(file_name="AUTON1_H"):
    # TODO: Dummy line for testing
    includes = ["config.h", "vex.h", "BezierCurve.h", "robotController.h"]

    ret = r'''#ifndef {name}
#define {name}
    '''.format(name=file_name)
    for include in includes:
        ret += "\n#include \"" + include +"\""
    ret += r'''
using namespace vex
class {name} {{
    public:
    '''.format(name=file_name)
    return ret

# -----------------------------------------------------------------------------
# Description: Generate the beginning of the run() method
# -----------------------------------------------------------------------------
def begin_run_method(drive_names=["leftDrive", "rightDrive", "centerDrive"], brake_styles=["brake", "brake", "brake"]):
    ret = r'''static void run() {{
'''
    ret += set_position()
    num_drives = len(drive_names)
    for i in range(num_drives):
        ret += "{drive}.setStopping({brake});".format(drive=drive_names[i], brake=brake_styles[i])
    return ret

def set_position(x=0, y=0, angle=0, tracker_name="tracker"):
    ret = r'''
    {tracker}.setPosition({x}, {y}, {angle});'''.format(tracker=tracker_name, x=x, y=y, angle=angle)
    return ret
# -----------------------------------------------------------------------------
# Description: Generate commands for an action
# -----------------------------------------------------------------------------
def do_action(action_name, path_id):
    # TODO: Dummy line for testing
    return f'ACTION TEST - {action_name} - {path_id}\n'

# -----------------------------------------------------------------------------
# Description: Generate the end of the run() method
# -----------------------------------------------------------------------------
def end_run_method():
    # TODO: Dummy line for testing
    ret = r'''    }}'''
    return 'RUN METHOD END TEST\n'

# -----------------------------------------------------------------------------
# Description: Generate the code defining what each path is
# -----------------------------------------------------------------------------
def create_path(path_id, path):
    # TODO: Dummy line for testing
    return f'PATH CREATION TEST - {path_id} - {path}\n'

# -----------------------------------------------------------------------------
# Description: Generate the end of the file
# -----------------------------------------------------------------------------
def end_file(file_name="AUTON1_H"):
    # TODO: Dummy line for testing
    ret = r'''
}};

endif // {name}
'''.format(name=file_name)
    return ret

# -----------------------------------------------------------------------------
# Description: Parse arguments, read the file, and call methods for each line
# -----------------------------------------------------------------------------
def build_bez_curve(p1, p2, spd):
    ret = ""
    return "TODO - INCOMPLETE" # Bez Curves aren't supported

def build_point_path(points, spd_weights, name="PointPath"):
    ret = "static inline PointPath* {path_name} = new PointPath(\"{path_name}\", ".format(path_name=name)
    # Build vector for PointPath
    vector = "{"
    vector += f'{{{', '.join([str(point) for point in points])}}}'
    vector += ", 100, 3, 1, false" # constants
    vector += "}"
    ret += ");"
    return ret
        

if __name__ == "__main__":
    # Process command line arguments
    parser = argparse.ArgumentParser(description="Translate an RL model's action sequence into C++ autonomous code")
    parser.add_argument('--sequence', type=str, required=True, help='The action sequence to translate')
    parser.add_argument('--output', type=str, required=True, help='The file path for the translated code')
    args = parser.parse_args()

    # Read source file
    with open(args.sequence, 'r') as f:
        sequence = f.readlines()

    # Generate the beginning of the output file
    output = begin_file()

    # Start the run() method
    output += begin_run_method()

    # Iterate through source file's lines
    action_name = None
    curr_x = None
    last_path_id = 0
    curr_path_id = 0
    paths = []
    curr_path = []
    for line in sequence + ['end,end']:
        # Get name and value from line
        print(line.strip().split(','))
        fields = line.strip().split(',')
        # Testing if TODO refactor later
        if len(fields) == 3:
            
            break
        if len(fields) != 2:
            continue
        name, value = fields

        # On action line (or at end)
        if name == 'action' or name == 'end':
            # For previous action
            if action_name is not None:
                # Now we know for sure whether a path was provided
                if curr_path_id == last_path_id:
                    output += do_action(action_name, None)
                else:
                    paths.append(curr_path)
                    output += do_action(action_name, curr_path_id)
                    last_path_id = curr_path_id

            # On new action
            if name == 'action':
                # Update state with new action
                curr_path = []
                action_name = value

        # On X position line
        elif name == 'x_pos':
            # A path is started, make sure that's noted
            if curr_path_id == last_path_id:
                curr_path_id += 1

            # Save the X coordinate
            curr_x = float(value)

        # On Y position line
        elif name == 'y_pos':
            # We have both coordinates now
            curr_path.append((curr_x, float(value)))
        
        # On Go action, TODO development for path_output.txt having no action commands yets
        else:
            print(line)
    # End the run() method
    output += end_run_method()

    # Generate lines defining the paths
    for idx, path in enumerate(paths):
        path_id = idx + 1
        output += create_path(path_id, path)

    # Generate the end of the file
    output += end_file()

    # Write output string to file
    with open(args.output, 'w') as f:
        f.write(output)

    # Exit successfully
    print('Translation complete')
    exit(0)