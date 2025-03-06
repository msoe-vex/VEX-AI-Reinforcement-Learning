import csv
import re

# Regular expressions to match commands and coordinates
command_pattern = re.compile(r'(\w+),?\s*([\d\s,.-]*)')
coordinate_pattern = re.compile(r'([\d.-]+),([\d.-]+)')

# Function to convert the input commands to the desired format
def convert_to_class_format(csv_file_path):
    output = []
    bezier_curves = []
    bezier_index = 0

    # Start the class definition
    output.append("class ConvertedClass {")
    output.append("    public:")
    output.append("    static void run() {")

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

# Convert the input commands from the CSV file
csv_file_path = 'run_agent_results/auton.csv'
converted_code = convert_to_class_format(csv_file_path)
print(converted_code)
