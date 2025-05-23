from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import random
from math import sqrt, exp  # Added missing imports if needed
import time

# =============================================================================
# Obstacle Definitions
# =============================================================================
class Obstacle:
    def __init__(self, x, y, r, i):
        self.x = x
        self.y = y
        self.radius = r
        self.ignore_collision = i

class FirstStateIndex:
    def __init__(self, n):
        self.px = 0
        self.py = self.px + n
        self.vx = self.py + n
        self.vy = self.vx + n - 1
        self.dt = self.vy + n - 1

INCHES_PER_FIELD = 144

# =============================================================================
# PathPlanner Class Definitions
# =============================================================================
class PathPlanner:
    # All values are based on a scale from 0 to 1, where 1 is the length of the field
    def __init__(self, robot_length, robot_width, buffer_radius, max_velocity, max_accel):
        # -------------------------------------------------------------------------
        # Initialization: Robot and field parameters
        # -------------------------------------------------------------------------

        self.NUM_OF_ACTS = 2   # Number of MPC actions (vx, vy)
        self.NUM_OF_STATES = 2  # Number of MPC states (px, py)

        default_steps = 30

        self.initialize(default_steps)

        self.robot_length = robot_length
        self.robot_width = robot_width
        self.buffer_radius = buffer_radius
        self.robot_radius = sqrt(self.robot_length**2 + self.robot_width**2) / 2

        self.max_velocity = max_velocity
        self.max_accel = max_accel

        self.center_circle_radius = 1/6+3.5/INCHES_PER_FIELD # Circle surrounding the center obstacles of the field
    
    def initialize(self, num_steps):
        self.num_steps = max(num_steps, 3)
        self.initial_time_step = 0.1

        self.max_time = 30 # Maximum time for the path
        self.min_time = 0  # Minimum time for the path
        self.time_step_min = self.min_time/self.num_steps  # Minimum time step
        self.time_step_max = self.max_time/self.num_steps  # Maximum time step


    def intersects(self, x1, y1, x2, y2, r):
        if(x1**2 + y1**2 < r**2 or x2**2 + y2**2 < r**2):
            return False  # Prevent error when start or end is inside the circle
        if x1 == x2:
            return abs(x1) < r  # Check if the vertical line intersects the circle
        m = (y2 - y1) / (x2 - x1)
        return 4*((m**2+1)*r**2 - (y1-m*x1)**2) > 0

    def get_initial_path(self, x1, y1, x2, y2, r):
        # -------------------------------------------------------------------------
        # Path Initialization: Compute initial guess for the path
        # -------------------------------------------------------------------------
        x1 = x1 - 0.5; y1 = y1 - 0.5; x2 = x2 - 0.5; y2 = y2 - 0.5
        if self.intersects(x1, y1, x2, y2, r):
            start1 = 2*np.arctan((y1-sqrt(-r**2+x1**2+y1**2))/(x1+r))
            start2 = 2*np.arctan((y1+sqrt(-r**2+x1**2+y1**2))/(x1+r))
            end1 = 2*np.arctan((y2-sqrt(-r**2+x2**2+y2**2))/(x2+r))
            end2 = 2*np.arctan((y2+sqrt(-r**2+x2**2+y2**2))/(x2+r))
            x3 = r*np.cos(start1); y3 = r*np.sin(start1)
            x4 = r*np.cos(start2); y4 = r*np.sin(start2)
            x5 = r*np.cos(end1);   y5 = r*np.sin(end1)
            x6 = r*np.cos(end2);   y6 = r*np.sin(end2)
            m3 = (y3-y1)/(x3-x1)
            m4 = (y4-y1)/(x4-x1)
            m5 = (y5-y2)/(x5-x2)
            m6 = (y6-y2)/(x6-x2)
            x7 = (m3*x1 - m6*x2 - y1 + y2)/(m3-m6)
            y7 = (m3*(m6*(x1-x2)+y2)-m6*y1)/(m3-m6)
            x8 = (m4*x1 - m5*x2 - y1 + y2)/(m4-m5)
            y8 = (m4*(m5*(x1-x2)+y2)-m5*y1)/(m4-m5)
            d1 = sqrt((x7-x1)**2 + (y7-y1)**2) + sqrt((x7-x2)**2 + (y7-y2)**2)
            d2 = sqrt((x8-x1)**2 + (y8-y1)**2) + sqrt((x8-x2)**2 + (y8-y2)**2)
            if d1 < d2:
                init_x = np.linspace(x1+0.5, x7+0.5, self.num_steps//2)
                init_y = np.linspace(y1+0.5, y7+0.5, self.num_steps//2)
                init_x2 = np.linspace(x7+0.5, x2+0.5, self.num_steps - self.num_steps//2)
                init_y2 = np.linspace(y7+0.5, y2+0.5, self.num_steps - self.num_steps//2)
            else:
                init_x = np.linspace(x1+0.5, x8+0.5, self.num_steps//2)
                init_y = np.linspace(y1+0.5, y8+0.5, self.num_steps//2)
                init_x2 = np.linspace(x8+0.5, x2+0.5, self.num_steps - self.num_steps//2)
                init_y2 = np.linspace(y8+0.5, y2+0.5, self.num_steps - self.num_steps//2)
            init_x = np.concatenate((init_x, init_x2))
            init_y = np.concatenate((init_y, init_y2))
        else:
            init_x = np.linspace(x1+0.5, x2+0.5, self.num_steps)
            init_y = np.linspace(y1+0.5, y2+0.5, self.num_steps)

        # # Add random perturbations to the initial path to encourage exploration
        # perturbation_scale = 0.1  # Adjust the scale of perturbations as needed
        # init_x += np.random.uniform(-perturbation_scale, perturbation_scale, self.num_steps)
        # init_y += np.random.uniform(-perturbation_scale, perturbation_scale, self.num_steps)
    
        return (init_x, init_y)

    def Solve(self, start_point, end_point, obstacles):
        start_time = time.time()

        inches_per_step = 6
        inches = int(np.linalg.norm(np.array(start_point) - np.array(end_point)) * INCHES_PER_FIELD)
        steps = int(inches / inches_per_step)
        self.initialize(steps)

        # Ensure start_point and end_point are numpy arrays of type float64
        start_point = np.array(start_point, dtype=np.float64)
        end_point = np.array(end_point, dtype=np.float64)
        # -------------------------------------------------------------------------
        # Initialization: Define indexes and problem dimensions
        # -------------------------------------------------------------------------
        self.indexes = FirstStateIndex(self.num_steps)
        self.num_of_x_ = (self.num_steps)*self.NUM_OF_STATES + (self.num_steps - 1)*self.NUM_OF_ACTS + 1  # plus one for time step variable
        self.num_of_g_ = (self.num_steps)*len(obstacles)  + (self.num_steps-1)*(self.NUM_OF_ACTS+1) + (self.num_steps - 2)

        # -------------------------------------------------------------------------
        # Solve Optimization Problem: Set up variables, constraints, and solve NLP
        # -------------------------------------------------------------------------
        x = SX.sym('x', self.num_of_x_)
        self.indexes.dt = self.num_of_x_ - 1

        w_time_step = 100.0  # Cost weight on time step
        w_obstacle_penalty = 1000.0  # Weight for obstacle penalty
        cost = 0.0

        # Perform A* search to get an initial path
        a_star_path = self.a_star_search(start_point, end_point, obstacles)
        if a_star_path:
            # Interpolate the A* path to match the number of steps
            a_star_path = np.array(a_star_path)
            init_x = np.interp(np.linspace(0, 1, self.num_steps), np.linspace(0, 1, len(a_star_path)), a_star_path[:, 0])
            init_y = np.interp(np.linspace(0, 1, self.num_steps), np.linspace(0, 1, len(a_star_path)), a_star_path[:, 1])
        else:
            # Fall back to the default initial path if A* fails
            init_x, init_y = self.get_initial_path(start_point[0], start_point[1], end_point[0], end_point[1], self.center_circle_radius)

        self.init_x = init_x  # For plotting
        self.init_y = init_y

        # Build initial guess using the A* path and zero velocity
        init_v = [self.max_velocity / 2] * ((self.num_steps - 1) * self.NUM_OF_ACTS)
        init_time_step = self.initial_time_step
        x_ = np.concatenate((init_x, init_y, init_v, [init_time_step]))

        time_step = x[self.indexes.dt]
        cost += w_time_step * time_step * self.num_steps

        # Add obstacle penalty to the cost function
        for i in range(self.num_steps):
            curr_px_index = i + self.indexes.px
            curr_py_index = i + self.indexes.py
            curr_px = x[curr_px_index]
            curr_py = x[curr_py_index]
            for obstacle in obstacles:
                distance_squared = (curr_px - obstacle.x)**2 + (curr_py - obstacle.y)**2
                penalty = w_obstacle_penalty / (distance_squared + 1e-6)  # Add a small value to avoid division by zero
                cost += penalty/(1e4) #if not obstacle.ignore_collision else 0

        # Define variable bounds
        x_lowerbound_ = [-exp(10)] * self.num_of_x_
        x_upperbound_ = [exp(10)] * self.num_of_x_

        for i in range(self.indexes.px, self.indexes.py + self.num_steps):
            x_lowerbound_[i] = 0 #+ self.robot_radius
            x_upperbound_[i] = 1 #- self.robot_radius
        for i in range(self.indexes.vx, self.indexes.vy + self.num_steps - 1):
            x_lowerbound_[i] = -self.max_velocity
            x_upperbound_[i] = self.max_velocity

        # Constrain start and final positions
        x_lowerbound_[self.indexes.px] = start_point[0]
        x_lowerbound_[self.indexes.py] = start_point[1]
        x_lowerbound_[self.indexes.px + self.num_steps - 1] = end_point[0]
        x_lowerbound_[self.indexes.py + self.num_steps - 1] = end_point[1]
        x_upperbound_[self.indexes.px] = start_point[0]
        x_upperbound_[self.indexes.py] = start_point[1]
        x_upperbound_[self.indexes.px + self.num_steps - 1] = end_point[0]
        x_upperbound_[self.indexes.py + self.num_steps - 1] = end_point[1]

        # Constrain start and final velocities
        # x_lowerbound_[self.indexes.vx] = 0
        # x_lowerbound_[self.indexes.vy] = 0
        # x_upperbound_[self.indexes.vx] = 0
        # x_upperbound_[self.indexes.vy] = 0

        # x_lowerbound_[self.indexes.vx + self.num_steps - 2] = 0
        # x_lowerbound_[self.indexes.vy + self.num_steps - 2] = 0
        # x_upperbound_[self.indexes.vx + self.num_steps - 2] = 0
        # x_upperbound_[self.indexes.vy + self.num_steps - 2] = 0

        # Constrain time step
        x_lowerbound_[self.indexes.dt] = self.time_step_min
        x_upperbound_[self.indexes.dt] = self.time_step_max

        # Define constraint bounds
        g_lowerbound_ = [exp(-10)] * self.num_of_g_
        g_upperbound_ = [exp(10)] * self.num_of_g_

        g = [SX(0)] * self.num_of_g_
        g_index = 0

        # Speed constraints
        for i in range(self.num_steps - 1):
            curr_vx_index = self.indexes.vx + i
            curr_vy_index = self.indexes.vy + i
            vx = x[curr_vx_index]
            vy = x[curr_vy_index]
            g[g_index] = vx**2 + vy**2
            g_lowerbound_[g_index] = 0
            g_upperbound_[g_index] = self.max_velocity**2
            g_index += 1
        
        # Acceleration constraints
        for i in range(self.num_steps - 2):
            curr_vx_index = self.indexes.vx + i
            curr_vy_index = self.indexes.vy + i
            next_vx_index = curr_vx_index + 1
            next_vy_index = curr_vy_index + 1
            ax = (x[next_vx_index] - x[curr_vx_index]) / time_step
            ay = (x[next_vy_index] - x[curr_vy_index]) / time_step
            g[g_index] = ax**2 + ay**2
            g_lowerbound_[g_index] = 0
            g_upperbound_[g_index] = self.max_accel**2
            g_index += 1

        # Dynamics (position update) constraints
        for i in range(self.num_steps - 1):
            curr_px_index = i + self.indexes.px
            curr_py_index = i + self.indexes.py
            curr_vx_index = i + self.indexes.vx
            curr_vy_index = i + self.indexes.vy
            curr_px = x[curr_px_index]
            curr_py = x[curr_py_index]
            curr_vx = x[curr_vx_index]
            curr_vy = x[curr_vy_index]
            next_px = x[1 + curr_px_index]
            next_py = x[1 + curr_py_index]
            next_m_px = curr_px + curr_vx * time_step
            next_m_py = curr_py + curr_vy * time_step
            g[g_index] = next_px - next_m_px
            g_lowerbound_[g_index] = 0; g_upperbound_[g_index] = 0
            g_index += 1
            g[g_index] = next_py - next_m_py
            g_lowerbound_[g_index] = 0; g_upperbound_[g_index] = 0
            g_index += 1

        # Obstacle constraints
        for i in range(self.num_steps):
            curr_px_index = i + self.indexes.px
            curr_py_index = i + self.indexes.py
            curr_px = x[curr_px_index]
            curr_py = x[curr_py_index]
            for obstacle in obstacles:
                g[g_index] = (curr_px - obstacle.x)**2 + (curr_py - obstacle.y)**2
                if obstacle.ignore_collision:
                    g_lowerbound_[g_index] = exp(-10)
                else:
                    g_lowerbound_[g_index] = (obstacle.radius + self.robot_radius + self.buffer_radius)**2
                g_upperbound_[g_index] = exp(10)
                g_index += 1

        nlp = {'x': x, 'f': cost, 'g': vertcat(*g)}
        opts = {"ipopt.print_level": 0, "print_time": 0, 'ipopt.tol': 1e-6, "ipopt.sb": "yes"}
        solver = nlpsol('solver', 'ipopt', nlp, opts)
        res = solver(x0=x_, lbx=x_lowerbound_, ubx=x_upperbound_, lbg=g_lowerbound_, ubg=g_upperbound_)
        solver_stats = solver.stats()
        self.status = solver_stats['return_status']
        
        self.solve_time = time.time() - start_time

        return res

    def print_trajectory_details(self, res, save_path):
        # -------------------------------------------------------------------------
        # Trajectory Output: Print details and save to a file
        # -------------------------------------------------------------------------
        x_opt = res['x'].full().flatten()
        final_cost = res['f'].full().item()
        print(f"{'Step':<5} {'Position (x, y)':<20}\t{'Velocity (vx, vy)':<20}\t{'Acceleration (ax, ay)':<25}")
        print("-" * 70)
        lemlib_output_string = ""
        optimized_time_step  = x_opt[self.num_of_x_-1]
        for i in range(self.num_steps):
            px = x_opt[self.indexes.px + i]
            py = x_opt[self.indexes.py + i]
            if i < self.num_steps - 1:
                vx = x_opt[self.indexes.vx + i]
                vy = x_opt[self.indexes.vy + i]
            else:
                vx = vy = 0
            if i < self.num_steps - 2:
                next_vx = x_opt[self.indexes.vx + i + 1]
                next_vy = x_opt[self.indexes.vy + i + 1]
                ax = (next_vx - vx) / optimized_time_step 
                ay = (next_vy - vy) / optimized_time_step 
            else:
                ax = ay = 0
            print(f"{i:<5} ({px*INCHES_PER_FIELD-72:.2f}, {py*INCHES_PER_FIELD-72:.2f})\t\t({vx*INCHES_PER_FIELD:.2f}, {vy*INCHES_PER_FIELD:.2f})\t\t({ax*INCHES_PER_FIELD:.2f}, {ay*INCHES_PER_FIELD:.2f})")
            lemlib_output_string += f"{px*INCHES_PER_FIELD-72:.3f}, {py*INCHES_PER_FIELD-72:.3f}\n"
        print(f"\nFinal cost: {final_cost:.2f}")
        print(f"\nTime step: {optimized_time_step:.2f}")
        print(f"Path time: {optimized_time_step * self.num_steps:.2f}")
        print(f"\nStatus: {self.status}")
        print(f"Solve time: {self.solve_time:.3f} seconds")
        lemlib_output_string += "endData"
        if save_path:
            with open(save_path, 'w') as file:
                file.write(lemlib_output_string)

    def plotResults(self, sol):
        # -------------------------------------------------------------------------
        # Plot Results: Display trajectory, obstacles, and robot boundaries
        # -------------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 8))
        planned_px = np.array(sol['x'][self.indexes.px:self.indexes.py]).flatten()
        planned_py = np.array(sol['x'][self.indexes.py:self.indexes.vx]).flatten()
        planned_vx = np.array(sol['x'][self.indexes.vx:self.indexes.vy]).flatten()
        planned_vy = np.array(sol['x'][self.indexes.vy:self.indexes.dt]).flatten()
        planned_theta = np.arctan2(planned_vy, planned_vx)
        planned_theta = np.concatenate(([planned_theta[1]], planned_theta[1:-1], [planned_theta[-2], planned_theta[-2]]))
        ax.plot(self.init_x, self.init_y, linestyle=':', color='gray', alpha=0.7, label='initial path')
        ax.plot(planned_px, planned_py, '-o', label='path', color="blue", alpha=0.5)
        theta_list = np.linspace(0, 2 * np.pi, 100)
        num_outlines = 3
        mod = round(self.num_steps / (num_outlines - 1))
        index = 0
        for px, py, theta in zip(planned_px, planned_py, planned_theta):
            rotation = Affine2D().rotate_around(px, py, theta)
            rectangle = plt.Rectangle((px - self.robot_length / 2, py - self.robot_width / 2), self.robot_length, self.robot_width,
                                      edgecolor='blue', facecolor='none', alpha=1)
            rectangle.set_transform(rotation + ax.transData)
            if index % mod == 0 or index == self.num_steps - 1:
                robot_circle_x = px + self.robot_radius * np.cos(theta_list)
                robot_circle_y = py + self.robot_radius * np.sin(theta_list)
                ax.plot(robot_circle_x, robot_circle_y, '--', color='blue', alpha=0.5, label='robot radius' if index == 0 else None)
                ax.add_patch(rectangle)
            index += 1
        ax.plot(start_point[0], start_point[1], 'o', color='orange', label='start')
        ax.plot(end_point[0], end_point[1], 'o', color='green', label='target')
        first_obstacle = True
        for obstacle in obstacles:
            danger_x = obstacle.x + (obstacle.radius - 0.005) * np.cos(theta_list)
            danger_y = obstacle.y + (obstacle.radius - 0.005) * np.sin(theta_list)
            buffer_x = obstacle.x + (obstacle.radius + self.buffer_radius) * np.cos(theta_list)
            buffer_y = obstacle.y + (obstacle.radius + self.buffer_radius) * np.sin(theta_list)
            if first_obstacle:
                ax.plot(danger_x, danger_y, 'r-', label='obstacle')
                ax.plot(buffer_x, buffer_y, 'r--', label='buffer zone', alpha=0.5)
                first_obstacle = False
            else:
                ax.plot(danger_x, danger_y, 'r-')
                ax.plot(buffer_x, buffer_y, 'r--', alpha=0.5)
        radius = self.center_circle_radius; center_x, center_y = 0.5, 0.5
        circle_x = center_x + radius * np.cos(theta_list)
        circle_y = center_y + radius * np.sin(theta_list)
        ax.plot(circle_x, circle_y, 'r--', alpha=0.25)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0., frameon=False)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid()
        plt.savefig('path.png')
        plt.close(fig)  # Close the figure to free memory

        print("Path saved to path.png")

        
    
    def getPath(self, sol):
        planned_px = np.array(sol['x'][self.indexes.px:self.indexes.py]).flatten()
        planned_py = np.array(sol['x'][self.indexes.py:self.indexes.vx]).flatten()
        total_path_time = sol['x'][self.indexes.dt] * self.num_steps
        return planned_px, planned_py, total_path_time

    def a_star_search(self, start_point, end_point, obstacles):
        """
        Perform A* search on a 48x48 grid.
        """
        grid_size = 48
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Block grid cells based on obstacles
        for obstacle in obstacles:
            cx, cy = int(obstacle.x * grid_size), int(obstacle.y * grid_size)
            radius = int((obstacle.radius + self.robot_radius + self.buffer_radius) * grid_size)
            for x in range(max(0, cx - radius), min(grid_size, cx + radius + 1)):
                for y in range(max(0, cy - radius), min(grid_size, cy + radius + 1)):
                    if (x - cx)**2 + (y - cy)**2 <= radius**2:
                        grid[x, y] = 1  # Mark as blocked

        # Convert start and end points to grid coordinates
        start = (int(start_point[0] * grid_size), int(start_point[1] * grid_size))
        end = (int(end_point[0] * grid_size), int(end_point[1] * grid_size))

        # A* search
        from heapq import heappop, heappush
        open_set = []
        heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, end)}

        while open_set:
            _, current = heappop(open_set)

            if current == end:
                return self.reconstruct_path(came_from, current, grid_size)

            for neighbor in self.get_neighbors(current, grid_size):
                if grid[neighbor[0], neighbor[1]] == 1:  # Skip blocked cells
                    continue

                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, end)
                    heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def heuristic(self, a, b):
        # """
        # Heuristic function for A* (Manhattan distance).
        # """
        # return abs(a[0] - b[0]) + abs(a[1] - b[1])
        return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)  # Euclidean distance

    def get_neighbors(self, node, grid_size):
        """
        Get valid neighbors for a node in the grid.
        """
        x, y = node
        neighbors = [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        return [(nx, ny) for nx, ny in neighbors if 0 <= nx < grid_size and 0 <= ny < grid_size]

    def reconstruct_path(self, came_from, current, grid_size):
        """
        Reconstruct the path from the A* search.
        """
        path = []
        while current in came_from:
            path.append((current[0] / grid_size, current[1] / grid_size))
            current = came_from[current]
        path.append((current[0] / grid_size, current[1] / grid_size))
        return path[::-1]

    def generate_grid_png(self, obstacles, start_point, end_point, filename="grid.png"):
        """
        Generate a PNG of the grid based on obstacles, swap black/white, and add dots for start and end points.
        """
        grid_size = 48
        grid = np.ones((grid_size, grid_size), dtype=int)  # Initialize grid with 1s (white)

        # Block grid cells based on obstacles
        for obstacle in obstacles:
            cx, cy = int(obstacle.x * grid_size), int(obstacle.y * grid_size)
            radius = int((obstacle.radius + self.robot_radius + self.buffer_radius) * grid_size)
            for x in range(max(0, cx - radius), min(grid_size, cx + radius + 1)):
                for y in range(max(0, cy - radius), min(grid_size, cy + radius + 1)):
                    if (x - cx)**2 + (y - cy)**2 <= radius**2:
                        grid[y, x] = 0  # Mark as blocked (black)

        # Convert start and end points to grid coordinates
        start = (int(start_point[1] * grid_size), int(start_point[0] * grid_size))
        end = (int(end_point[1] * grid_size), int(end_point[0] * grid_size))

        # Plot the grid
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(grid, cmap="gray", origin="lower")  # Use 'upper' to fix flipping
        ax.set_title("Grid Visualization")
        ax.set_xticks([]); ax.set_yticks([])

        # Add dots for start and end points
        ax.plot(start[1], start[0], 'go', label="Start")  # Green dot for start
        ax.plot(end[1], end[0], 'ro', label="End")  # Red dot for end
        ax.legend(loc="upper right")

        plt.tight_layout()

        # Save the grid as a PNG
        plt.savefig(filename)
        plt.close(fig)
        print(f"Grid PNG saved as {filename}")

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    robot_length = 15/INCHES_PER_FIELD
    robot_width = 15/INCHES_PER_FIELD
    buffer_radius = 2/INCHES_PER_FIELD
    max_velocity = 70/INCHES_PER_FIELD
    max_accel = 70/INCHES_PER_FIELD

    planner = PathPlanner(robot_length, robot_width, buffer_radius, max_velocity, max_accel)

    total_solve_time = 0
    successful_solve_time = 0
    unsuccessful_solve_time = 0
    successful_trials = 0
    unsuccessful_trials = 0
    total_trials = 1000

    for i in range(total_trials):
        # start_point = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)]
        # end_point = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)]

        start_point = [0.45, 0.15]
        end_point = [0.50, 0.85]


        obstacles = [Obstacle(3/6, 2/6, 3.5/INCHES_PER_FIELD, False),
                     Obstacle(3/6, 4/6, 3.5/INCHES_PER_FIELD, False),
                     Obstacle(2/6, 3/6, 3.5/INCHES_PER_FIELD, False),
                     Obstacle(4/6, 3/6, 3.5/INCHES_PER_FIELD, False)]
        radius = 5.75 / INCHES_PER_FIELD
        # obstacles.append(Obstacle(.1, .4, radius, True))
        # obstacles.append(Obstacle(.2, .9, radius, True))
        # obstacles.append(Obstacle(.4, .7, radius, True))
        # obstacles.append(Obstacle(.7, .8, radius, True))
        # obstacles.append(Obstacle(.8, .4, radius, True))   

        for i in range(5):
            while True:
                x = random.uniform(0.1, 0.9)
                y = random.uniform(0.1, 0.9)
                center_x, center_y = 0.5, 0.5
                center_radius = 1/6 + 3.5 / INCHES_PER_FIELD

                # Ensure the obstacle is outside the center circle
                if (x - center_x)**2 + (y - center_y)**2 > center_radius**2:
                    # Ensure the obstacle does not touch any other obstacle
                    if all((x - obs.x)**2 + (y - obs.y)**2 > (radius + obs.radius)**2 for obs in obstacles):
                        obstacles.append(Obstacle(x, y, radius, True))
                        break

        sol = planner.Solve(start_point=start_point, end_point=end_point, obstacles=obstacles)

        if planner.status == 'Solve_Succeeded':
            successful_trials += 1
            successful_solve_time += planner.solve_time
        else:
            unsuccessful_trials += 1
            unsuccessful_solve_time += planner.solve_time

        total_solve_time += planner.solve_time

        planner.print_trajectory_details(sol, None)
        planner.plotResults(sol)
        planner.generate_grid_png(obstacles, start_point, end_point)

        input()

    print(f"Average solve time (successful): {successful_solve_time / successful_trials:.3f} seconds" if successful_trials > 0 else "No successful trials")
    print(f"Average solve time (unsuccessful): {unsuccessful_solve_time / unsuccessful_trials:.3f} seconds" if unsuccessful_trials > 0 else "No unsuccessful trials")
    print(f"Average solve time (overall): {total_solve_time / total_trials:.3f} seconds")
    print(f"Successful trials: {successful_trials}")
    print(f"Success rate: {successful_trials / total_trials * 100:.2f}%")