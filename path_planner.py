from casadi import *
import numpy as np
from math import sqrt, exp  # Added missing imports if needed
from collections import deque
import time
import os
from vex_core.base_game import Robot, Team, RobotSize

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
GRID_SIZE = 128

# =============================================================================
# PathPlanner Class Definitions
# =============================================================================
class PathPlanner:
    # All values are based on a scale from 0 to 1, where 1 is the length of the field
    def __init__(self, field_size_inches=144, field_center=(0, 0), output_dir="path_planner"):
        # -------------------------------------------------------------------------
        # Initialization: Field parameters
        # -------------------------------------------------------------------------

        self.NUM_OF_ACTS = 2   # Number of MPC actions (vx, vy)
        self.NUM_OF_STATES = 2  # Number of MPC states (px, py)

        default_steps = 30

        self.initialize(default_steps)

        # Field coordinate system parameters
        self.field_size_inches = field_size_inches
        self.field_center = np.array(field_center, dtype=np.float64)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def inches_to_normalized(self, pos_inches):
        """Convert position from inches (field coordinates) to normalized scale (0-1)."""
        if isinstance(pos_inches, np.ndarray):
            return (pos_inches - self.field_center + self.field_size_inches / 2) / self.field_size_inches
        pos_array = np.array(pos_inches, dtype=np.float64)
        return (pos_array - self.field_center + self.field_size_inches / 2) / self.field_size_inches
    
    def normalized_to_inches(self, pos_normalized):
        """Convert position from normalized scale (0-1) to inches (field coordinates)."""
        if isinstance(pos_normalized, np.ndarray):
            return pos_normalized * self.field_size_inches - self.field_size_inches / 2 + self.field_center
        pos_array = np.array(pos_normalized, dtype=np.float64)
        return pos_array * self.field_size_inches - self.field_size_inches / 2 + self.field_center
    
    def initialize(self, num_steps):
        self.num_steps = max(num_steps, 3)
        self.initial_time_step = 0.1

        self.max_time = 30 # Maximum time for the path
        self.min_time = 0  # Minimum time for the path
        self.time_step_min = self.min_time/self.num_steps  # Minimum time step
        self.time_step_max = self.max_time/self.num_steps  # Maximum time step


    def get_initial_path(self, x1, y1, x2, y2):
        # -------------------------------------------------------------------------
        # Path Initialization: Compute initial guess for the path
        # -------------------------------------------------------------------------
        x1 = x1 - 0.5; y1 = y1 - 0.5; x2 = x2 - 0.5; y2 = y2 - 0.5
        init_x = np.linspace(x1+0.5, x2+0.5, self.num_steps)
        init_y = np.linspace(y1+0.5, y2+0.5, self.num_steps)

        # # Add random perturbations to the initial path to encourage exploration
        # perturbation_scale = 0.1  # Adjust the scale of perturbations as needed
        # init_x += np.random.uniform(-perturbation_scale, perturbation_scale, self.num_steps)
        # init_y += np.random.uniform(-perturbation_scale, perturbation_scale, self.num_steps)
    
        return (init_x, init_y)

    def Solve(self, start_point, end_point, obstacles, robot: Robot, optimize=True):
        """Solve for optimal path. Accepts all inputs in field inches and returns positions in field inches.
        
        Args:
            start_point: Starting position in field inches
            end_point: Ending position in field inches
            obstacles: List of obstacles
            robot: Robot object with dimensions and constraints
            optimize: When False, skip NLP and return BFS+A*-based path
        """
        # Calculate robot parameters locally
        robot_radius_norm, buffer_radius_norm, total_radius_norm = self._get_robot_norms(robot)
        
        max_velocity_norm = robot.max_speed / self.field_size_inches
        max_accel_norm = robot.max_acceleration / self.field_size_inches

        start_time = time.time()

        # Convert input positions from inches to normalized coordinates
        start_point_inches = np.array(start_point, dtype=np.float64)
        start_point_norm = self.inches_to_normalized(start_point_inches)
        end_point_inches = np.array(end_point, dtype=np.float64)
        end_point_norm = self.inches_to_normalized(end_point_inches)

        # Convert obstacles to normalized coordinates if they appear to be in inches
        normalized_obstacles = self._normalize_obstacles(obstacles)

        # Snap start/end to nearest valid grid cells if either is invalid
        planning_start_norm, planning_end_norm = self._snap_endpoints_to_valid_cells(
            start_point_norm,
            end_point_norm,
            normalized_obstacles,
            total_radius_norm,
            GRID_SIZE,
        )


        # Calculate steps based on distance in inches
        inches_per_step = 6
        inches = int(np.linalg.norm(end_point_inches - start_point_inches))
        steps = int(inches / inches_per_step)
        self.initialize(steps)
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

        # Perform A* search to get an initial path (using normalized coordinates)
        a_star_path = self.a_star_search(planning_start_norm, planning_end_norm, normalized_obstacles, total_radius_norm)
        if a_star_path:
            # Interpolate the A* path to match the number of steps
            a_star_path = np.array(a_star_path)
            init_x = np.interp(np.linspace(0, 1, self.num_steps), np.linspace(0, 1, len(a_star_path)), a_star_path[:, 0])
            init_y = np.interp(np.linspace(0, 1, self.num_steps), np.linspace(0, 1, len(a_star_path)), a_star_path[:, 1])
        else:
            # Fall back to the default initial path if A* fails
            init_x, init_y = self.get_initial_path(planning_start_norm[0], planning_start_norm[1], planning_end_norm[0], planning_end_norm[1])

        self.init_x = init_x  # For plotting
        self.init_y = init_y

        if not optimize:
            positions_normalized = self._build_fallback_positions_normalized(
                a_star_path,
                planning_start_norm,
                planning_end_norm,
            )
            dt = self.initial_time_step
            positions_inches, velocities_inches = self._positions_normalized_to_trajectory(positions_normalized, dt)
            self.optimizer_status = 'NLP_Disabled'
            self.status = 'AStar_Only_Succeeded'
            self.solve_time = time.time() - start_time

            positions_inches, velocities_inches = self._apply_start_end_connectors(
                positions_inches,
                velocities_inches,
                dt,
                start_point_inches,
                end_point_inches,
                start_point_norm,
                end_point_norm,
                planning_start_norm,
                planning_end_norm,
            )

            return positions_inches, velocities_inches, dt

        # Build initial guess using the A* path and zero velocity
        init_v = [max_velocity_norm / 2] * ((self.num_steps - 1) * self.NUM_OF_ACTS)
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
            for obstacle in normalized_obstacles:
                distance_squared = (curr_px - obstacle.x)**2 + (curr_py - obstacle.y)**2
                penalty = w_obstacle_penalty / (distance_squared + 1e-6)  # Add a small value to avoid division by zero
                cost += penalty/(1e4) #if not obstacle.ignore_collision else 0

        # Define variable bounds
        x_lowerbound_ = [-exp(10)] * self.num_of_x_
        x_upperbound_ = [exp(10)] * self.num_of_x_
        for i in range(self.indexes.px, self.indexes.py + self.num_steps):
            x_lowerbound_[i] = total_radius_norm
            x_upperbound_[i] = 1 - total_radius_norm
        for i in range(self.indexes.vx, self.indexes.vy + self.num_steps - 1):
            x_lowerbound_[i] = -max_velocity_norm
            x_upperbound_[i] = max_velocity_norm

        # Constrain start and final positions (using normalized coordinates)
        x_lowerbound_[self.indexes.px] = planning_start_norm[0]
        x_lowerbound_[self.indexes.py] = planning_start_norm[1]
        x_lowerbound_[self.indexes.px + self.num_steps - 1] = planning_end_norm[0]
        x_lowerbound_[self.indexes.py + self.num_steps - 1] = planning_end_norm[1]
        x_upperbound_[self.indexes.px] = planning_start_norm[0]
        x_upperbound_[self.indexes.py] = planning_start_norm[1]
        x_upperbound_[self.indexes.px + self.num_steps - 1] = planning_end_norm[0]
        x_upperbound_[self.indexes.py + self.num_steps - 1] = planning_end_norm[1]

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
            g_upperbound_[g_index] = max_velocity_norm**2
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
            g_upperbound_[g_index] = max_accel_norm**2
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
            for obstacle in normalized_obstacles:
                g[g_index] = (curr_px - obstacle.x)**2 + (curr_py - obstacle.y)**2
                if obstacle.ignore_collision:
                    g_lowerbound_[g_index] = exp(-10)
                else:
                    g_lowerbound_[g_index] = (obstacle.radius + total_radius_norm)**2
                g_upperbound_[g_index] = exp(10)
                g_index += 1

        nlp = {'x': x, 'f': cost, 'g': vertcat(*g)}
        opts = {"ipopt.print_level": 0, "print_time": 0, 'ipopt.tol': 1e-6, "ipopt.sb": "yes"}
        solver = nlpsol('solver', 'ipopt', nlp, opts)
        res = solver(x0=x_, lbx=x_lowerbound_, ubx=x_upperbound_, lbg=g_lowerbound_, ubg=g_upperbound_)
        solver_stats = solver.stats()
        self.optimizer_status = solver_stats['return_status']
        self.status = self.optimizer_status
        
        self.solve_time = time.time() - start_time

        # Extract solution components
        if self.optimizer_status == 'Solve_Succeeded':
            x_opt = res['x'].full().flatten()
            pos_x = x_opt[self.indexes.px:self.indexes.py]
            pos_y = x_opt[self.indexes.py:self.indexes.vx]
            vel_x = x_opt[self.indexes.vx:self.indexes.vy]
            vel_y = x_opt[self.indexes.vy:self.indexes.dt]
            dt = x_opt[self.indexes.dt]

            # Convert positions from normalized coordinates back to inches
            positions_normalized = np.column_stack((pos_x, pos_y))
            positions_inches = np.array([self.normalized_to_inches(pos) for pos in positions_normalized])

            # Convert velocities from normalized units back to inches per second
            velocities_normalized = np.column_stack((vel_x, vel_y))
            velocities_inches = velocities_normalized * self.field_size_inches
        else:
            positions_normalized = self._build_fallback_positions_normalized(
                a_star_path,
                planning_start_norm,
                planning_end_norm,
            )
            dt = self.initial_time_step
            positions_inches, velocities_inches = self._positions_normalized_to_trajectory(positions_normalized, dt)
            self.status = 'Grid_Fallback_Succeeded'

        positions_inches, velocities_inches = self._apply_start_end_connectors(
            positions_inches,
            velocities_inches,
            dt,
            start_point_inches,
            end_point_inches,
            start_point_norm,
            end_point_norm,
            planning_start_norm,
            planning_end_norm,
        )
        
        return positions_inches, velocities_inches, dt

    def print_trajectory_details(self, positions, velocities, dt, save_path=None):
        # -------------------------------------------------------------------------
        # Trajectory Output: Print details and save to a file
        # Positions are already in inches
        # -------------------------------------------------------------------------
        print(f"{'Step':<5} {'Position (x, y)':<20}\t{'Velocity (vx, vy)':<20}\t{'Acceleration (ax, ay)':<25}")
        print("-" * 70)
        lemlib_output_string = ""
        
        for i in range(len(positions)):
            px, py = positions[i]  # Already in inches
            if i < len(velocities):
                vx, vy = velocities[i]
            else:
                vx = vy = 0
            
            if i < len(velocities) - 1:
                next_vx, next_vy = velocities[i + 1]
                ax = (next_vx - vx) / dt
                ay = (next_vy - vy) / dt
            else:
                ax = ay = 0
            
            print(f"{i:<5} ({px:.2f}, {py:.2f})\t\t({vx:.2f}, {vy:.2f})\t\t({ax:.2f}, {ay:.2f})")
            lemlib_output_string += f"{px:.3f}, {py:.3f}\n"
        
        print(f"\nTime step: {dt:.2f}")
        print(f"Path time: {dt * len(positions):.2f}")
        print(f"\nStatus: {self.status}")
        if hasattr(self, 'optimizer_status') and self.optimizer_status != self.status:
            print(f"Optimizer status: {self.optimizer_status}")
        print(f"Solve time: {self.solve_time:.3f} seconds")
        lemlib_output_string += "endData"
        if save_path:
            with open(save_path, 'w') as file:
                file.write(lemlib_output_string)

    def plotResults(self, positions, velocities, start_point, end_point, obstacles, robot: Robot):
        # -------------------------------------------------------------------------
        # Plot Results: Display trajectory, obstacles, and robot boundaries
        # All inputs are in inches, convert to normalized for plotting
        # -------------------------------------------------------------------------
        
        import matplotlib.pyplot as plt
        from matplotlib.transforms import Affine2D

        # Calculate local norms
        robot_length_norm = robot.length / self.field_size_inches
        robot_width_norm = robot.width / self.field_size_inches
        robot_radius_norm, buffer_radius_norm, _ = self._get_robot_norms(robot)

        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Convert positions from inches to normalized for plotting
        positions_norm = np.array([self.inches_to_normalized(pos) for pos in positions])
        planned_px = positions_norm[:, 0]
        planned_py = positions_norm[:, 1]
        planned_vx = velocities[:, 0]
        planned_vy = velocities[:, 1]
        planned_theta = np.arctan2(planned_vy, planned_vx)
        planned_theta = np.concatenate(([planned_theta[0]], planned_theta, [planned_theta[-1]]))
        
        # Convert start/end points to normalized
        start_norm, end_norm, pseudo_start_norm, pseudo_end_norm = self._get_marker_points_normalized(start_point, end_point)
        
        ax.plot(self.init_x, self.init_y, linestyle=':', color='gray', alpha=0.7, label='initial path')
        ax.plot(planned_px, planned_py, '-o', label='path', color="blue", alpha=0.5)
        theta_list = np.linspace(0, 2 * np.pi, 100)
        num_outlines = 3
        mod = round(self.num_steps / (num_outlines - 1))
        index = 0
        for px, py, theta in zip(planned_px, planned_py, planned_theta):
            rotation = Affine2D().rotate_around(px, py, theta)
            rectangle = plt.Rectangle((px - robot_length_norm / 2, py - robot_width_norm / 2), robot_length_norm, robot_width_norm,
                                      edgecolor='blue', facecolor='none', alpha=1)
            rectangle.set_transform(rotation + ax.transData)
            if index % mod == 0 or index == self.num_steps - 1:
                robot_circle_x = px + robot_radius_norm * np.cos(theta_list)
                robot_circle_y = py + robot_radius_norm * np.sin(theta_list)
                ax.plot(robot_circle_x, robot_circle_y, '--', color='blue', alpha=0.5, label='robot radius' if index == 0 else None)
                ax.add_patch(rectangle)
            index += 1
        ax.plot(start_norm[0], start_norm[1], '*', color='orange', markersize=12, label='start')
        ax.plot(end_norm[0], end_norm[1], '*', color='green', markersize=12, label='end')

        ax.plot(
            pseudo_start_norm[0],
            pseudo_start_norm[1],
            'o',
            markerfacecolor='none',
            markeredgecolor='orange',
            markersize=8,
            label='pseudo start',
        )
        ax.plot(
            pseudo_end_norm[0],
            pseudo_end_norm[1],
            'o',
            markerfacecolor='none',
            markeredgecolor='green',
            markersize=8,
            label='pseudo end',
        )
        first_obstacle = True
        for obstacle in obstacles:
            # Convert obstacle from inches to normalized
            obs_norm = self.inches_to_normalized(np.array([obstacle.x, obstacle.y]))
            radius_norm = obstacle.radius / self.field_size_inches
            danger_x = obs_norm[0] + (radius_norm - 0.005/self.field_size_inches) * np.cos(theta_list)
            danger_y = obs_norm[1] + (radius_norm - 0.005/self.field_size_inches) * np.sin(theta_list)
            buffer_x = obs_norm[0] + (radius_norm + buffer_radius_norm) * np.cos(theta_list)
            buffer_y = obs_norm[1] + (radius_norm + buffer_radius_norm) * np.sin(theta_list)
            if first_obstacle:
                ax.plot(danger_x, danger_y, 'r-', label='obstacle')
                ax.plot(buffer_x, buffer_y, 'r--', label='buffer zone', alpha=0.5)
                first_obstacle = False
            else:
                ax.plot(danger_x, danger_y, 'r-')
                ax.plot(buffer_x, buffer_y, 'r--', alpha=0.5)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0., frameon=False)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid()
        ax.set_xticks([1/6, 2/6, 3/6, 4/6, 5/6]); ax.set_yticks([1/6, 2/6, 3/6, 4/6, 5/6])
        plt.savefig(f'{self.output_dir}/path.png')
        plt.close(fig)  # Close the figure to free memory

        print(f"Path saved to {self.output_dir}/path.png")
        self.generate_grid_png(obstacles, start_point, end_point, robot)
        
    
    def getPath(self, positions, dt):
        """Legacy method for compatibility - returns separate x, y arrays and total time."""
        planned_px = positions[:, 0]
        planned_py = positions[:, 1]
        total_path_time = dt * len(positions)
        return planned_px, planned_py, total_path_time

    def _generate_obstacle_grid(self, obstacles, start_point, end_point, total_radius_norm, grid_size=GRID_SIZE):
        """
        Helper method to generate an obstacle grid for A* search and visualization.
        
        Args:
            obstacles: List of obstacles in normalized coordinates
            start_point: Start point in normalized coordinates
            end_point: End point in normalized coordinates
            total_radius_norm: Combined robot + buffer radius in normalized units
            grid_size: Size of the grid
            
        Returns:
            grid: 2D numpy array where 0=free, 1=blocked
            start: Grid coordinates (x, y) for start point
            end: Grid coordinates (x, y) for end point
        """
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Block grid cells based on obstacles
        for obstacle in obstacles:
            cx, cy = int(obstacle.x * grid_size), int(obstacle.y * grid_size)
            radius = int((obstacle.radius + total_radius_norm) * grid_size)
            for x in range(max(0, cx - radius), min(grid_size, cx + radius + 1)):
                for y in range(max(0, cy - radius), min(grid_size, cy + radius + 1)):
                    if (x - cx)**2 + (y - cy)**2 <= radius**2:
                        grid[y, x] = 1  # Mark as blocked (grid is [row=y, col=x])
        
        # Block any cell within robot radius of the field boundary
        robot_radius_cells = int(total_radius_norm * grid_size)
        for i in range(grid_size):
            for j in range(grid_size):
                if i < robot_radius_cells or i >= grid_size - robot_radius_cells or j < robot_radius_cells or j >= grid_size - robot_radius_cells:
                    grid[i, j] = 1  # Mark as blocked

        # Convert start and end points to grid coordinates
        start = self._normalized_to_grid(start_point, grid_size)
        end = self._normalized_to_grid(end_point, grid_size)
        
        return grid, start, end

    def _normalized_to_grid(self, point, grid_size=GRID_SIZE):
        x = int(np.clip(point[0] * grid_size, 0, grid_size - 1))
        y = int(np.clip(point[1] * grid_size, 0, grid_size - 1))
        return (x, y)

    def _grid_to_normalized(self, grid_point, grid_size=GRID_SIZE):
        return np.array([grid_point[0] / grid_size, grid_point[1] / grid_size], dtype=np.float64)

    def _is_grid_cell_valid(self, grid, point):
        return grid[point[1], point[0]] == 0

    def _find_nearest_valid_cell_bfs(self, grid, origin, validator=None):
        if self._is_grid_cell_valid(grid, origin) and (validator is None or validator(origin)):
            return origin

        grid_size = grid.shape[0]
        queue = deque([origin])
        visited = {origin}

        while queue:
            current = queue.popleft()
            for neighbor in self.get_neighbors(current, grid_size):
                if neighbor in visited:
                    continue
                if self._is_grid_cell_valid(grid, neighbor) and (validator is None or validator(neighbor)):
                    return neighbor
                visited.add(neighbor)
                queue.append(neighbor)

        return None

    def _is_normalized_point_valid_for_nlp(self, point_norm, obstacles, total_radius_norm):
        px, py = float(point_norm[0]), float(point_norm[1])

        # Keep robot center inside field with boundary margin.
        if px < total_radius_norm or px > 1.0 - total_radius_norm:
            return False
        if py < total_radius_norm or py > 1.0 - total_radius_norm:
            return False

        for obstacle in obstacles:
            if obstacle.ignore_collision:
                continue
            min_dist = obstacle.radius + total_radius_norm
            if (px - obstacle.x) ** 2 + (py - obstacle.y) ** 2 < min_dist ** 2:
                return False

        return True

    def _snap_endpoints_to_valid_cells(self, start_point, end_point, obstacles, total_radius_norm, grid_size=GRID_SIZE):
        grid, start, end = self._generate_obstacle_grid(
            obstacles,
            start_point,
            end_point,
            total_radius_norm,
            grid_size,
        )

        def nlp_validator(grid_point):
            point_norm = self._grid_to_normalized(grid_point, grid_size)
            return self._is_normalized_point_valid_for_nlp(
                point_norm,
                obstacles,
                total_radius_norm,
            )

        snapped_start = self._find_nearest_valid_cell_bfs(grid, start, validator=nlp_validator)
        snapped_end = self._find_nearest_valid_cell_bfs(grid, end, validator=nlp_validator)

        if snapped_start is None:
            snapped_start = start
        if snapped_end is None:
            snapped_end = end

        return self._grid_to_normalized(snapped_start, grid_size), self._grid_to_normalized(snapped_end, grid_size)

    def _build_straight_line_points(self, start_point_inches, end_point_inches, step_inches=6.0):
        start = np.array(start_point_inches, dtype=np.float64)
        end = np.array(end_point_inches, dtype=np.float64)
        distance = np.linalg.norm(end - start)

        if distance <= 1e-9:
            return np.array([start], dtype=np.float64)

        num_segments = max(1, int(np.ceil(distance / step_inches)))
        alpha = np.linspace(0.0, 1.0, num_segments + 1)
        return np.array([start + a * (end - start) for a in alpha], dtype=np.float64)

    def _apply_start_end_connectors(
        self,
        positions_inches,
        velocities_inches,
        dt,
        start_point_inches,
        end_point_inches,
        start_point_norm,
        end_point_norm,
        planning_start_norm,
        planning_end_norm,
    ):
        connectors_added = False
        planning_start_inches = np.array(self.normalized_to_inches(planning_start_norm), dtype=np.float64)
        planning_end_inches = np.array(self.normalized_to_inches(planning_end_norm), dtype=np.float64)
        self.pseudo_start_inches = planning_start_inches
        self.pseudo_end_inches = planning_end_inches

        if not np.allclose(planning_start_norm, start_point_norm):
            start_connector = self._build_straight_line_points(start_point_inches, planning_start_inches)
            positions_inches = np.vstack((start_connector[:-1], positions_inches))
            connectors_added = True

        if not np.allclose(planning_end_norm, end_point_norm):
            end_connector = self._build_straight_line_points(planning_end_inches, end_point_inches)
            positions_inches = np.vstack((positions_inches, end_connector[1:]))
            connectors_added = True

        if connectors_added:
            if len(positions_inches) >= 2:
                velocities_inches = np.diff(positions_inches, axis=0) / dt
            else:
                velocities_inches = np.zeros((0, 2), dtype=np.float64)

        return positions_inches, velocities_inches

    def _build_fallback_positions_normalized(self, a_star_path, planning_start_norm, planning_end_norm):
        if a_star_path is not None and len(a_star_path) >= 2:
            fallback_path = np.array(a_star_path, dtype=np.float64)
            if len(fallback_path) == self.num_steps:
                return fallback_path

            interp_axis = np.linspace(0, 1, len(fallback_path))
            sample_axis = np.linspace(0, 1, self.num_steps)
            fallback_x = np.interp(sample_axis, interp_axis, fallback_path[:, 0])
            fallback_y = np.interp(sample_axis, interp_axis, fallback_path[:, 1])
            return np.column_stack((fallback_x, fallback_y))

        fallback_x, fallback_y = self.get_initial_path(
            planning_start_norm[0],
            planning_start_norm[1],
            planning_end_norm[0],
            planning_end_norm[1],
        )
        return np.column_stack((fallback_x, fallback_y))

    def _positions_normalized_to_trajectory(self, positions_normalized, dt):
        positions_inches = np.array([self.normalized_to_inches(pos) for pos in positions_normalized])
        if len(positions_inches) >= 2:
            velocities_inches = np.diff(positions_inches, axis=0) / dt
        else:
            velocities_inches = np.zeros((0, 2), dtype=np.float64)
        return positions_inches, velocities_inches

    def _get_robot_norms(self, robot: Robot):
        buffer_radius_norm = robot.buffer / self.field_size_inches
        robot_radius = sqrt(robot.length**2 + robot.width**2) / 2
        robot_radius_norm = robot_radius / self.field_size_inches
        total_radius_norm = robot_radius_norm + buffer_radius_norm
        return robot_radius_norm, buffer_radius_norm, total_radius_norm

    def _normalize_obstacles(self, obstacles):
        normalized_obstacles = []
        for obs in obstacles:
            obs_pos_norm = self.inches_to_normalized(np.array([obs.x, obs.y]))
            normalized_obstacles.append(Obstacle(
                obs_pos_norm[0],
                obs_pos_norm[1],
                obs.radius / self.field_size_inches,
                obs.ignore_collision,
            ))
        return normalized_obstacles

    def _get_marker_points_normalized(self, start_point, end_point):
        start_norm = self.inches_to_normalized(np.array(start_point, dtype=np.float64))
        end_norm = self.inches_to_normalized(np.array(end_point, dtype=np.float64))
        pseudo_start_inches = getattr(self, 'pseudo_start_inches', np.array(start_point, dtype=np.float64))
        pseudo_end_inches = getattr(self, 'pseudo_end_inches', np.array(end_point, dtype=np.float64))
        pseudo_start_norm = self.inches_to_normalized(np.array(pseudo_start_inches, dtype=np.float64))
        pseudo_end_norm = self.inches_to_normalized(np.array(pseudo_end_inches, dtype=np.float64))
        return start_norm, end_norm, pseudo_start_norm, pseudo_end_norm

    def a_star_search(self, start_point, end_point, obstacles, total_radius_norm):
        """
        Perform A* search on the planning grid.
        """
        grid_size = GRID_SIZE
        grid, start, end = self._generate_obstacle_grid(obstacles, start_point, end_point, total_radius_norm, grid_size)

        start = self._find_nearest_valid_cell_bfs(grid, start)
        end = self._find_nearest_valid_cell_bfs(grid, end)
        if start is None or end is None:
            return None

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
                if grid[neighbor[1], neighbor[0]] == 1:  # Skip blocked cells (grid is [row=y, col=x])
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

    def generate_grid_png(self, obstacles, start_point, end_point, robot: Robot):
        """
        Generate a PNG of the grid based on obstacles, swap black/white, and add dots for start and end points.
        All inputs are in inches.
        """
        import matplotlib.pyplot as plt

        grid_size = GRID_SIZE
        
        _, _, total_radius_norm = self._get_robot_norms(robot)
        
        normalized_obstacles = self._normalize_obstacles(obstacles)
        start_norm, end_norm, pseudo_start_norm, pseudo_end_norm = self._get_marker_points_normalized(start_point, end_point)
        
        # Generate the grid using the helper method
        grid, start_grid, end_grid = self._generate_obstacle_grid(normalized_obstacles, start_norm, end_norm, total_radius_norm, grid_size)
        pseudo_start_grid = self._normalized_to_grid(pseudo_start_norm, grid_size)
        pseudo_end_grid = self._normalized_to_grid(pseudo_end_norm, grid_size)
        
        # Invert the grid (0=white, 1=black for display)
        grid = 1 - grid

        # Plot the grid
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(grid, cmap="gray", origin="lower")
        ax.set_title("Grid Visualization")
        ax.set_xticks([]); ax.set_yticks([])

        # Add requested start/end as stars and pseudo start/end as circles
        ax.plot(start_grid[0], start_grid[1], '*', color='orange', markersize=12, label='start')
        ax.plot(end_grid[0], end_grid[1], '*', color='green', markersize=12, label='end')
        ax.plot(
            pseudo_start_grid[0],
            pseudo_start_grid[1],
            'o',
            markerfacecolor='none',
            markeredgecolor='orange',
            markersize=9,
            label='pseudo start',
        )
        ax.plot(
            pseudo_end_grid[0],
            pseudo_end_grid[1],
            'o',
            markerfacecolor='none',
            markeredgecolor='green',
            markersize=9,
            label='pseudo end',
        )
        ax.legend(loc="upper right")

        plt.tight_layout()

        # Save the grid as a PNG
        plt.savefig(f"{self.output_dir}/grid.png")
        plt.close(fig)
        print(f"Grid PNG saved as {self.output_dir}/grid.png")

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    robot = Robot(
        name="TestRobot",
        team=Team.RED,
        size=RobotSize.INCH_24,
        max_speed=60,
        max_acceleration=60,
        buffer=1.0,
        length=18.0,
        width=18.0,
    )

    planner = PathPlanner(
        field_size_inches=INCHES_PER_FIELD,
        field_center=(0, 0)
    )

    total_solve_time = 0
    successful_solve_time = 0
    unsuccessful_solve_time = 0
    successful_trials = 0
    unsuccessful_trials = 0
    total_trials = 1000

    for i in range(total_trials):
        # All values should be in INCHES for the Solve() method
        start_point = [np.random.uniform(-60, 60), np.random.uniform(-60, 60)]
        end_point = [np.random.uniform(-60, 60), np.random.uniform(-60, 60)]

        # start_point = [-7.2, -50.4]  # Converted from normalized (0.45, 0.15) to inches
        # end_point = [0.0, 50.4]      # Converted from normalized (0.50, 0.85) to inches


        obstacles = [
            Obstacle(0.0, 0.0, 11.3, False),       # Center Goal Structure
            Obstacle(-21.0, 48.0, 3.0, False),     # Long Goal Top - Left End
            Obstacle(0.0, 48.0, 3.0, False),       # Long Goal Top - Center
            Obstacle(21.0, 48.0, 3.0, False),      # Long Goal Top - Right End
            Obstacle(-21.0, -48.0, 3.0, False),    # Long Goal Bottom - Left End
            Obstacle(0.0, -48.0, 3.0, False),      # Long Goal Bottom - Center
            Obstacle(21.0, -48.0, 3.0, False),     # Long Goal Bottom - Right End
            Obstacle(58.0, -10.0, 0.0, False),     # Blue Park Zone Bottom Corner
            Obstacle(58.0, 10.0, 0.0, False),      # Blue Park Zone Top Corner
            Obstacle(-58.0, -10.0, 0.0, False),    # Red Park Zone Bottom Corner
            Obstacle(-58.0, 10.0, 0.0, False),     # Red Park Zone Top Corner
                    ]

        positions, velocities, dt = planner.Solve(start_point=start_point, end_point=end_point, obstacles=obstacles, robot=robot, optimize=True)

        if planner.status == 'Solve_Succeeded':
            successful_trials += 1
            successful_solve_time += planner.solve_time
        else:
            unsuccessful_trials += 1
            unsuccessful_solve_time += planner.solve_time

        total_solve_time += planner.solve_time

        planner.print_trajectory_details(positions, velocities, dt, None)
        #planner.plotResults(positions, velocities, start_point, end_point, obstacles, robot=robot)
        input()

    print(f"Average solve time (successful): {successful_solve_time / successful_trials:.3f} seconds" if successful_trials > 0 else "No successful trials")
    print(f"Average solve time (unsuccessful): {unsuccessful_solve_time / unsuccessful_trials:.3f} seconds" if unsuccessful_trials > 0 else "No unsuccessful trials")
    print(f"Average solve time (overall): {total_solve_time / total_trials:.3f} seconds")
    print(f"Successful trials: {successful_trials}")
    print(f"Success rate: {successful_trials / total_trials * 100:.2f}%")