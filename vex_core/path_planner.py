from casadi import *
import numpy as np
from math import sqrt, exp
from collections import deque
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vex_core.robot import Robot, Team, RobotSize

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
GRID_SIZE = 144
CELL_SIZE = INCHES_PER_FIELD / GRID_SIZE

# =============================================================================
# PathPlanner Class Definitions
# =============================================================================
class PathPlanner:
    # All values are based on a scale from 0 to 1, where 1 is the length of the field
    def __init__(self, field_size=144, field_center=(0, 0), output_dir="path_planner"):
        # -------------------------------------------------------------------------
        # Initialization: Field parameters
        # -------------------------------------------------------------------------

        self.NUM_OF_ACTS = 2   # Number of MPC actions (vx, vy)
        self.NUM_OF_STATES = 2  # Number of MPC states (px, py)

        # Field coordinate system parameters
        self.field_size = field_size
        self.field_center = np.array(field_center, dtype=np.float64)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _normalize(self, pos):
        """Convert position from inches (field coordinates) to normalized scale (0-1)."""
        if isinstance(pos, np.ndarray):
            return (pos - self.field_center + self.field_size / 2) / self.field_size
        pos_array = np.array(pos, dtype=np.float64)
        return (pos_array - self.field_center + self.field_size / 2) / self.field_size
    
    def _denormalize(self, pos_normalized):
        """Convert position from normalized scale (0-1) to inches (field coordinates)."""
        if isinstance(pos_normalized, np.ndarray):
            return pos_normalized * self.field_size - self.field_size / 2 + self.field_center
        pos_array = np.array(pos_normalized, dtype=np.float64)
        return pos_array * self.field_size - self.field_size / 2 + self.field_center
    
    def initialize(self, planning_start, planning_end, robot: Robot):
        inches_per_step = 6.0
        distance = float(np.linalg.norm(planning_end - planning_start))
        self.num_steps = max(3, int(distance / inches_per_step))
        self.initial_time_step = 0.1

        self.max_time = distance / (0.5 * robot.max_speed)
        self.min_time = 1e-2
        self.time_step_min = self.min_time/self.num_steps  # Minimum time step
        self.time_step_max = self.max_time/self.num_steps  # Maximum time step


    def get_initial_path(self, x1, y1, x2, y2):
        # -------------------------------------------------------------------------
        # Path Initialization: Compute initial guess for the path
        # -------------------------------------------------------------------------
        x1 = x1 - 0.5; y1 = y1 - 0.5; x2 = x2 - 0.5; y2 = y2 - 0.5
        init_x = np.linspace(x1+0.5, x2+0.5, self.num_steps)
        init_y = np.linspace(y1+0.5, y2+0.5, self.num_steps)
    
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
        robot_radius_normalized, buffer_radius_normalized, total_radius_normalized = self._get_robot_norms(robot)
        
        max_velocity_normalized = robot.max_speed / self.field_size
        max_accel_normalized = robot.max_acceleration / self.field_size

        start_time = time.time()
        
        start_point = np.array(start_point, dtype=np.float64)
        end_point = np.array(end_point, dtype=np.float64)

        # Convert input positions from inches to normalized coordinates
        # Generate the obstacle grid ONCE per solve using inches directly
        grid, start_grid, end_grid = self._generate_obstacle_grid(
            obstacles, start_point, end_point, robot.total_radius
        )

        # Snap start/end to nearest valid grid cells if either is invalid
        planning_start, planning_end = self._snap_endpoints_to_valid_cells(
            grid, start_grid, end_grid, obstacles, robot.total_radius
        )
        
        self.initialize(planning_start, planning_end, robot)
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

        # Perform A* search to get an initial path in inches
        a_star_path = self.a_star_search(grid, planning_start, planning_end, obstacles, robot.total_radius)
        if a_star_path:
            # Interpolate the A* path to match the number of steps
            a_star_path = np.array(a_star_path)
            init_x = np.interp(np.linspace(0, 1, self.num_steps), np.linspace(0, 1, len(a_star_path)), a_star_path[:, 0])
            init_y = np.interp(np.linspace(0, 1, self.num_steps), np.linspace(0, 1, len(a_star_path)), a_star_path[:, 1])
        else:
            # Fall back to the default initial path if A* fails
            init_x, init_y = self.get_initial_path(planning_start[0], planning_start[1], planning_end[0], planning_end[1])

        init_x[0], init_y[0] = planning_start[0], planning_start[1]
        init_x[-1], init_y[-1] = planning_end[0], planning_end[1]

        self.init_x = init_x.copy()  # For plotting
        self.init_y = init_y.copy()

        if not optimize:
            positions = self._build_fallback_positions(
                a_star_path,
                planning_start,
                planning_end,
            )
            dt = self.initial_time_step
            self.optimizer_status = 'NLP_Disabled'
            self.status = 'AStar_Only_Succeeded'

            if len(positions) >= 2:
                velocities = np.diff(positions, axis=0) / dt
            else:
                velocities = np.zeros((0, 2), dtype=np.float64)

            positions, velocities = self._apply_start_end_connectors(
                positions,
                velocities,
                dt,
                start_point,
                end_point,
                planning_start,
                planning_end,
                robot,
            )
            self.solve_time = time.time() - start_time
            return positions, velocities, dt, grid

        # Build initial guess using the A* path and zero velocity
        init_time_step = self.initial_time_step

        # Calculate dynamically sound initial velocities from the position guess
        init_vx = np.diff(init_x) / init_time_step
        init_vy = np.diff(init_y) / init_time_step

        # Clip to ensure they don't violate your strict velocity bounds (in inches/s)
        max_vel = robot.max_speed
        init_vx = np.clip(init_vx, -max_vel, max_vel)
        init_vy = np.clip(init_vy, -max_vel, max_vel)

        init_v = np.concatenate((init_vx, init_vy))

        # Convert the guesses to normalized space exactly for the solver
        init_xy = np.column_stack((init_x, init_y))
        init_xy_normalized = self._normalize(init_xy)
        init_x_normalized = init_xy_normalized[:, 0]
        init_y_normalized = init_xy_normalized[:, 1]
        init_v_normalized = init_v / self.field_size
        
        x_ = np.concatenate((init_x_normalized, init_y_normalized, init_v_normalized, [init_time_step]))

        time_step = x[self.indexes.dt]
        cost += w_time_step * time_step * self.num_steps

        # Define variable bounds
        x_lowerbound_ = [0] * self.num_of_x_
        x_upperbound_ = [1] * self.num_of_x_
        for i in range(self.indexes.px, self.indexes.py + self.num_steps):
            x_lowerbound_[i] = total_radius_normalized
            x_upperbound_[i] = 1 - total_radius_normalized
        for i in range(self.indexes.vx, self.indexes.vy + self.num_steps - 1):
            x_lowerbound_[i] = -max_velocity_normalized
            x_upperbound_[i] = max_velocity_normalized

        # Constrain start and final positions (using normalized coordinates)
        planning_start_normalized = self._normalize(planning_start)
        planning_end_normalized = self._normalize(planning_end)

        x_lowerbound_[self.indexes.px] = planning_start_normalized[0]
        x_lowerbound_[self.indexes.py] = planning_start_normalized[1]
        x_lowerbound_[self.indexes.px + self.num_steps - 1] = planning_end_normalized[0]
        x_lowerbound_[self.indexes.py + self.num_steps - 1] = planning_end_normalized[1]
        x_upperbound_[self.indexes.px] = planning_start_normalized[0]
        x_upperbound_[self.indexes.py] = planning_start_normalized[1]
        x_upperbound_[self.indexes.px + self.num_steps - 1] = planning_end_normalized[0]
        x_upperbound_[self.indexes.py + self.num_steps - 1] = planning_end_normalized[1]

        # Constrain time step
        x_lowerbound_[self.indexes.dt] = self.time_step_min
        x_upperbound_[self.indexes.dt] = self.time_step_max

        # Define constraint bounds
        g_lowerbound_ = [0] * self.num_of_g_
        g_upperbound_ = [1] * self.num_of_g_

        g = [SX(0) for _ in range(self.num_of_g_)]
        g_index = 0

        # Speed constraints
        for i in range(self.num_steps - 1):
            curr_vx_index = self.indexes.vx + i
            curr_vy_index = self.indexes.vy + i
            vx = x[curr_vx_index]
            vy = x[curr_vy_index]
            g[g_index] = vx**2 + vy**2
            g_lowerbound_[g_index] = 0
            g_upperbound_[g_index] = max_velocity_normalized**2
            g_index += 1
        
        # Acceleration constraints
        for i in range(self.num_steps - 2):
            curr_vx_index = self.indexes.vx + i
            curr_vy_index = self.indexes.vy + i
            next_vx_index = curr_vx_index + 1
            next_vy_index = curr_vy_index + 1
            dvx = x[next_vx_index] - x[curr_vx_index]
            dvy = x[next_vy_index] - x[curr_vy_index]
            g[g_index] = dvx**2 + dvy**2 - (max_accel_normalized * time_step)**2
            g_lowerbound_[g_index] = -1e10
            g_upperbound_[g_index] = 0
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
                obstacle_normalized = self._normalize(np.array([obstacle.x, obstacle.y]))
                g[g_index] = (curr_px - obstacle_normalized[0])**2 + (curr_py - obstacle_normalized[1])**2
                if obstacle.ignore_collision:
                    g_lowerbound_[g_index] = 0.0
                else:
                    g_lowerbound_[g_index] = ((obstacle.radius / self.field_size) + total_radius_normalized)**2
                g_upperbound_[g_index] = 2.0 # Max distance (sqrt2) squared
                g_index += 1

        nlp = {'x': x, 'f': cost, 'g': vertcat(*g)}

        opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": False,
            
            # Precision Settings
            "ipopt.tol": 1e-4,
            "ipopt.max_iter": 250,
            
            # Early Exit (Speed Hack)
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.acceptable_iter": 3,
            
            # Stability and Scaling
            "ipopt.nlp_scaling_method": "gradient-based",
            "ipopt.mu_strategy": "adaptive",

            # Prevent slacks from getting too small and causing singularities
            "ipopt.bound_relax_factor": 1e-8,

            "ipopt.diverging_iterates_tol": 1e20,

            # HSL Solver (If you have it on the MSOE cluster, use 'ma27')
            "ipopt.linear_solver": "mumps", 
        }

        solver = nlpsol('solver', 'ipopt', nlp, opts)
        res = solver(
            x0=DM(x_), 
            lbx=DM(x_lowerbound_), 
            ubx=DM(x_upperbound_), 
            lbg=DM(g_lowerbound_), 
            ubg=DM(g_upperbound_)
        )
        solver_stats = solver.stats()

        self.optimizer_status = solver_stats['return_status']
        if self.optimizer_status == 'Solved_To_Acceptable_Level':
            self.optimizer_status = 'Solve_Succeeded' # Treat acceptable as solved
        self.status = self.optimizer_status

        # Extract solution components
        if self.optimizer_status == 'Solve_Succeeded':
            x_opt = res['x'].full().flatten()
            pos_x_normalized = x_opt[self.indexes.px:self.indexes.py]
            pos_y_normalized = x_opt[self.indexes.py:self.indexes.vx]
            vel_x_normalized = x_opt[self.indexes.vx:self.indexes.vy]
            vel_y_normalized = x_opt[self.indexes.vy:self.indexes.dt]
            dt = x_opt[self.indexes.dt]

            # Convert positions from normalized coordinates back to inches
            positions_normalized = np.column_stack((pos_x_normalized, pos_y_normalized))
            positions = np.array([self._denormalize(pos) for pos in positions_normalized])

            # Convert velocities from normalized units back to inches per second
            velocities_normalized = np.column_stack((vel_x_normalized, vel_y_normalized))
            velocities = velocities_normalized * self.field_size
        else:
            positions = self._build_fallback_positions(
                a_star_path,
                planning_start,
                planning_end,
            )
            dt = self.initial_time_step
            self.status = 'Grid_Fallback_Succeeded'
            if len(positions) >= 2:
                velocities = np.diff(positions, axis=0) / dt
            else:
                velocities = np.zeros((0, 2), dtype=np.float64)

        positions, velocities = self._apply_start_end_connectors(
            positions,
            velocities,
            dt,
            start_point,
            end_point,
            planning_start,
            planning_end,
            robot,
        )

        self.solve_time = time.time() - start_time

        return positions, velocities, dt, grid

    def print_trajectory_details(self, positions, velocities, dt):
        # -------------------------------------------------------------------------
        # Trajectory Output: Print details and save to a file
        # Positions are already in inches
        # -------------------------------------------------------------------------
        print(f"{'Step':<5} | {'Position (x, y)':<20} | {'Velocity (vx, vy, speed)':<30} | {'Acceleration (ax, ay, magnitude)':<30}")
        print("-" * 100)
        
        num_start_points = getattr(self, 'num_start_points', 0)
        num_end_points = getattr(self, 'num_end_points', 0)
        total_points = len(positions)
        
        for i in range(total_points):
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
                
            connector_str = ""
            if i < num_start_points:
                connector_str = " (Start Connector)"
            elif i >= total_points - num_end_points:
                connector_str = " (End Connector)"
            
            print(f"{i:<5} | ({px:>8.2f}, {py:>8.2f}) | ({vx:>8.2f}, {vy:>8.2f}, {np.sqrt(vx**2 + vy**2):>8.2f}) | ({ax:>8.2f}, {ay:>8.2f}, {np.sqrt(ax**2 + ay**2):>8.2f}){connector_str}")
        
        print(f"\nTime step: {dt:.2f}")
        print(f"Path time: {dt * len(positions):.2f}")
        print(f"\nStatus: {self.status}")
        if hasattr(self, 'optimizer_status') and self.optimizer_status != self.status:
            print(f"Optimizer status: {self.optimizer_status}")
        print(f"Solve time: {self.solve_time:.3f} seconds")

    def plot_results(self, positions, velocities, start_point, end_point, obstacles, robot: Robot, grid=None):
        # -------------------------------------------------------------------------
        # Plot Results: Display trajectory, obstacles, and robot boundaries
        # All inputs are in inches
        # -------------------------------------------------------------------------
        
        import matplotlib.pyplot as plt
        from matplotlib.transforms import Affine2D

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # --- SUBPLOT 1: Field Path ---
        planned_px = positions[:, 0]
        planned_py = positions[:, 1]
        planned_vx = velocities[:, 0]
        planned_vy = velocities[:, 1]
        planned_theta = np.arctan2(planned_vy, planned_vx)
        planned_theta = np.concatenate(([planned_theta[0]], planned_theta, [planned_theta[-1]]))
        
        # Calculate marker points in inches
        start, end, pseudo_start, pseudo_end = self._get_marker_points(start_point, end_point)
        
        ax1.plot(self.init_x, self.init_y, linestyle=':', color='gray', alpha=0.7, label='initial path')
        ax1.plot(planned_px, planned_py, '-o', label='path', color="blue", alpha=0.5)
        
        theta_list = np.linspace(0, 2 * np.pi, 100)
        num_outlines = 3
        mod = max(1, round(self.num_steps / (num_outlines - 1)))
        index = 0
        
        robot_radius = robot.radius
        buffer_radius = robot.buffer
        
        for px, py, theta in zip(planned_px, planned_py, planned_theta):
            rotation = Affine2D().rotate_around(px, py, theta)
            rectangle = plt.Rectangle((px - robot.length / 2, py - robot.width / 2), robot.length, robot.width,
                                      edgecolor='blue', facecolor='none', alpha=1)
            rectangle.set_transform(rotation + ax1.transData)
            if index % mod == 0 or index == self.num_steps - 1:
                robot_circle_x = px + robot_radius * np.cos(theta_list)
                robot_circle_y = py + robot_radius * np.sin(theta_list)
                ax1.plot(robot_circle_x, robot_circle_y, '--', color='blue', alpha=0.5, label='robot radius' if index == 0 else None)
                ax1.add_patch(rectangle)
            index += 1
            
        ax1.plot(start[0], start[1], '*', color='orange', markersize=12, label='start')
        ax1.plot(end[0], end[1], '*', color='green', markersize=12, label='end')

        ax1.plot(
            pseudo_start[0], pseudo_start[1], 'o', markerfacecolor='none', markeredgecolor='orange', markersize=8, label='pseudo start'
        )
        ax1.plot(
            pseudo_end[0], pseudo_end[1], 'o', markerfacecolor='none', markeredgecolor='green', markersize=8, label='pseudo end'
        )
        first_obstacle = True
        for obstacle in obstacles:
            danger_x = obstacle.x + obstacle.radius * np.cos(theta_list)
            danger_y = obstacle.y + obstacle.radius * np.sin(theta_list)
            buffer_x = obstacle.x + (obstacle.radius + buffer_radius) * np.cos(theta_list)
            buffer_y = obstacle.y + (obstacle.radius + buffer_radius) * np.sin(theta_list)
            if first_obstacle:
                ax1.plot(danger_x, danger_y, 'r-', label='obstacle')
                ax1.plot(buffer_x, buffer_y, 'r--', label='buffer zone', alpha=0.5)
                first_obstacle = False
            else:
                ax1.plot(danger_x, danger_y, 'r-')
                ax1.plot(buffer_x, buffer_y, 'r--', alpha=0.5)
                
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0., frameon=False)
        ax1.set_aspect('equal', adjustable='box')
        half_field = self.field_size / 2
        ax1.set_xlim(-half_field, half_field)
        ax1.set_ylim(-half_field, half_field)
        
        import matplotlib.ticker as ticker
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(24))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(24))
        ax1.grid(True)
        ax1.set_title("Planned Path")

        # --- SUBPLOT 2: Grid Visualization ---
        grid_size = GRID_SIZE
        total_radius = robot.total_radius
        
        # Use existing grid or generate it
        if grid is None:
            grid, _, _ = self._generate_obstacle_grid(obstacles, start, end, total_radius, grid_size)
        else:
            # Need to copy to avoid mutating the shared object with the inversion step later
            grid = np.copy(grid)
            
        # Invert the grid (0=white, 1=black for display)
        grid = 1 - grid

        extent = [-half_field, half_field, -half_field, half_field]
        ax2.imshow(grid, cmap="gray", origin="lower", extent=extent)

        # Add requested start/end as stars and pseudo start/end as circles
        ax2.plot(start[0], start[1], '*', color='orange', markersize=12, label='start')
        ax2.plot(end[0], end[1], '*', color='green', markersize=12, label='end')
        ax2.plot(
            pseudo_start[0], pseudo_start[1],
            'o', markerfacecolor='none', markeredgecolor='orange', markersize=9, label='pseudo start',
        )
        ax2.plot(
            pseudo_end[0], pseudo_end[1],
            'o', markerfacecolor='none', markeredgecolor='green', markersize=9, label='pseudo end',
        )
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0., frameon=False)

        ax2.xaxis.set_major_locator(ticker.MultipleLocator(24))
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(24))
        ax2.grid(True)
        ax2.set_title("Obstacle Grid")

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/path.png', bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory

        print(f"Visualization saved to {self.output_dir}/path.png")
        
    
    def getPath(self, positions, dt):
        """Legacy method for compatibility - returns separate x, y arrays and total time."""
        planned_px = positions[:, 0]
        planned_py = positions[:, 1]
        total_path_time = dt * len(positions)
        return planned_px, planned_py, total_path_time

    def _generate_obstacle_grid(self, obstacles, start_point, end_point, total_radius, grid_size=GRID_SIZE):
        """
        Helper method to generate an obstacle grid for A* search and visualization in inches.
        """
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Helper to convert physical inch to grid index
        def in_to_grid(inch_val):
            norm_val = (inch_val - (-self.field_size / 2)) / self.field_size
            return int(np.clip(norm_val * grid_size, 0, grid_size - 1))
            
        def in_to_grid_radius(inch_radius):
            return int((inch_radius / self.field_size) * grid_size)

        # Block grid cells based on obstacles
        for obstacle in obstacles:
            if obstacle.ignore_collision:
                continue
            cx, cy = in_to_grid(obstacle.x), in_to_grid(obstacle.y)
            radius = in_to_grid_radius(obstacle.radius + total_radius)
            for x in range(max(0, cx - radius), min(grid_size, cx + radius + 1)):
                for y in range(max(0, cy - radius), min(grid_size, cy + radius + 1)):
                    if (x - cx)**2 + (y - cy)**2 <= radius**2:
                        grid[y, x] = 1  # Mark as blocked (grid is [row=y, col=x])
        
        # Block any cell within robot radius of the field boundary
        robot_radius_cells = in_to_grid_radius(total_radius)
        for i in range(grid_size):
            for j in range(grid_size):
                if i < robot_radius_cells or i >= grid_size - robot_radius_cells or j < robot_radius_cells or j >= grid_size - robot_radius_cells:
                    grid[i, j] = 1  # Mark as blocked

        start = (in_to_grid(start_point[0]), in_to_grid(start_point[1]))
        end = (in_to_grid(end_point[0]), in_to_grid(end_point[1]))
        
        return grid, start, end

    def _to_grid(self, point, grid_size=GRID_SIZE):
        norm_val_x = (point[0] - (-self.field_size / 2)) / self.field_size
        norm_val_y = (point[1] - (-self.field_size / 2)) / self.field_size
        x = int(np.clip(norm_val_x * grid_size, 0, grid_size - 1))
        y = int(np.clip(norm_val_y * grid_size, 0, grid_size - 1))
        return (x, y)

    def _grid_to(self, grid_point, grid_size=GRID_SIZE):
        norm_x = (grid_point[0] + 0.5) / grid_size # Centering it
        norm_y = (grid_point[1] + 0.5) / grid_size
        inch_x = norm_x * self.field_size - (self.field_size / 2)
        inch_y = norm_y * self.field_size - (self.field_size / 2)
        return np.array([inch_x, inch_y], dtype=np.float64)

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

    def _is_point_valid_for_nlp(self, point, obstacles, total_radius):
        px, py = float(point[0]), float(point[1])

        # Keep robot center inside field with boundary margin.
        half_field = self.field_size / 2
        if px < -half_field + total_radius or px > half_field - total_radius:
            return False
        if py < -half_field + total_radius or py > half_field - total_radius:
            return False

        for obstacle in obstacles:
            if obstacle.ignore_collision:
                continue
            min_dist = obstacle.radius + total_radius
            if (px - obstacle.x) ** 2 + (py - obstacle.y) ** 2 < min_dist ** 2:
                return False

        return True

    def _snap_endpoints_to_valid_cells(self, grid, start, end, obstacles, total_radius, grid_size=GRID_SIZE):

        def nlp_validator(grid_point):
            point = self._grid_to(grid_point, grid_size)
            return self._is_point_valid_for_nlp(
                point,
                obstacles,
                total_radius,
            )

        start_pt = self._grid_to(start, grid_size)
        end_pt = self._grid_to(end, grid_size)
        
        # Check if the exact points are acceptable
        start_valid = self._is_point_valid_for_nlp(start_pt, obstacles, total_radius)
        end_valid = self._is_point_valid_for_nlp(end_pt, obstacles, total_radius)

        if not start_valid:
            snapped_start = self._find_nearest_valid_cell_bfs(grid, start, validator=nlp_validator)
            if snapped_start is not None:
                start_pt = self._grid_to(snapped_start, grid_size)

        if not end_valid:
            snapped_end = self._find_nearest_valid_cell_bfs(grid, end, validator=nlp_validator)
            if snapped_end is not None:
                end_pt = self._grid_to(snapped_end, grid_size)

        return start_pt, end_pt

    def _build_straight_line_points(self, start, end):
        step = 6.0
        distance = np.linalg.norm(end - start)

        if distance <= CELL_SIZE: # If the distance is less than a grid cell, just return the start point
            return np.array([start], dtype=np.float64)

        num_segments = max(1, int(np.ceil(distance / step)))
        alpha = np.linspace(0.0, 1.0, num_segments + 1)
        return np.array([start + a * (end - start) for a in alpha], dtype=np.float64)

    def _apply_start_end_connectors(
        self,
        positions,
        velocities,
        dt,
        start_point,
        end_point,
        planning_start,
        planning_end,
        robot,
    ):
        connectors_added = False
        self.pseudo_start = planning_start
        self.pseudo_end = planning_end

        num_start_points = 0
        num_end_points = 0

        if not np.allclose(planning_start, start_point, atol=CELL_SIZE):
            start_connector = self._build_straight_line_points(start_point, planning_start)
            positions = np.vstack((start_connector[:-1], positions))
            connectors_added = True
            num_start_points = len(start_connector) - 1

        if not np.allclose(planning_end, end_point, atol=CELL_SIZE):
            end_connector = self._build_straight_line_points(planning_end, end_point)
            positions = np.vstack((positions, end_connector[1:]))
            connectors_added = True
            num_end_points = len(end_connector) - 1

        if connectors_added:
            if len(positions) >= 2:
                # Calculate exact velocities based on position differences
                velocities = np.diff(positions, axis=0) / dt
            else:
                velocities = np.zeros((0, 2), dtype=np.float64)

        self.num_start_points = num_start_points
        self.num_end_points = num_end_points

        return positions, velocities

    def _build_fallback_positions(self, a_star_path, planning_start, planning_end):
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
            planning_start[0],
            planning_start[1],
            planning_end[0],
            planning_end[1],
        )
        return np.column_stack((fallback_x, fallback_y))

    def _get_robot_norms(self, robot: Robot):
        buffer_radius_normalized = robot.buffer / self.field_size
        robot_radius_normalized = robot.radius / self.field_size
        total_radius_normalized = robot.total_radius / self.field_size
        return robot_radius_normalized, buffer_radius_normalized, total_radius_normalized

    def _get_marker_points(self, start_point, end_point):
        start = np.array(start_point, dtype=np.float64)
        end = np.array(end_point, dtype=np.float64)
        pseudo_start = getattr(self, 'pseudo_start', start)
        pseudo_end = getattr(self, 'pseudo_end', end)
        return start, end, pseudo_start, pseudo_end

    def a_star_search(self, grid, start_point, end_point, obstacles, total_radius):
        """
        Perform A* search on the planning grid. Inputs are entirely in inches.
        """
        grid_size = GRID_SIZE
        # grid already passed in, just need to parse the start/end points to grid indices
        start = self._to_grid(start_point, grid_size)
        end = self._to_grid(end_point, grid_size)

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

                dx = abs(current[0] - neighbor[0])
                dy = abs(current[1] - neighbor[1])
                move_cost = 1.414 if dx + dy == 2 else 1.0
                tentative_g_score = g_score[current] + move_cost
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, end)
                    heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def heuristic(self, a, b):
        # Euclidean distance
        return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def get_neighbors(self, node, grid_size):
        """
        Get valid neighbors for a node in the grid.
        """
        x, y = node
        neighbors = [
            (x + dx, y + dy) for dx, dy in [
                (-1, 0), (1, 0), (0, -1), (0, 1),      # Cardinal
                (-1, -1), (-1, 1), (1, -1), (1, 1)     # Diagonal
            ]
        ]
        return [(nx, ny) for nx, ny in neighbors if 0 <= nx < grid_size and 0 <= ny < grid_size]

    def reconstruct_path(self, came_from, current, grid_size):
        """
        Reconstruct the path from the A* search.
        """
        original_path = []
        while current in came_from:
            original_path.append(current)
            current = came_from[current]
        original_path.append(current)
        original_path.reverse()
        
        path = []
        # Center the point inside the inch-based cell
        offset = self.field_size / grid_size * 0.5
        for point in original_path:
            norm_x, norm_y = point[0] / grid_size, point[1] / grid_size
            inch_x = norm_x * self.field_size - (self.field_size / 2) + offset
            inch_y = norm_y * self.field_size - (self.field_size / 2) + offset
            path.append((inch_x, inch_y))
        return path


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    robot = Robot(
        name="TestRobot",
        team=Team.RED,
        size=RobotSize.INCH_24,
        max_speed=25,
        max_acceleration=25,
        buffer=1.0,
        length=18.0,
        width=18.0,
    )

    planner = PathPlanner(
        field_size=INCHES_PER_FIELD,
        field_center=(0, 0)
    )

    total_solve_time = 0
    successful_solve_time = 0
    unsuccessful_solve_time = 0
    successful_trials = 0
    unsuccessful_trials = 0
    total_trials = 100

    for i in range(total_trials):
        # All values should be in INCHES for the Solve() method
        start_point = [np.random.uniform(-60, 60), np.random.uniform(-60, 60)]
        end_point = [np.random.uniform(-60, 60), np.random.uniform(-60, 60)]

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

        positions, velocities, dt, grid = planner.Solve(start_point=start_point, end_point=end_point, obstacles=obstacles, robot=robot, optimize=True)

        if planner.status == 'Solve_Succeeded':
            successful_trials += 1
            successful_solve_time += planner.solve_time
        else:
            unsuccessful_trials += 1
            unsuccessful_solve_time += planner.solve_time

            print(f"Unsuccessful trial: {planner.optimizer_status}")

        planner.plot_results(positions, velocities, start_point, end_point, obstacles, robot, grid)
        planner.print_trajectory_details(positions, velocities, dt)
        input()

        total_solve_time += planner.solve_time

    print(f"Average solve time (successful): {successful_solve_time / successful_trials:.3f} seconds" if successful_trials > 0 else "No successful trials")
    print(f"Average solve time (unsuccessful): {unsuccessful_solve_time / unsuccessful_trials:.3f} seconds" if unsuccessful_trials > 0 else "No unsuccessful trials")
    print(f"Average solve time (overall): {total_solve_time / total_trials:.3f} seconds")
    print(f"Successful trials: {successful_trials}")
    print(f"Success rate: {successful_trials / total_trials * 100:.2f}%")