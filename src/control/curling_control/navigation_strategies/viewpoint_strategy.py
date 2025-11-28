"""
Grid-based Direct Goal Navigation Strategy.

Idea:
- The field is discretized into a regular grid of nodes ("bins").
- On each control step, we plan a shortest path over this grid from
  the robot's current cell to the target's cell.
- Obstacles are projected onto the grid and their cells are removed
  from the search space.
- We then drive toward the *next* node on that grid path using a
  DirectGoal-style sub-controller:
    * orient toward the waypoint,
    * then move forward when approximately aligned.

New behavior in this version:
- The robot explicitly "stops at each bin":
    * For each grid node on the A* path, we consider the cell center
      a waypoint.
    * When the robot is within `waypoint_tolerance` of that waypoint,
      we command zero velocity.
    * Optionally we dwell there for `dwell_time_at_waypoint` seconds
      (to let localization settle).
    * On the next cycle, A* is recomputed from the new current pose,
      and the next grid cell along the path becomes the new waypoint.

If there is no valid grid path (e.g., target surrounded by obstacles),
we fall back to plain DirectGoal behavior (ignore obstacles).

Field / coordinate system:
- Same as for markers and /robot_pose:
    * (0, 0)  : bottom-right corner of the field
    * (6, 0)  : bottom-left  corner
    * (0, 9)  : top-right    corner
    * (6, 9)  : top-left     corner
- So the field is 6 m wide (x: 0 -> 6) and 9 m long (y: 0 -> 9).

Grid:
- By default: resolution = 1.0 m, so:
    * nx = 6 cells along x (0..6)
    * ny = 9 cells along y (0..9)
- Each cell is represented by its center coordinate.
"""

import math
import heapq
from typing import List, Tuple, Optional, Dict, Set

from geometry_msgs.msg import Twist, Vector3

from .base_strategy import BaseNavigationStrategy


# ---------------------------------------------------------------------------
# Field constants (in meters)
# ---------------------------------------------------------------------------

FIELD_WIDTH: float = 6.0   # along +x
FIELD_LENGTH: float = 9.0  # along +y


class GridDirectGoalStrategy(BaseNavigationStrategy):
    """
    Grid-based navigation strategy that approximates an optimal path
    on a discrete field mesh, then drives along that path using a
    DirectGoal-style controller toward the next grid node.

    Use cases:
    - You want something close to DirectGoal behavior, but with the
      ability to "route around" blocked regions (obstacles).
    - Later, you can feed in specific grid nodes to mark as blocked.

    This version:
    - Stops at each grid-cell waypoint (within waypoint_tolerance),
      optionally dwelling for a short time before progressing.
    """

    def __init__(
        self,
        goal_tolerance: float = 0.1,
        max_linear_velocity: float = 0.2,
        angular_gain: float = 1.0,
        min_angular_error_for_forward: float = 0.3,  # about 17 degrees
        grid_resolution: float = 1.0,
        obstacle_inflation_radius: float = 0.0,       # extra radius around obstacles (m)
        waypoint_tolerance: float = 0.25,             # how close to cell center counts as "reached"
        dwell_time_at_waypoint: float = 0.0,          # seconds to pause at each bin (0.0 = no dwell)
        control_dt: float = 0.05,                     # should match follow.py timer
        log_throttle_steps: int = 20,                 # print every N control cycles
        *args,
        **kwargs,
    ):
        """
        Args:
            goal_tolerance: Distance threshold to consider goal reached (meters).
            max_linear_velocity: Forward velocity when moving (m/s).
            angular_gain: Proportional gain for angular velocity control.
            min_angular_error_for_forward: Minimum heading error to allow
                forward motion (radians). Larger error -> turn in place.
            grid_resolution: Size (in m) of each grid cell (e.g. 1.0).
            obstacle_inflation_radius: If > 0, we block not just the obstacle
                cell but a neighborhood around it.
            waypoint_tolerance: Distance (m) to cell center to consider
                that bin "reached".
            dwell_time_at_waypoint: Duration (s) to stop at each bin before
                moving on. Set to 0.0 to not dwell.
            control_dt: Control-loop period (seconds). Used to convert
                dwell time into a number of control steps.
            log_throttle_steps: How many control steps between log prints
                (to avoid spamming at 20 Hz). Set to 1 to log every step.
        """
        self.goal_tolerance = goal_tolerance
        self.max_linear_velocity = max_linear_velocity
        self.angular_gain = angular_gain
        self.min_angular_error_for_forward = min_angular_error_for_forward

        # Grid definition
        self.grid_resolution = grid_resolution
        self.nx = max(1, int(FIELD_WIDTH / grid_resolution))
        self.ny = max(1, int(FIELD_LENGTH / grid_resolution))

        self.obstacle_inflation_radius = max(0.0, obstacle_inflation_radius)

        # Waypoint / dwell behavior
        self.waypoint_tolerance = waypoint_tolerance
        self.dwell_time_at_waypoint = dwell_time_at_waypoint
        self.control_dt = control_dt

        self.dwell_steps_required = (
            int(dwell_time_at_waypoint / control_dt)
            if dwell_time_at_waypoint > 0.0
            else 0
        )
        self._reset_dwell_state()

        # Simple logging throttle
        self.log_throttle_steps = max(1, log_throttle_steps)
        self._step_counter: int = 0

        print(
            f"[GridDirectGoalStrategy] Init: grid_resolution={self.grid_resolution:.2f} m, "
            f"nx={self.nx}, ny={self.ny}, goal_tolerance={self.goal_tolerance:.2f}, "
            f"waypoint_tolerance={self.waypoint_tolerance:.2f}, "
            f"dwell_time={self.dwell_time_at_waypoint:.2f}s "
            f"(steps={self.dwell_steps_required})"
        )

    # ------------------------------------------------------------------
    # Dwell helper
    # ------------------------------------------------------------------

    def _reset_dwell_state(self) -> None:
        self._in_dwell: bool = False
        self._dwell_steps_done: int = 0

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    def _world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """
        Map world coordinates (meters) to integer grid cell indices (ix, iy).
        ix: 0..nx-1 (x direction), iy: 0..ny-1 (y direction).
        """
        xr = min(max(x, 0.0), FIELD_WIDTH - 1e-6)
        yr = min(max(y, 0.0), FIELD_LENGTH - 1e-6)

        ix = int(xr / self.grid_resolution)
        iy = int(yr / self.grid_resolution)

        ix = max(0, min(self.nx - 1, ix))
        iy = max(0, min(self.ny - 1, iy))

        return ix, iy

    def _cell_to_world(self, ix: int, iy: int) -> Tuple[float, float]:
        """
        Map grid cell indices (ix, iy) back to world coordinates (center of cell).
        """
        x = (ix + 0.5) * self.grid_resolution
        y = (iy + 0.5) * self.grid_resolution

        # Clamp to field in case of rounding
        x = min(max(x, 0.0), FIELD_WIDTH)
        y = min(max(y, 0.0), FIELD_LENGTH)
        return x, y

    def _obstacles_to_blocked_cells(
        self, obstacles: List[Tuple[float, float]]
    ) -> Set[Tuple[int, int]]:
        """
        Convert obstacle world positions into a set of blocked grid cells.
        If obstacle_inflation_radius > 0, we also block neighboring cells
        within that radius.
        """
        blocked: Set[Tuple[int, int]] = set()
        if not obstacles:
            return blocked

        inflation_cells = int(self.obstacle_inflation_radius / self.grid_resolution)

        for (ox, oy) in obstacles:
            ix, iy = self._world_to_cell(ox, oy)
            if inflation_cells <= 0:
                blocked.add((ix, iy))
            else:
                # Inflate a square neighborhood around this cell
                for dx in range(-inflation_cells, inflation_cells + 1):
                    for dy in range(-inflation_cells, inflation_cells + 1):
                        jx = ix + dx
                        jy = iy + dy
                        if 0 <= jx < self.nx and 0 <= jy < self.ny:
                            blocked.add((jx, jy))

        return blocked

    # ------------------------------------------------------------------
    # Path planning (A*)
    # ------------------------------------------------------------------

    def _neighbors(self, ix: int, iy: int) -> List[Tuple[int, int]]:
        """8-connected neighborhood in the grid."""
        nbrs: List[Tuple[int, int]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                jx = ix + dx
                jy = iy + dy
                if 0 <= jx < self.nx and 0 <= jy < self.ny:
                    nbrs.append((jx, jy))
        return nbrs

    def _dist_cells(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance between centers of two grid cells (in meters)."""
        (ax, ay) = self._cell_to_world(*a)
        (bx, by) = self._cell_to_world(*b)
        return math.hypot(bx - ax, by - ay)

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Heuristic for A* (Euclidean distance, in meters)."""
        return self._dist_cells(a, b)

    def _plan_path(
        self,
        start_cell: Tuple[int, int],
        goal_cell: Tuple[int, int],
        blocked: Set[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """
        A* search on the grid, avoiding blocked cells.
        Returns a list of cells [start_cell, ..., goal_cell].
        If no path found, returns [].
        """
        if start_cell == goal_cell:
            return [start_cell]

        if goal_cell in blocked:
            # For this basic version, treat a blocked goal cell as unreachable.
            print("[GridDirectGoalStrategy] Goal cell is blocked; no grid path.")
            return []

        open_set: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, start_cell))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start_cell: 0.0}
        f_score: Dict[Tuple[int, int], float] = {start_cell: self._heuristic(start_cell, goal_cell)}

        closed: Set[Tuple[int, int]] = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in closed:
                continue
            closed.add(current)

            if current == goal_cell:
                # reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for nb in self._neighbors(*current):
                if nb in blocked:
                    continue
                tentative_g = g_score[current] + self._dist_cells(current, nb)

                if nb not in g_score or tentative_g < g_score[nb]:
                    came_from[nb] = current
                    g_score[nb] = tentative_g
                    f_score[nb] = tentative_g + self._heuristic(nb, goal_cell)
                    heapq.heappush(open_set, (f_score[nb], nb))

        print("[GridDirectGoalStrategy] A* failed to find a path.")
        return []

    # ------------------------------------------------------------------
    # Core control
    # ------------------------------------------------------------------

    def compute_control(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        target_x: float,
        target_y: float,
        obstacles: List[Tuple[float, float]],
    ) -> Tuple[Twist, Optional[Vector3], bool]:
        """
        Main control method.

        1. If robot is within goal_tolerance -> stop & goal_reached True.
        2. Convert obstacles to blocked grid cells.
        3. Compute grid cells for start & goal.
        4. Run A* to get cell path.
        5. Choose the next waypoint (grid node) from that path.
        6. If robot is close enough to that waypoint, STOP (and optionally
           dwell) so we effectively "lock in" each bin.
        7. Otherwise, drive toward that waypoint using DirectGoal-like logic.
        8. If no grid path found, fall back to plain direct-to-goal.
        """
        self._step_counter += 1

        # --- 1. Check goal reached (true goal, not just waypoint) ---
        if self.is_goal_reached(robot_x, robot_y, target_x, target_y):
            if self._step_counter % self.log_throttle_steps == 0:
                print(
                    f"[GridDirectGoalStrategy] Goal reached at "
                    f"({robot_x:.2f},{robot_y:.2f}) ~ ({target_x:.2f},{target_y:.2f})"
                )
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            return cmd, None, True

        # --- 2. Map obstacles to blocked cells ---
        blocked_cells = self._obstacles_to_blocked_cells(obstacles)

        # --- 3. Compute start and goal grid cells ---
        start_cell = self._world_to_cell(robot_x, robot_y)
        goal_cell = self._world_to_cell(target_x, target_y)

        # Ensure start/goal aren't blocked (we still want to stand on them)
        if start_cell in blocked_cells:
            blocked_cells.discard(start_cell)
        if goal_cell in blocked_cells:
            blocked_cells.discard(goal_cell)

        # --- 4. A* path on grid ---
        cell_path = self._plan_path(start_cell, goal_cell, blocked_cells)

        # If no path -> fallback to plain DirectGoal behavior
        if not cell_path:
            if self._step_counter % self.log_throttle_steps == 0:
                print(
                    "[GridDirectGoalStrategy] No grid path, falling back to "
                    "plain DirectGoal behavior."
                )
            return self._direct_goal_control(robot_x, robot_y, robot_yaw, target_x, target_y)

        # --- 5. Choose waypoint (next node on path) ---
        # Typically, cell_path[0] == start_cell. We want the next one if it exists.
        if len(cell_path) >= 2:
            next_cell = cell_path[1]
        else:
            # Already in same cell as goal: just use real goal coords
            next_cell = goal_cell

        waypoint_x, waypoint_y = self._cell_to_world(*next_cell)

        # For logging: sample print
        if self._step_counter % self.log_throttle_steps == 0:
            # Show at most first 5 cells in path in log
            preview_cells = cell_path[:5]
            preview_world = [self._cell_to_world(ix, iy) for (ix, iy) in preview_cells]
            print(
                f"[GridDirectGoalStrategy] robot=({robot_x:.2f},{robot_y:.2f}), "
                f"target=({target_x:.2f},{target_y:.2f}), "
                f"start_cell={start_cell}, goal_cell={goal_cell}, "
                f"blocked_cells={len(blocked_cells)}"
            )
            print(
                f"[GridDirectGoalStrategy] cell_path_len={len(cell_path)}, "
                f"path_preview_world={[(round(px,2), round(py,2)) for (px,py) in preview_world]}"
            )
            print(
                f"[GridDirectGoalStrategy] Next waypoint (grid center) = "
                f"({waypoint_x:.2f},{waypoint_y:.2f})"
            )

        # --- 6. Check if we've "reached" this bin and should stop/dwell ---
        dist_to_wp = math.hypot(waypoint_x - robot_x, waypoint_y - robot_y)

        if dist_to_wp < self.waypoint_tolerance:
            # We're sufficiently close to this bin's center.
            heading_vec = Vector3()
            heading_vec.x = 0.0
            heading_vec.y = 0.0
            heading_vec.z = 0.0

            # Handle dwell if configured
            if self.dwell_steps_required > 0:
                if not self._in_dwell:
                    self._in_dwell = True
                    self._dwell_steps_done = 0
                    print(
                        f"[GridDirectGoalStrategy] Reached waypoint at "
                        f"({waypoint_x:.2f},{waypoint_y:.2f}), entering dwell for "
                        f"{self.dwell_steps_required} steps."
                    )

                self._dwell_steps_done += 1

                if self._dwell_steps_done >= self.dwell_steps_required:
                    print(
                        f"[GridDirectGoalStrategy] Finished dwell at waypoint "
                        f"({waypoint_x:.2f},{waypoint_y:.2f})."
                    )
                    self._in_dwell = False
                    self._dwell_steps_done = 0

                # While dwelling we just stay stopped.
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                return cmd, heading_vec, False

            else:
                # No dwell configured: just stop for this cycle.
                if self._step_counter % self.log_throttle_steps == 0:
                    print(
                        f"[GridDirectGoalStrategy] Reached waypoint at "
                        f"({waypoint_x:.2f},{waypoint_y:.2f}), stopping (no dwell)."
                    )
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                return cmd, heading_vec, False

        # If we're not yet close enough to the bin center, ensure dwell
        # state is cleared and drive toward it.
        if self._in_dwell:
            # We've moved away from the waypoint; reset dwell state.
            self._in_dwell = False
            self._dwell_steps_done = 0

        # --- 7. Drive toward waypoint using DirectGoal-style control ---
        cmd, heading_vec, _ = self._direct_goal_control(
            robot_x, robot_y, robot_yaw, waypoint_x, waypoint_y
        )
        # We do *not* claim the final goal is reached here.
        return cmd, heading_vec, False

    # ------------------------------------------------------------------
    # Direct-goal-like sub-controller
    # ------------------------------------------------------------------

    def _direct_goal_control(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        target_x: float,
        target_y: float,
    ) -> Tuple[Twist, Optional[Vector3], bool]:
        """
        Same behavior as DirectGoalStrategy: orient toward target, then
        move forward if heading error is small enough.
        """
        dx = target_x - robot_x
        dy = target_y - robot_y

        target_heading = math.atan2(dy, dx)
        heading_error = self._angle_diff(target_heading, robot_yaw)

        angular_velocity = self.angular_gain * heading_error

        if abs(heading_error) < self.min_angular_error_for_forward:
            linear_velocity = self.max_linear_velocity
        else:
            linear_velocity = 0.0

        cmd = Twist()
        cmd.linear.x = linear_velocity
        cmd.angular.z = angular_velocity

        heading_vec = Vector3()
        heading_vec.x = math.cos(target_heading)
        heading_vec.y = math.sin(target_heading)
        heading_vec.z = 0.0

        return cmd, heading_vec, False

    # ------------------------------------------------------------------
    # Interface methods
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset any internal state."""
        print("[GridDirectGoalStrategy] Reset called.")
        self._step_counter = 0
        self._reset_dwell_state()

    def is_goal_reached(
        self,
        robot_x: float,
        robot_y: float,
        target_x: float,
        target_y: float,
    ) -> bool:
        """Check if robot is within goal tolerance of the FINAL goal."""
        dx = target_x - robot_x
        dy = target_y - robot_y
        distance = math.sqrt(dx * dx + dy * dy)
        return distance < self.goal_tolerance

    def get_parameters(self) -> dict:
        """Get current strategy parameters."""
        return {
            "goal_tolerance": self.goal_tolerance,
            "max_linear_velocity": self.max_linear_velocity,
            "angular_gain": self.angular_gain,
            "min_angular_error_for_forward": self.min_angular_error_for_forward,
            "grid_resolution": self.grid_resolution,
            "obstacle_inflation_radius": self.obstacle_inflation_radius,
            "waypoint_tolerance": self.waypoint_tolerance,
            "dwell_time_at_waypoint": self.dwell_time_at_waypoint,
            "control_dt": self.control_dt,
            "nx": self.nx,
            "ny": self.ny,
        }

    def set_parameters(self, params: dict) -> None:
        """Update strategy parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
                if key in ("grid_resolution",):
                    # If grid resolution changes at runtime, recompute grid sizes
                    self.nx = max(1, int(FIELD_WIDTH / self.grid_resolution))
                    self.ny = max(1, int(FIELD_LENGTH / self.grid_resolution))
                    print(
                        f"[GridDirectGoalStrategy] grid_resolution updated to "
                        f"{self.grid_resolution:.2f}, nx={self.nx}, ny={self.ny}"
                    )
                if key in ("dwell_time_at_waypoint", "control_dt"):
                    # Recompute dwell steps if timing-related params change
                    self.dwell_steps_required = (
                        int(self.dwell_time_at_waypoint / self.control_dt)
                        if self.dwell_time_at_waypoint > 0.0
                        else 0
                    )
                    print(
                        f"[GridDirectGoalStrategy] dwell_time={self.dwell_time_at_waypoint:.2f}s, "
                        f"control_dt={self.control_dt:.3f}s -> "
                        f"dwell_steps_required={self.dwell_steps_required}"
                    )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Compute the shortest angular difference between two angles."""
        d = a - b
        return math.atan2(math.sin(d), math.cos(d))
