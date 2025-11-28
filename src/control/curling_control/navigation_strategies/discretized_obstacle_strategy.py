"""
Grid-based Direct Goal Navigation Strategy (global/local version).

High-level idea
---------------
- Discretize the field into a regular grid.
- Maintain a *global* shortest path over that grid from the robot's
  current grid cell to the goal cell (A* over free cells).
- Obstacles (from follower.py) are projected into grid cells and those
  cells are removed from the search space (optionally inflated).
- The controller only ever drives toward the *next* cell on that global
  grid path (local controller).
- At each grid cell boundary, the robot stops (and optionally dwells),
  then the global path is recomputed using the latest obstacle info.

Behavior details
----------------
- "Global path": the full list of cells from start to goal.
- "Local target": the adjacent cell on that path that we are currently
  driving toward.
- When we ENTER the local target cell:
    * If it's an intermediate cell (not the goal cell):
        - we stop,
        - optionally dwell for dwell_time_at_waypoint seconds,
        - then mark the global path as invalid so that on the next call
          we recompute A* from this new cell using updated obstacles.
    * If it's the goal cell:
        - we stop using grid centers and instead drive directly to the
          *true* goal position (not just the cell center) using a
          DirectGoal-style controller until within goal_tolerance.

Debug info
----------
The strategy exposes get_debug_info(), which returns:
    - path_world:   list of (x, y) centers along the global grid path,
    - blocked_world: list of (x, y) centers of blocked cells,
    - current_target_center: (x, y) of the current local target cell.
Your follower node already publishes these to /control/grid_path,
/control/grid_blocked_cells, and /control/grid_target_cell.

NOTE: Adjust FIELD_WIDTH and FIELD_LENGTH to your actual field if
      needed (e.g. the lab area used for your UGV Rover exercises). :contentReference[oaicite:0]{index=0}
"""

import math
import heapq
from typing import List, Tuple, Optional, Dict, Set

from geometry_msgs.msg import Twist, Vector3

from .base_strategy import BaseNavigationStrategy


# ---------------------------------------------------------------------------
# Field constants (in meters) - tune to your arena dimensions if needed.
# ---------------------------------------------------------------------------

FIELD_WIDTH: float = 6.0   # along +x
FIELD_LENGTH: float = 9.0  # along +y


class GridDirectGoalStrategyObstacle(BaseNavigationStrategy):
    """
    Grid-based navigation strategy that:
      - computes a global A* path over a grid,
      - follows it cell-by-cell with a DirectGoal sub-controller,
      - re-plans whenever we reach a new cell or obstacles change.
    """

    def __init__(
        self,
        goal_tolerance: float = 0.1,
        max_linear_velocity: float = 0.2,
        angular_gain: float = 1.0,
        min_angular_error_for_forward: float = 0.3,  # ~17 degrees
        grid_resolution: float = 0.5,
        obstacle_inflation_radius: float = 0.0,
        waypoint_tolerance: float = 0.25,             # kept for compatibility
        dwell_time_at_waypoint: float = 0.0,          # seconds
        control_dt: float = 0.05,
        log_throttle_steps: int = 20,
        *args,
        **kwargs,
    ):
        """
        Args:
            goal_tolerance: Distance threshold to consider goal reached.
            max_linear_velocity: Forward velocity when moving.
            angular_gain: P-gain for angular velocity control.
            min_angular_error_for_forward: Heading error threshold under
                which we allow forward motion.
            grid_resolution: Size (meters) of each grid cell.
            obstacle_inflation_radius: Extra radius around obstacles; all
                cells within that radius are blocked.
            waypoint_tolerance: Retained for API compatibility; not used
                in the new cell-based logic.
            dwell_time_at_waypoint: Pause duration at each intermediate
                cell (0.0 means "no dwell" but we still briefly stop and
                re-plan).
            control_dt: Control-loop dt for converting dwell time to
                discrete steps (should match follower.py timer).
            log_throttle_steps: How many control steps between log
                prints (>=1).
        """
        # --- primary control params ---
        self.goal_tolerance = goal_tolerance
        self.max_linear_velocity = max_linear_velocity
        self.angular_gain = angular_gain
        self.min_angular_error_for_forward = min_angular_error_for_forward

        # --- grid geometry ---
        self.grid_resolution = grid_resolution
        self.nx = max(1, int(FIELD_WIDTH / grid_resolution))
        self.ny = max(1, int(FIELD_LENGTH / grid_resolution))

        self.obstacle_inflation_radius = max(0.0, obstacle_inflation_radius)

        # --- dwell behavior ---
        self.waypoint_tolerance = waypoint_tolerance
        self.dwell_time_at_waypoint = dwell_time_at_waypoint
        self.control_dt = control_dt
        self.dwell_steps_required = (
            int(dwell_time_at_waypoint / control_dt)
            if dwell_time_at_waypoint > 0.0
            else 0
        )
        self._reset_dwell_state()

        # --- global path state ---
        self._global_path_cells: List[Tuple[int, int]] = []
        self._global_path_valid: bool = False
        self._last_planning_start_cell: Optional[Tuple[int, int]] = None
        self._last_planning_goal_cell: Optional[Tuple[int, int]] = None
        self._last_planning_blocked: Set[Tuple[int, int]] = set()

        # --- local target state (adjacent cell along global path) ---
        self._current_target_index: Optional[int] = None
        self._current_target_cell: Optional[Tuple[int, int]] = None
        self._current_target_center: Optional[Tuple[float, float]] = None

        # --- debug / logging ---
        self.log_throttle_steps = max(1, log_throttle_steps)
        self._step_counter: int = 0

        self._debug_last_blocked_cells: Set[Tuple[int, int]] = set()
        self._debug_last_path_cells: List[Tuple[int, int]] = []
        self._debug_last_start_cell: Optional[Tuple[int, int]] = None
        self._debug_last_goal_cell: Optional[Tuple[int, int]] = None
        self._debug_last_target_cell: Optional[Tuple[int, int]] = None
        self._debug_last_target_center: Optional[Tuple[float, float]] = None

        print(
            f"[GridDirectGoalStrategy] Init: grid_resolution={self.grid_resolution:.2f}m, "
            f"nx={self.nx}, ny={self.ny}, goal_tolerance={self.goal_tolerance:.2f}, "
            f"dwell={self.dwell_time_at_waypoint:.2f}s "
            f"({self.dwell_steps_required} steps)"
        )

    # ------------------------------------------------------------------
    # Dwell helpers
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
        Map grid cell indices (ix, iy) to the world coordinates of the cell
        center (x, y).
        """
        x = (ix + 0.5) * self.grid_resolution
        y = (iy + 0.5) * self.grid_resolution

        x = min(max(x, 0.0), FIELD_WIDTH)
        y = min(max(y, 0.0), FIELD_LENGTH)
        return x, y

    def _obstacles_to_blocked_cells(
        self, obstacles: List[Tuple[float, float]]
    ) -> Set[Tuple[int, int]]:
        """
        Convert obstacle world positions into a set of blocked grid cells.
        If obstacle_inflation_radius > 0, also block neighboring cells
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
        """Euclidean distance between centers of two grid cells (meters)."""
        (ax, ay) = self._cell_to_world(*a)
        (bx, by) = self._cell_to_world(*b)
        return math.hypot(bx - ax, by - ay)

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Heuristic for A* (Euclidean distance)."""
        return self._dist_cells(a, b)

    def _plan_path(
        self,
        start_cell: Tuple[int, int],
        goal_cell: Tuple[int, int],
        blocked: Set[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """
        A* search on the grid, avoiding blocked cells.
        Returns [start_cell, ..., goal_cell] or [] if no path exists.
        """
        if start_cell == goal_cell:
            return [start_cell]

        if goal_cell in blocked:
            print("[GridDirectGoalStrategy] Goal cell is blocked; no grid path.")
            return []

        open_set: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, start_cell))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start_cell: 0.0}
        f_score: Dict[Tuple[int, int], float] = {
            start_cell: self._heuristic(start_cell, goal_cell)
        }

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
    # Global path management
    # ------------------------------------------------------------------

    def _should_replan(
        self,
        start_cell: Tuple[int, int],
        goal_cell: Tuple[int, int],
        blocked: Set[Tuple[int, int]],
    ) -> bool:
        """Decide if we need to recompute the global grid path."""
        if not self._global_path_valid:
            return True
        if self._last_planning_start_cell != start_cell:
            return True
        if self._last_planning_goal_cell != goal_cell:
            return True
        if blocked != self._last_planning_blocked:
            return True
        if not self._global_path_cells:
            return True
        return False

    def _replan_global_path(
        self,
        start_cell: Tuple[int, int],
        goal_cell: Tuple[int, int],
        blocked: Set[Tuple[int, int]],
    ) -> None:
        """Run A* and store the global path (or mark invalid if none)."""
        path = self._plan_path(start_cell, goal_cell, blocked)

        if not path:
            self._global_path_cells = []
            self._global_path_valid = False
            self._current_target_index = None
            self._current_target_cell = None
            self._current_target_center = None
            if self._step_counter % self.log_throttle_steps == 0:
                print("[GridDirectGoalStrategy] No grid path; stopping.")
            return

        self._global_path_cells = path
        self._global_path_valid = True
        self._last_planning_start_cell = start_cell
        self._last_planning_goal_cell = goal_cell
        self._last_planning_blocked = set(blocked)

        # reset local target; will be re-selected below
        self._current_target_index = None
        self._current_target_cell = None
        self._current_target_center = None

        if self._step_counter % self.log_throttle_steps == 0:
            preview_cells = path[:5]
            preview_world = [self._cell_to_world(ix, iy) for (ix, iy) in preview_cells]
            print(
                f"[GridDirectGoalStrategy] Replanned global path with "
                f"{len(path)} cells. Start={start_cell}, goal={goal_cell}, "
                f"preview_world={[(round(px,2), round(py,2)) for (px,py) in preview_world]}"
            )

    def _index_of_cell_in_path(
        self, cell: Tuple[int, int], path: List[Tuple[int, int]]
    ) -> int:
        """Find exact index of cell in path, or nearest-by-distance index."""
        if cell in path:
            return path.index(cell)

        # Not exactly on the path; pick nearest cell in path
        best_idx = 0
        best_dist = float("inf")
        cx, cy = self._cell_to_world(*cell)
        for i, c in enumerate(path):
            px, py = self._cell_to_world(*c)
            d = math.hypot(px - cx, py - cy)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _update_local_target(
        self,
        start_cell: Tuple[int, int],
        goal_cell: Tuple[int, int],
    ) -> None:
        """Choose the next cell on the global path as the local target."""
        if not self._global_path_valid or not self._global_path_cells:
            self._current_target_index = None
            self._current_target_cell = None
            self._current_target_center = None
            return

        idx_on_path = self._index_of_cell_in_path(start_cell, self._global_path_cells)

        if idx_on_path >= len(self._global_path_cells) - 1:
            # Already at (or beyond) the final cell; we treat this as the
            # "goal cell" case and let DirectGoal refine to the true goal.
            self._current_target_index = idx_on_path
        else:
            # Local target is the next cell on the path
            self._current_target_index = idx_on_path + 1

        self._current_target_cell = self._global_path_cells[self._current_target_index]
        self._current_target_center = self._cell_to_world(*self._current_target_cell)

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
    ):
        """
        Main control:
          1. Check if within goal_tolerance -> stop.
          2. Convert obstacles to blocked grid cells.
          3. Convert robot & goal to start/goal cells.
          4. Re-plan global grid path if needed.
          5. Pick adjacent cell along that path as local target.
          6. If local target cell == goal cell -> DirectGoal to true goal.
          7. Else drive towards local target cell center.
          8. When we ENTER that cell, stop/dwell, then mark the global
             path invalid so next cycle re-plans with updated obstacles.
        """
        self._step_counter += 1

        # --- 1. Check *true* goal distance ---
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

        # --- 2. Obstacles -> blocked cells ---
        blocked_cells = self._obstacles_to_blocked_cells(obstacles)
        self._debug_last_blocked_cells = set(blocked_cells)

        # --- 3. Start & goal cells ---
        start_cell = self._world_to_cell(robot_x, robot_y)
        goal_cell = self._world_to_cell(target_x, target_y)

        # Allow standing in start/goal even if detected as blocked
        blocked_cells.discard(start_cell)
        blocked_cells.discard(goal_cell)

        # --- 4. Global path re-planning ---
        if self._should_replan(start_cell, goal_cell, blocked_cells):
            self._replan_global_path(start_cell, goal_cell, blocked_cells)

        # If planning failed, stop but don't claim goal reached
        if not self._global_path_valid or not self._global_path_cells:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            # debug
            self._debug_last_path_cells = []
            self._debug_last_start_cell = start_cell
            self._debug_last_goal_cell = goal_cell
            self._debug_last_target_cell = None
            self._debug_last_target_center = None
            return cmd, None, False

        # --- 5. Choose local target cell along global path ---
        self._update_local_target(start_cell, goal_cell)

        if self._current_target_cell is None or self._current_target_center is None:
            # Shouldn't happen, but be safe
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            return cmd, None, False

        target_cell = self._current_target_cell
        waypoint_x, waypoint_y = self._current_target_center

        # Debug bookkeeping
        self._debug_last_path_cells = list(self._global_path_cells)
        self._debug_last_start_cell = start_cell
        self._debug_last_goal_cell = goal_cell
        self._debug_last_target_cell = target_cell
        self._debug_last_target_center = (waypoint_x, waypoint_y)

        if self._step_counter % self.log_throttle_steps == 0:
            print(
                f"[GridDirectGoalStrategy] robot=({robot_x:.2f},{robot_y:.2f}), "
                f"target=({target_x:.2f},{target_y:.2f}), "
                f"start_cell={start_cell}, goal_cell={goal_cell}, "
                f"blocked_cells={len(blocked_cells)}"
            )
            print(
                f"[GridDirectGoalStrategy] local target cell={target_cell}, "
                f"center=({waypoint_x:.2f},{waypoint_y:.2f})"
            )

        in_target_cell = (start_cell == target_cell)
        is_final_cell = (target_cell == goal_cell)

        # --- 6. Final cell -> DirectGoal to the *true* goal pose ---
        if is_final_cell:
            cmd, heading_vec, _ = self._direct_goal_control(
                robot_x, robot_y, robot_yaw, target_x, target_y
            )
            return cmd, heading_vec, False

        # --- 7. Intermediate cell: reached? then stop & (optionally) dwell ---
        if in_target_cell:
            heading_vec = Vector3()
            heading_vec.x = 0.0
            heading_vec.y = 0.0
            heading_vec.z = 0.0

            if self.dwell_steps_required > 0:
                if not self._in_dwell:
                    self._in_dwell = True
                    self._dwell_steps_done = 0
                    print(
                        f"[GridDirectGoalStrategy] Entered cell={target_cell}, "
                        f"center=({waypoint_x:.2f},{waypoint_y:.2f}), "
                        f"starting dwell for {self.dwell_steps_required} steps."
                    )

                self._dwell_steps_done += 1

                if self._dwell_steps_done >= self.dwell_steps_required:
                    print(
                        f"[GridDirectGoalStrategy] Finished dwell at cell={target_cell}. "
                        f"Will re-plan next cycle."
                    )
                    self._reset_dwell_state()
                    # force re-plan next call (new start cell + updated obstacles)
                    self._global_path_valid = False

                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                return cmd, heading_vec, False

            else:
                # No dwell requested: still stop once, then re-plan next cycle
                if self._step_counter % self.log_throttle_steps == 0:
                    print(
                        f"[GridDirectGoalStrategy] Reached cell={target_cell}, "
                        f"center=({waypoint_x:.2f},{waypoint_y:.2f}), "
                        f"stopping and forcing re-plan (no dwell)."
                    )
                self._global_path_valid = False
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                return cmd, heading_vec, False

        # If we're not in the target cell, ensure dwell state is cleared
        if self._in_dwell:
            self._reset_dwell_state()

        # --- 8. Drive toward local target cell center (DirectGoal-style) ---
        cmd, heading_vec, _ = self._direct_goal_control(
            robot_x, robot_y, robot_yaw, waypoint_x, waypoint_y
        )
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
    ):
        """Orient toward target, then move forward when heading error small."""
        dx = target_x - robot_x
        dy = target_y - robot_y

        target_heading = math.atan2(dy, dx)
        heading_error = self._angle_diff(target_heading, robot_yaw)

        angular_velocity = self.angular_gain * heading_error
        linear_velocity = (
            self.max_linear_velocity
            if abs(heading_error) < self.min_angular_error_for_forward
            else 0.0
        )

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
        """Reset internal state."""
        print("[GridDirectGoalStrategy] Reset called.")
        self._step_counter = 0
        self._reset_dwell_state()

        self._global_path_cells = []
        self._global_path_valid = False
        self._last_planning_start_cell = None
        self._last_planning_goal_cell = None
        self._last_planning_blocked.clear()

        self._current_target_index = None
        self._current_target_cell = None
        self._current_target_center = None

        self._debug_last_blocked_cells.clear()
        self._debug_last_path_cells = []
        self._debug_last_start_cell = None
        self._debug_last_goal_cell = None
        self._debug_last_target_cell = None
        self._debug_last_target_center = None

    def is_goal_reached(
        self,
        robot_x: float,
        robot_y: float,
        target_x: float,
        target_y: float,
    ) -> bool:
        """Check if robot is within goal_tolerance of the final goal."""
        dx = target_x - robot_x
        dy = target_y - robot_y
        distance = math.hypot(dx, dy)
        return distance < self.goal_tolerance

    def get_parameters(self) -> dict:
        """Return current tunable parameters (for debugging / dynamic reconfig)."""
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
        """Update strategy parameters at runtime."""
        for key, value in params.items():
            if not hasattr(self, key):
                continue
            setattr(self, key, value)

            if key == "grid_resolution":
                self.nx = max(1, int(FIELD_WIDTH / self.grid_resolution))
                self.ny = max(1, int(FIELD_LENGTH / self.grid_resolution))
                print(
                    f"[GridDirectGoalStrategy] grid_resolution updated to "
                    f"{self.grid_resolution:.2f}, nx={self.nx}, ny={self.ny}"
                )
                # force re-plan with new grid
                self._global_path_valid = False

            if key in ("dwell_time_at_waypoint", "control_dt"):
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
    # Debug / visualization API
    # ------------------------------------------------------------------

    def get_debug_info(self) -> dict:
        """
        Return latest grid planning info for visualization.

        Keys:
          grid_resolution: float
          nx, ny: int
          start_cell, goal_cell: (ix, iy) or None
          path_cells: List[(ix, iy)]
          path_world: List[(x, y)]
          blocked_cells: List[(ix, iy)]
          blocked_world: List[(x, y)]
          current_target_cell: (ix, iy) or None
          current_target_center: (x, y) or None
        """
        path_world = [
            self._cell_to_world(ix, iy) for (ix, iy) in self._debug_last_path_cells
        ]
        blocked_world = [
            self._cell_to_world(ix, iy) for (ix, iy) in self._debug_last_blocked_cells
        ]

        return {
            "grid_resolution": self.grid_resolution,
            "nx": self.nx,
            "ny": self.ny,
            "start_cell": self._debug_last_start_cell,
            "goal_cell": self._debug_last_goal_cell,
            "path_cells": list(self._debug_last_path_cells),
            "path_world": path_world,
            "blocked_cells": list(self._debug_last_blocked_cells),
            "blocked_world": blocked_world,
            "current_target_cell": self._debug_last_target_cell,
            "current_target_center": self._debug_last_target_center,
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Shortest angular difference between two angles."""
        d = a - b
        return math.atan2(math.sin(d), math.cos(d))
