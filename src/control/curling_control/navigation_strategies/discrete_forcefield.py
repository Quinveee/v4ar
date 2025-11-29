#!/usr/bin/env python3
"""
Hybrid Grid + Potential Field Navigation Strategy.

High-level idea
---------------
- Maintain a *global* shortest path over a discrete grid (A* over free cells).
- Maintain a *persistent* obstacle map with voting (rover detections stick
  around but noise gets forgotten).
- Use a *local potential field* controller to generate smooth motion commands:
    * attractive field toward:
        - the next grid cell along the global path (local guidance),
        - the final goal (global bias),
    * repulsive field from:
        - confirmed obstacle cells on the grid (persistent),
        - raw obstacle detections (fast reaction).
- Detect local minima / "stuck" conditions (not making progress toward goal)
  and inject small random perturbations into the potential field to escape.

So you get:
- Global + local grid decisions (A* + next-cell path guidance),
- Persistent grid obstacles,
- Potential-field style smooth motion with local minimum escape.

This class is intended as a drop-in alternative to:
- PotentialFieldStrategy
- GridDirectGoalStrategyObstacle
"""

import math
import heapq
import time
from typing import List, Tuple, Optional, Dict, Set

import numpy as np
from geometry_msgs.msg import Twist, Vector3

from .base_strategy import BaseNavigationStrategy


# ---------------------------------------------------------------------------
# Field constants (in meters) - tune to your arena dimensions if needed.
# ---------------------------------------------------------------------------

FIELD_WIDTH: float = 6.0   # along +x
FIELD_LENGTH: float = 9.0  # along +y


class GridPotentialFieldStrategy(BaseNavigationStrategy):
    """
    Hybrid grid + potential field navigation strategy.

    - Uses a grid and persistent obstacle voting just like
      GridDirectGoalStrategyObstacle.
    - Uses a potential-field style controller (like PotentialFieldStrategy)
      to generate local commands, biased by the global A* path.
    """

    def __init__(
        self,
        # --- global / grid params ---
        goal_tolerance: float = 0.1,
        max_linear_velocity: float = 0.2,
        angular_gain: float = 0.8,
        min_angular_error_for_forward: float = 0.3,  # ~17 degrees
        grid_resolution: float = 0.5,
        obstacle_inflation_radius: float = 0.0,
        control_dt: float = 0.05,
        log_throttle_steps: int = 20,

        # --- potential field params ---
        safe_distance: float = 1.2,
        repulse_strength: float = 1.5,
        min_speed_scale: float = 0.1,
        path_attract_weight: float = 1.0,
        goal_attract_weight: float = 0.4,

        # --- local minimum detection ---
        progress_threshold: float = 0.02,   # m progress threshold
        stuck_time_threshold: float = 2.0,  # seconds before considered stuck
        noise_strength: float = 0.3,        # magnitude of random perturbation

        *args,
        **kwargs,
    ):
        """
        Args (most important):
            goal_tolerance: distance at which we declare the goal reached.
            max_linear_velocity: forward speed scale.
            angular_gain: P-gain for orienting toward the resultant field vector.
            min_angular_error_for_forward: if |heading_error| is larger
                than this, we stop linear motion and just rotate.
            grid_resolution: size of each grid cell in meters.
            obstacle_inflation_radius: inflation of blocked cells (clamped
                to at most 1-cell radius).
            safe_distance: radius within which obstacles generate repulsive
                forces in the potential field.
            repulse_strength: coefficient for repulsive forces.
            min_speed_scale: minimum speed factor when repulsed.
            path_attract_weight: weight for attraction toward the next
                grid cell along the global path.
            goal_attract_weight: extra attraction toward the final goal.
            progress_threshold, stuck_time_threshold, noise_strength:
                parameters for local minimum detection and escape.
        """
        # ------------------------
        # primary control params
        # ------------------------
        self.goal_tolerance = goal_tolerance
        self.max_linear_velocity = max_linear_velocity
        self.angular_gain = angular_gain
        self.min_angular_error_for_forward = min_angular_error_for_forward

        # ------------------------
        # grid geometry
        # ------------------------
        self.grid_resolution = grid_resolution
        self.nx = max(1, int(FIELD_WIDTH / grid_resolution))
        self.ny = max(1, int(FIELD_LENGTH / grid_resolution))

        clamped_inflation = min(
            max(0.0, obstacle_inflation_radius),
            1.01 * self.grid_resolution,  # at most a 1-cell radius
        )
        self.obstacle_inflation_radius = clamped_inflation

        self.control_dt = control_dt

        # ------------------------
        # potential field params
        # ------------------------
        self.safe_distance = safe_distance
        self.repulse_strength = repulse_strength
        self.min_speed_scale = min_speed_scale
        self.path_attract_weight = path_attract_weight
        self.goal_attract_weight = goal_attract_weight

        # local minimum detection
        self.progress_threshold = progress_threshold
        self.stuck_time_threshold = stuck_time_threshold
        self.noise_strength = noise_strength
        self.prev_goal_distance: Optional[float] = None
        self.last_progress_time: float = time.time()
        self.is_stuck: bool = False

        # ------------------------
        # global path state
        # ------------------------
        self._global_path_cells: List[Tuple[int, int]] = []
        self._global_path_valid: bool = False
        self._last_planning_start_cell: Optional[Tuple[int, int]] = None
        self._last_planning_goal_cell: Optional[Tuple[int, int]] = None
        self._last_planning_blocked: Set[Tuple[int, int]] = set()

        # local target (cell on path)
        self._current_target_index: Optional[int] = None
        self._current_target_cell: Optional[Tuple[int, int]] = None
        self._current_target_center: Optional[Tuple[float, float]] = None

        # ------------------------
        # obstacle voting grid
        # ------------------------
        # cell -> {"votes": int, "age": int}
        self._obstacle_cells: Dict[Tuple[int, int], Dict[str, int]] = {}

        self.obstacle_confirm_votes: int = 5          # votes to confirm a cell
        self.obstacle_max_age_steps: int = 200        # steps before forgetting (~10s at 20 Hz)
        self.obstacle_max_votes: int = 50             # cap votes

        # ------------------------
        # debug / logging
        # ------------------------
        self.log_throttle_steps = max(1, log_throttle_steps)
        self._step_counter: int = 0

        self._debug_last_blocked_cells: Set[Tuple[int, int]] = set()
        self._debug_last_path_cells: List[Tuple[int, int]] = []
        self._debug_last_start_cell: Optional[Tuple[int, int]] = None
        self._debug_last_goal_cell: Optional[Tuple[int, int]] = None
        self._debug_last_target_cell: Optional[Tuple[int, int]] = None
        self._debug_last_target_center: Optional[Tuple[float, float]] = None

        print(
            f"[GridPotentialFieldStrategy] Init: grid_resolution={self.grid_resolution:.2f}m, "
            f"nx={self.nx}, ny={self.ny}, goal_tolerance={self.goal_tolerance:.2f}, "
            f"safe_distance={self.safe_distance:.2f}m, "
            f"obstacle_inflation_radius={self.obstacle_inflation_radius:.2f}m"
        )

    # ------------------------------------------------------------------
    # Utility: world <-> grid
    # ------------------------------------------------------------------

    def _world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Map world coordinates (meters) to integer grid cell indices (ix, iy)."""
        xr = min(max(x, 0.0), FIELD_WIDTH - 1e-6)
        yr = min(max(y, 0.0), FIELD_LENGTH - 1e-6)

        ix = int(xr / self.grid_resolution)
        iy = int(yr / self.grid_resolution)

        ix = max(0, min(self.nx - 1, ix))
        iy = max(0, min(self.ny - 1, iy))
        return ix, iy

    def _cell_to_world(self, ix: int, iy: int) -> Tuple[float, float]:
        """Map grid cell indices (ix, iy) to the world coordinates of the cell center."""
        x = (ix + 0.5) * self.grid_resolution
        y = (iy + 0.5) * self.grid_resolution

        x = min(max(x, 0.0), FIELD_WIDTH)
        y = min(max(y, 0.0), FIELD_LENGTH)
        return x, y

    # ------------------------------------------------------------------
    # Obstacle voting grid
    # ------------------------------------------------------------------

    def _update_obstacle_votes(self, raw_obstacles: List[Tuple[float, float]]) -> None:
        """
        Update the voting grid with the latest raw obstacle detections.

        - All existing cells age by 1.
        - Cells that are detected get a vote increment and age reset.
        - Cells that have not been seen for too long are removed.
        """
        # Age all existing
        for rec in self._obstacle_cells.values():
            rec["age"] += 1

        # Increment votes for newly seen cells
        for (ox, oy) in raw_obstacles:
            ix, iy = self._world_to_cell(ox, oy)
            cell = (ix, iy)
            rec = self._obstacle_cells.get(cell)
            if rec is None:
                self._obstacle_cells[cell] = {"votes": 1, "age": 0}
            else:
                rec["votes"] = min(rec["votes"] + 1, self.obstacle_max_votes)
                rec["age"] = 0

        # Remove old cells
        to_delete: List[Tuple[int, int]] = []
        for cell, rec in self._obstacle_cells.items():
            if rec["age"] > self.obstacle_max_age_steps:
                to_delete.append(cell)

        for cell in to_delete:
            del self._obstacle_cells[cell]

    def _votes_to_blocked_cells(self) -> Set[Tuple[int, int]]:
        """
        Convert the voting grid into a set of blocked cells.

        A cell is blocked if it has votes >= obstacle_confirm_votes.
        If obstacle_inflation_radius > 0, block neighbors within that radius.
        """
        blocked: Set[Tuple[int, int]] = set()
        if not self._obstacle_cells:
            return blocked

        inflation_cells = int(self.obstacle_inflation_radius / self.grid_resolution)

        for (ix, iy), rec in self._obstacle_cells.items():
            if rec["votes"] < self.obstacle_confirm_votes:
                continue

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
    # A* grid planner (global path)
    # ------------------------------------------------------------------

    def _neighbors(self, ix: int, iy: int) -> List[Tuple[int, int]]:
        """8-connected neighborhood."""
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
        """Heuristic for A* (Euclidean)."""
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
            print("[GridPotentialFieldStrategy] Goal cell is blocked; no grid path.")
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
                # reconstruct
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

        print("[GridPotentialFieldStrategy] A* failed to find a path.")
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
                print("[GridPotentialFieldStrategy] No grid path; falling back to pure potential field.")
            return

        self._global_path_cells = path
        self._global_path_valid = True
        self._last_planning_start_cell = start_cell
        self._last_planning_goal_cell = goal_cell
        self._last_planning_blocked = set(blocked)

        self._current_target_index = None
        self._current_target_cell = None
        self._current_target_center = None

        if self._step_counter % self.log_throttle_steps == 0:
            preview_cells = path[:5]
            preview_world = [self._cell_to_world(ix, iy) for (ix, iy) in preview_cells]
            print(
                f"[GridPotentialFieldStrategy] Replanned global path with {len(path)} cells. "
                f"Start={start_cell}, goal={goal_cell}, "
                f"preview_world={[(round(px,2), round(py,2)) for (px,py) in preview_world]}"
            )

    def _index_of_cell_in_path(
        self, cell: Tuple[int, int], path: List[Tuple[int, int]]
    ) -> int:
        """Find exact index of cell in path, or nearest-by-distance index."""
        if cell in path:
            return path.index(cell)

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
    ) -> None:
        """
        Choose the next cell on the global path as the local target.

        If we're at or past the final cell, we keep the last cell as target
        and let the potential field use the true goal as well.
        """
        if not self._global_path_valid or not self._global_path_cells:
            self._current_target_index = None
            self._current_target_cell = None
            self._current_target_center = None
            return

        idx_on_path = self._index_of_cell_in_path(start_cell, self._global_path_cells)

        if idx_on_path >= len(self._global_path_cells) - 1:
            self._current_target_index = idx_on_path
        else:
            self._current_target_index = idx_on_path + 1

        self._current_target_cell = self._global_path_cells[self._current_target_index]
        self._current_target_center = self._cell_to_world(*self._current_target_cell)

    # ------------------------------------------------------------------
    # Potential field controller with grid + raw obstacles
    # ------------------------------------------------------------------

    def _compute_potential_field_control(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        goal_x: float,
        goal_y: float,
        blocked_cells: Set[Tuple[int, int]],
        raw_obstacles: List[Tuple[float, float]],
        local_path_target: Optional[Tuple[float, float]],
    ) -> Tuple[Twist, Vector3, bool]:
        """
        Compute a potential-field-based control.

        Attractive:
            - toward final goal (goal_x, goal_y)
            - optionally toward local_path_target (next cell center on grid path)
        Repulsive:
            - from confirmed obstacle cells (grid)
            - from raw obstacle detections (world coords)
        """
        # --- attractive toward goal + path target ---
        pos = np.array([robot_x, robot_y], dtype=float)

        goal_vec = np.array([goal_x - robot_x, goal_y - robot_y], dtype=float)
        goal_dist = np.linalg.norm(goal_vec)
        goal_dir = goal_vec / (goal_dist + 1e-6)

        attract = self.goal_attract_weight * goal_dir

        if local_path_target is not None:
            lx, ly = local_path_target
            path_vec = np.array([lx - robot_x, ly - robot_y], dtype=float)
            path_dist = np.linalg.norm(path_vec)
            if path_dist > 1e-4:
                path_dir = path_vec / path_dist
                attract += self.path_attract_weight * path_dir

        # --- repulsive forces from grid obstacles ---
        repulse = np.zeros(2, dtype=float)

        for (ix, iy) in blocked_cells:
            ox, oy = self._cell_to_world(ix, iy)
            o_vec = pos - np.array([ox, oy], dtype=float)
            dist = np.linalg.norm(o_vec)
            if dist < 1e-6:
                continue
            if dist < self.safe_distance:
                direction = o_vec / dist
                strength = self.repulse_strength / (dist**2 + 1e-6)
                repulse += direction * strength

        # --- repulsive forces from raw obstacles (fast response) ---
        for (ox, oy) in raw_obstacles:
            o_vec = pos - np.array([ox, oy], dtype=float)
            dist = np.linalg.norm(o_vec)
            if dist < 1e-6:
                continue
            if dist < self.safe_distance:
                direction = o_vec / dist
                strength = self.repulse_strength / (dist**2 + 1e-6)
                repulse += direction * strength

        combined = attract + repulse

        # --- local minimum detection & noise injection (like PotentialFieldStrategy) ---
        current_time = time.time()
        if self.prev_goal_distance is not None:
            delta = self.prev_goal_distance - goal_dist
            if delta < self.progress_threshold:
                # not making significant progress
                if current_time - self.last_progress_time > self.stuck_time_threshold:
                    if not self.is_stuck:
                        self.is_stuck = True
                        print(
                            "[GridPotentialFieldStrategy] Local minimum detected — injecting noise."
                        )
                    noise = np.random.uniform(
                        -self.noise_strength, self.noise_strength, size=2
                    )
                    combined += noise
            else:
                # made progress
                self.last_progress_time = current_time
                self.is_stuck = False

        self.prev_goal_distance = goal_dist

        # --- normalize resultant vector ---
        combined_norm = combined / (np.linalg.norm(combined) + 1e-6)

        target_heading = math.atan2(combined_norm[1], combined_norm[0])
        heading_error = self._angle_diff(target_heading, robot_yaw)

        angular_velocity = self.angular_gain * heading_error

        repulse_mag = float(np.linalg.norm(repulse))
        speed_scale = max(self.min_speed_scale, 1.0 - repulse_mag)
        linear_velocity = self.max_linear_velocity * speed_scale

        # Optional: don't move forward if heading error is large
        if abs(heading_error) > self.min_angular_error_for_forward:
            linear_velocity = 0.0

        cmd = Twist()
        cmd.linear.x = linear_velocity
        cmd.angular.z = angular_velocity

        heading_vec = Vector3()
        heading_vec.x = combined_norm[0]
        heading_vec.y = combined_norm[1]
        heading_vec.z = 0.0

        return cmd, heading_vec, False

    # ------------------------------------------------------------------
    # Main control entry point
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
        Main control:
          1. Check if within goal_tolerance -> stop & goal_reached=True.
          2. Update obstacle voting grid with raw obstacles.
          3. Convert voting grid to blocked cells.
          4. Convert robot & goal to start/goal cells.
          5. Re-plan global grid path if needed.
          6. Select next cell on global path as local target.
          7. Use potential field (goal + local target + obstacles) to compute
             control command.
          8. If no global path exists, fall back to pure potential field
             (goal + obstacles, no path target).
        """
        self._step_counter += 1

        # --- 1. Check true goal distance ---
        if self.is_goal_reached(robot_x, robot_y, target_x, target_y):
            if self._step_counter % self.log_throttle_steps == 0:
                print(
                    f"[GridPotentialFieldStrategy] Goal reached at "
                    f"({robot_x:.2f},{robot_y:.2f}) ~ ({target_x:.2f},{target_y:.2f})"
                )
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            heading_vec = Vector3()
            heading_vec.x = 0.0
            heading_vec.y = 0.0
            heading_vec.z = 0.0
            return cmd, heading_vec, True

        # --- 2. Update obstacle voting grid ---
        self._update_obstacle_votes(obstacles)

        # --- 3. Voting grid -> blocked cells ---
        blocked_cells = self._votes_to_blocked_cells()
        self._debug_last_blocked_cells = set(blocked_cells)

        # --- 4. Start & goal cells ---
        start_cell = self._world_to_cell(robot_x, robot_y)
        goal_cell = self._world_to_cell(target_x, target_y)

        # Allow standing in start/goal even if detected as blocked
        blocked_cells.discard(start_cell)
        blocked_cells.discard(goal_cell)

        # --- 5. Global path re-planning ---
        if self._should_replan(start_cell, goal_cell, blocked_cells):
            self._replan_global_path(start_cell, goal_cell, blocked_cells)

        # --- 6. Determine local path target (if any) ---
        local_target_center: Optional[Tuple[float, float]] = None
        if self._global_path_valid and self._global_path_cells:
            self._update_local_target(start_cell)
            if self._current_target_cell is not None:
                local_target_center = self._current_target_center

        # debug info
        self._debug_last_path_cells = list(self._global_path_cells)
        self._debug_last_start_cell = start_cell
        self._debug_last_goal_cell = goal_cell
        self._debug_last_target_cell = self._current_target_cell
        self._debug_last_target_center = local_target_center

        if self._step_counter % self.log_throttle_steps == 0:
            print(
                f"[GridPotentialFieldStrategy] robot=({robot_x:.2f},{robot_y:.2f}), "
                f"target=({target_x:.2f},{target_y:.2f}), "
                f"start_cell={start_cell}, goal_cell={goal_cell}, "
                f"blocked_cells={len(blocked_cells)}, "
                f"obstacle_cells={len(self._obstacle_cells)}, "
                f"have_path={self._global_path_valid}"
            )
            if local_target_center is not None:
                print(
                    f"[GridPotentialFieldStrategy] local target cell={self._current_target_cell}, "
                    f"center=({local_target_center[0]:.2f},{local_target_center[1]:.2f})"
                )

        # --- 7 & 8. Potential field control (with or without path target) ---
        cmd, heading_vec, _ = self._compute_potential_field_control(
            robot_x=robot_x,
            robot_y=robot_y,
            robot_yaw=robot_yaw,
            goal_x=target_x,
            goal_y=target_y,
            blocked_cells=blocked_cells,
            raw_obstacles=obstacles,
            local_path_target=local_target_center,
        )
        return cmd, heading_vec, False

    # ------------------------------------------------------------------
    # Interface methods
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset internal state."""
        print("[GridPotentialFieldStrategy] Reset called.")

        self._step_counter = 0

        # potential-field local-min state
        self.prev_goal_distance = None
        self.last_progress_time = time.time()
        self.is_stuck = False

        # global path state
        self._global_path_cells = []
        self._global_path_valid = False
        self._last_planning_start_cell = None
        self._last_planning_goal_cell = None
        self._last_planning_blocked.clear()

        # local target
        self._current_target_index = None
        self._current_target_cell = None
        self._current_target_center = None

        # debug
        self._debug_last_blocked_cells.clear()
        self._debug_last_path_cells = []
        self._debug_last_start_cell = None
        self._debug_last_goal_cell = None
        self._debug_last_target_cell = None
        self._debug_last_target_center = None

        # obstacle voting grid
        self._obstacle_cells.clear()

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
        """Return current tunable parameters."""
        return {
            # global / grid
            "goal_tolerance": self.goal_tolerance,
            "max_linear_velocity": self.max_linear_velocity,
            "angular_gain": self.angular_gain,
            "min_angular_error_for_forward": self.min_angular_error_for_forward,
            "grid_resolution": self.grid_resolution,
            "obstacle_inflation_radius": self.obstacle_inflation_radius,
            "control_dt": self.control_dt,
            "nx": self.nx,
            "ny": self.ny,
            "obstacle_confirm_votes": self.obstacle_confirm_votes,
            "obstacle_max_age_steps": self.obstacle_max_age_steps,
            "obstacle_max_votes": self.obstacle_max_votes,
            # potential field
            "safe_distance": self.safe_distance,
            "repulse_strength": self.repulse_strength,
            "min_speed_scale": self.min_speed_scale,
            "path_attract_weight": self.path_attract_weight,
            "goal_attract_weight": self.goal_attract_weight,
            # local minimum detection
            "progress_threshold": self.progress_threshold,
            "stuck_time_threshold": self.stuck_time_threshold,
            "noise_strength": self.noise_strength,
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
                    f"[GridPotentialFieldStrategy] grid_resolution updated to "
                    f"{self.grid_resolution:.2f}, nx={self.nx}, ny={self.ny}"
                )
                self._global_path_valid = False

            if key == "obstacle_inflation_radius":
                clamped_inflation = min(
                    max(0.0, self.obstacle_inflation_radius),
                    1.01 * self.grid_resolution,
                )
                self.obstacle_inflation_radius = clamped_inflation
                print(
                    f"[GridPotentialFieldStrategy] obstacle_inflation_radius set to "
                    f"{self.obstacle_inflation_radius:.2f}m"
                )

            if key in ("progress_threshold", "stuck_time_threshold", "noise_strength"):
                print(
                    "[GridPotentialFieldStrategy] Updated local-minimum parameters: "
                    f"progress_threshold={self.progress_threshold:.3f}, "
                    f"stuck_time_threshold={self.stuck_time_threshold:.2f}s, "
                    f"noise_strength={self.noise_strength:.2f}"
                )

    # ------------------------------------------------------------------
    # Debug / visualization
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