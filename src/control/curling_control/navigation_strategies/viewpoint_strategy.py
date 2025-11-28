"""
Viewpoint Navigation Strategy.

Idea:
- We have a small set of hard-coded "viewpoints" on the field where
  marker detection / localization works well.
- The strategy moves the robot from its current pose through each
  viewpoint in sequence, and finally to the goal (target_x, target_y)
  provided by follow.py.
- For each waypoint we use an ORIENT -> DRIVE pattern (turn in place,
  then drive forward) similar to OrientThenDriveStrategy.
- At each intermediate viewpoint we can optionally "dwell" (stop for a
  short time) to let the localization stack update from markers.

Field / coordinate system:
- We use the same world frame as the marker_map:
    * (0, 0)  : bottom-right corner of the field
    * (6, 0)  : bottom-left  corner
    * (0, 9)  : top-right    corner
    * (6, 9)  : top-left     corner
- So the field is 6 m wide (x: 0 -> 6) and 9 m long (y: 0 -> 9).

From Assignment 2 (all in mm):
    A = Field length          = 9000
    B = Field width           = 6000
    G = Penalty area length   = 1650
    H = Penalty area width    = 4000

Converted to our frame (in meters):
    FIELD_LENGTH         = 9.0   (along +y)
    FIELD_WIDTH          = 6.0   (along +x)
    PENALTY_AREA_LENGTH  = 1.65  (distance from goal line to penalty-box line)
    PENALTY_AREA_WIDTH   = 4.0   (size of penalty box across the field)

This gives:
    - Distance from each sideline to the vertical edge of the penalty box:
          side_offset = (FIELD_WIDTH - PENALTY_AREA_WIDTH) / 2 = 1.0 m
      -> right-side penalty-vertical line  at x = 1.0
      -> left-side  penalty-vertical line  at x = 5.0

    - Distance from bottom / top goal line to the penalty-box line:
          penalty_y_near = PENALTY_AREA_LENGTH          = 1.65
          penalty_y_far  = FIELD_LENGTH - 1.65         ≈ 7.35

We use these to define viewpoint paths that go:
    1. From the first penalty-box corner on the chosen side,
    2. To a point just before the halfway line near that sideline,
    3. To a point just after the halfway line near that sideline,
    4. To the far-side penalty-box corner on the same side,
    5. Then finally to the requested (target_x, target_y).
"""

import math
from enum import Enum
from typing import List, Tuple, Optional

from geometry_msgs.msg import Twist, Vector3

from .base_strategy import BaseNavigationStrategy


# ---------------------------------------------------------------------------
# Field / geometry constants (in meters)
# ---------------------------------------------------------------------------

FIELD_LENGTH: float = 9.0         # goal-to-goal, along +y
FIELD_WIDTH: float = 6.0          # sideline-to-sideline, along +x

PENALTY_AREA_LENGTH: float = 1.65  # distance from goal line to box line (G)
PENALTY_AREA_WIDTH: float = 4.0    # width of penalty area across the field (H)

# Vertical (along x) offset from each sideline to the penalty-box edge
SIDE_OFFSET: float = (FIELD_WIDTH - PENALTY_AREA_WIDTH) / 2.0  # = 1.0

# Halfway line along the length of the field (y)
HALF_LINE_Y: float = FIELD_LENGTH / 2.0  # = 4.5

# How far before / after the halfway line we place intermediate viewpoints
HALF_LINE_OFFSET: float = 1.0  # 1 m before / after half-way


class NavigationPhase(Enum):
    """Current phase of motion w.r.t. the active waypoint."""
    ORIENT = "orient"
    DRIVE = "drive"
    COMPLETE = "complete"


class ViewpointStrategy(BaseNavigationStrategy):
    """
    Navigation strategy that chains ORIENT->DRIVE segments through a set
    of predefined "viewpoints" and then goes to the final goal.

    The final goal is always the (target_x, target_y) that follow.py
    passes in on each control step. The intermediate viewpoints are
    chosen based on the field geometry and can be placed on the left or
    right "lane" near the sidelines.

    Viewpoint sequence on a given side (right or left):
        1. Near own penalty-box edge (on that side)
        2. Just before the halfway line, near sideline
        3. Just after the halfway line, near sideline
        4. Near the far penalty-box edge (same side)
        5. Final target
    """

    def __init__(
        self,
        safe_distance: float = 1.0,
        repulse_strength: float = 1.0,
        goal_tolerance: float = 0.1,
        max_linear_velocity: float = 0.2,
        angular_gain: float = 1.0,
        orientation_tolerance: float = 0.15,   # radians
        max_angular_velocity: float = 1.0,     # rad/s
        control_dt: float = 0.05,              # must match follow.py timer
        dwell_time_at_viewpoint: float = 1.0,  # seconds to pause at each VP
        viewpoints: Optional[List[Tuple[float, float]]] = None,
        side: Optional[str] = None,            # 'left', 'right', or None (auto)
        **kwargs,
    ):
        """
        Args:
            safe_distance, repulse_strength: kept for compatibility with
                other strategies; not yet used in this basic version.
            goal_tolerance: distance threshold for "reached waypoint".
            max_linear_velocity: forward speed when driving (m/s).
            angular_gain: proportional gain for heading control.
            orientation_tolerance: allowed heading error before switching
                ORIENT -> DRIVE (rad).
            max_angular_velocity: clamp for angular speed (rad/s).
            control_dt: control-loop period (seconds).
            dwell_time_at_viewpoint: how long to stop at each intermediate
                viewpoint (seconds). Set to 0.0 to not dwell.
            viewpoints: explicit list of (x, y) in world frame where we
                want to localize / see markers well. If None, a default
                path based on penalty-box corners and the halfway line
                will be chosen.
            side: if viewpoints is None, this can force the lane:
                'left'  -> path near x ≈ 5 (left sideline)
                'right' -> path near x ≈ 1 (right sideline)
                None    -> side chosen automatically from initial pose.
        """
        self.safe_distance = safe_distance
        self.repulse_strength = repulse_strength
        self.goal_tolerance = goal_tolerance
        self.max_linear_velocity = max_linear_velocity
        self.angular_gain = angular_gain
        self.orientation_tolerance = orientation_tolerance
        self.max_angular_velocity = max_angular_velocity
        self.control_dt = control_dt

        # Which "lane" we ended up choosing ("left" / "right" / None)
        self.side: Optional[str] = None

        # If True, we'll construct viewpoints lazily on first call to
        # compute_control, based on the robot pose and/or preferred side.
        self._auto_viewpoints: bool = False

        # Handle side preference
        side_pref: Optional[str] = None
        if isinstance(side, str):
            s = side.lower()
            if s in ("left", "right"):
                side_pref = s

        # Hard-coded viewpoints or auto-generated path
        if viewpoints is None:
            # We'll choose left/right lane at runtime.
            self.viewpoints: List[Tuple[float, float]] = []
            self._auto_viewpoints = True
            self.side = side_pref  # may be None -> auto choose
        else:
            # Explicit viewpoints override the automatic lane generation.
            self.viewpoints = list(viewpoints)
            self.side = side_pref

        # Index in the list [viewpoints..., final_goal]
        self.current_waypoint_index: int = 0
        self.phase: NavigationPhase = NavigationPhase.ORIENT

        # Dwell duration (in control steps) at each intermediate viewpoint
        self.dwell_steps_required = (
            int(dwell_time_at_viewpoint / control_dt)
            if dwell_time_at_viewpoint > 0.0
            else 0
        )
        self._reset_dwell_state()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_dwell_state(self) -> None:
        self.in_dwell: bool = False
        self.dwell_steps_done: int = 0

    def _side_x(self, side: str) -> float:
        """
        x coordinate for the given side's "lane".

        'right' -> near the right sideline (x ≈ 0)
        'left'  -> near the left  sideline (x ≈ 6)
        """
        if side == "right":
            return SIDE_OFFSET                # 1.0 m from right sideline at x=0
        else:  # 'left'
            return FIELD_WIDTH - SIDE_OFFSET  # 1.0 m from left sideline at x=6

    def _build_side_viewpoints(self, side: str) -> List[Tuple[float, float]]:
        """
        Build the 4 default viewpoints along the chosen side:

            1. Near own penalty-box edge (on that side)
            2. Just before the halfway line, near sideline
            3. Just after the halfway line, near sideline
            4. Near the far penalty-box edge (same side)
        """
        x_lane = self._side_x(side)

        penalty_near_y = PENALTY_AREA_LENGTH
        penalty_far_y = FIELD_LENGTH - PENALTY_AREA_LENGTH

        pre_half_y = HALF_LINE_Y - HALF_LINE_OFFSET
        post_half_y = HALF_LINE_Y + HALF_LINE_OFFSET

        return [
            (x_lane, penalty_near_y),
            (x_lane, pre_half_y),
            (x_lane, post_half_y),
            (x_lane, penalty_far_y),
        ]

    def _init_viewpoints_if_needed(
        self,
        robot_x: float,
        robot_y: float,
        target_x: float,
        target_y: float,
    ) -> None:
        """
        If viewpoints were not explicitly provided, choose a lane (left /
        right) and build the default viewpoint list based on the initial
        robot pose.

        Logic:
            - If self.side was set ('left' or 'right'), use that.
            - Otherwise, choose the side whose lane x is closer to the
              current robot x position.
        """
        if not self._auto_viewpoints:
            return

        # Decide which side to use
        if self.side in ("left", "right"):
            chosen_side = self.side
        else:
            # Choose side based on which lane is closer in x
            right_x = self._side_x("right")
            left_x = self._side_x("left")

            if abs(robot_x - right_x) <= abs(robot_x - left_x):
                chosen_side = "right"
            else:
                chosen_side = "left"

        self.viewpoints = self._build_side_viewpoints(chosen_side)
        self.side = chosen_side
        self._auto_viewpoints = False

    def _full_path(
        self, target_x: float, target_y: float
    ) -> List[Tuple[float, float]]:
        """
        Returns the full list of waypoints for this run: all viewpoints
        plus the final goal.
        """
        return self.viewpoints + [(target_x, target_y)]

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Shortest signed angular difference a - b in [-pi, pi]."""
        d = a - b
        return math.atan2(math.sin(d), math.cos(d))

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    # ------------------------------------------------------------------
    # BaseNavigationStrategy interface
    # ------------------------------------------------------------------

    def compute_control(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        target_x: float,
        target_y: float,
        obstacles,
    ):
        """
        Move through viewpoints then to final target.

        Returns:
            (Twist, heading_vector, goal_reached)
            goal_reached is True only when the *final* goal is reached
            (not for intermediate viewpoints).
        """
        # Initialize default viewpoint path lazily, if needed
        if self._auto_viewpoints:
            self._init_viewpoints_if_needed(robot_x, robot_y, target_x, target_y)

        path = self._full_path(target_x, target_y)
        last_index = len(path) - 1

        # Clamp index in case something weird happens
        if self.current_waypoint_index > last_index:
            self.current_waypoint_index = last_index

        goal_x, goal_y = path[self.current_waypoint_index]
        is_final_goal = self.current_waypoint_index == last_index

        dx = goal_x - robot_x
        dy = goal_y - robot_y
        distance = math.hypot(dx, dy)

        # --------------------------------------------------------------
        # 1. Check if we have reached the current waypoint
        # --------------------------------------------------------------
        if distance < self.goal_tolerance:
            # Final goal reached: stop and signal completion
            if is_final_goal:
                self.phase = NavigationPhase.COMPLETE
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                return cmd, None, True

            # Intermediate viewpoint reached:
            # optionally dwell here to localize from markers
            if self.dwell_steps_required > 0 and not self.in_dwell:
                self.in_dwell = True
                self.dwell_steps_done = 0

            if self.in_dwell:
                self.dwell_steps_done += 1

                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0

                heading_vec = Vector3()
                heading_vec.x = math.cos(robot_yaw)
                heading_vec.y = math.sin(robot_yaw)
                heading_vec.z = 0.0

                # After we've dwelled long enough, advance to next VP
                if self.dwell_steps_done >= self.dwell_steps_required:
                    self.in_dwell = False
                    self.dwell_steps_done = 0
                    self.current_waypoint_index += 1
                    self.phase = NavigationPhase.ORIENT

                # While dwelling we never claim the final goal is reached
                return cmd, heading_vec, False

            # No dwell configured: jump to next viewpoint but emit a stop
            self.current_waypoint_index += 1
            self.phase = NavigationPhase.ORIENT

            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            return cmd, None, False

        # --------------------------------------------------------------
        # 2. Not yet at waypoint -> ORIENT / DRIVE logic
        # --------------------------------------------------------------

        target_heading = math.atan2(dy, dx)
        heading_error = self._angle_diff(target_heading, robot_yaw)

        heading_vec = Vector3()
        heading_vec.x = math.cos(target_heading)
        heading_vec.y = math.sin(target_heading)
        heading_vec.z = 0.0

        # Phase: ORIENT
        if self.phase == NavigationPhase.ORIENT:
            if abs(heading_error) < self.orientation_tolerance:
                self.phase = NavigationPhase.DRIVE
            else:
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = self._clamp(
                    self.angular_gain * heading_error,
                    -self.max_angular_velocity,
                    self.max_angular_velocity,
                )
                return cmd, heading_vec, False

        # Phase: DRIVE
        if self.phase == NavigationPhase.DRIVE:
            # If we drift too far in heading, go back to ORIENT
            if abs(heading_error) > self.orientation_tolerance * 2.0:
                self.phase = NavigationPhase.ORIENT
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = self._clamp(
                    self.angular_gain * heading_error,
                    -self.max_angular_velocity,
                    self.max_angular_velocity,
                )
                return cmd, heading_vec, False

            cmd = Twist()
            cmd.linear.x = self.max_linear_velocity
            cmd.angular.z = self._clamp(
                self.angular_gain * heading_error,
                -self.max_angular_velocity,
                self.max_angular_velocity,
            )
            return cmd, heading_vec, False

        # Should not really get here; default to stop
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        return cmd, None, False

    def reset(self) -> None:
        """Reset strategy state to initial conditions."""
        self.current_waypoint_index = 0
        self.phase = NavigationPhase.ORIENT
        self._reset_dwell_state()

    def is_goal_reached(
        self,
        robot_x: float,
        robot_y: float,
        target_x: float,
        target_y: float,
    ) -> bool:
        """
        Only checks the *final* goal, not the intermediate viewpoints.
        """
        dx = target_x - robot_x
        dy = target_y - robot_y
        return math.hypot(dx, dy) < self.goal_tolerance

    def get_parameters(self) -> dict:
        """Return current tunable parameters (for debugging / introspection)."""
        return {
            "safe_distance": self.safe_distance,
            "repulse_strength": self.repulse_strength,
            "goal_tolerance": self.goal_tolerance,
            "max_linear_velocity": self.max_linear_velocity,
            "angular_gain": self.angular_gain,
            "orientation_tolerance": self.orientation_tolerance,
            "max_angular_velocity": self.max_angular_velocity,
            "viewpoints": self.viewpoints,
            "current_waypoint_index": self.current_waypoint_index,
            "phase": self.phase.value,
            "side": self.side,
        }

    def set_parameters(self, params: dict) -> None:
        """
        Simple parameter updating helper. You can call this from outside
        if you want to adjust gains or viewpoints at runtime.
        """
        for key, value in params.items():
            if key == "viewpoints":
                self.viewpoints = list(value)
                # If viewpoints are explicitly set, we no longer auto-generate.
                self._auto_viewpoints = False
            elif key == "side":
                if isinstance(value, str) and value.lower() in ("left", "right"):
                    self.side = value.lower()
            elif hasattr(self, key):
                setattr(self, key, value)
