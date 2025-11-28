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

Notes:
- Coordinates are in the same world frame as /robot_pose.
- Obstacles are currently ignored, but are available for future logic
  (e.g. choosing between alternative viewpoint paths).
"""

import math
from enum import Enum
from typing import List, Tuple, Optional

from geometry_msgs.msg import Twist, Vector3

from .base_strategy import BaseNavigationStrategy


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
    hard-coded here for now, but can later be made parametric or chosen
    based on perception (e.g. which markers / rovers are visible).
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
            viewpoints: list of (x, y) in world frame where we want to
                localize / see markers well. If None, a simple default
                path is used.
        """
        self.safe_distance = safe_distance
        self.repulse_strength = repulse_strength
        self.goal_tolerance = goal_tolerance
        self.max_linear_velocity = max_linear_velocity
        self.angular_gain = angular_gain
        self.orientation_tolerance = orientation_tolerance
        self.max_angular_velocity = max_angular_velocity
        self.control_dt = control_dt

        # Hard-coded viewpoints for now.
        # TODO: tune these for your actual AprilTag-rich regions.
        if viewpoints is None:
            # Example: three viewpoints going "forward" along x.
            # Replace with positions that make sense in your field.
            self.viewpoints: List[Tuple[float, float]] = [
                (1.5, 0.0),
                (3.0, 0.5),
                (4.5, 0.0),
            ]
        else:
            self.viewpoints = list(viewpoints)

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
            goal_reached is True only when the *final* target is reached
            (not for intermediate viewpoints).
        """
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
        }

    def set_parameters(self, params: dict) -> None:
        """
        Simple parameter updating helper. You can call this from outside
        if you want to adjust gains or viewpoints at runtime.
        """
        for key, value in params.items():
            if key == "viewpoints":
                self.viewpoints = list(value)
            elif hasattr(self, key):
                setattr(self, key, value)
