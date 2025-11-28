import math
from typing import List, Tuple, Optional

from geometry_msgs.msg import Twist, Vector3
from .base_strategy import BaseNavigationStrategy


class IntegratedYawDirectGoalStrategy(BaseNavigationStrategy):
    """
    Direct-goal strategy that ignores external yaw and instead keeps its own
    internal yaw estimate integrated from the commanded angular velocity.

    - Position (x, y) is taken from the caller (e.g., localization).
    - Orientation (yaw) is internal:
        yaw_est(t+1) = yaw_est(t) + last_cmd.angular.z * control_dt

    This lets you:
    - Hard-code the initial yaw to match how you place the robot.
    - Still benefit from decent x,y from your existing localization.
    """

    def __init__(
        self,
        goal_tolerance: float = 0.1,
        max_linear_velocity: float = 0.2,
        angular_gain: float = 1.0,
        min_angular_error_for_forward: float = 0.3,  # ~17 deg
        control_dt: float = 0.05,  # MUST match follow.py timer period
        initial_yaw: float = 0.0,  # hard-coded starting orientation (rad)
        *args,
        **kwargs
    ):
        """
        Args:
            goal_tolerance: Distance threshold to consider goal reached (m)
            max_linear_velocity: Forward velocity when moving (m/s)
            angular_gain: P-gain for angular velocity control
            min_angular_error_for_forward: Heading error threshold for forward motion
            control_dt: Assumed time step between control calls (seconds)
            initial_yaw: Internal yaw at t=0, in radians.
                         Place the robot on the field at this angle.
        """
        # Store control params (same as DirectGoalStrategy)
        self.goal_tolerance = goal_tolerance
        self.max_linear_velocity = max_linear_velocity
        self.angular_gain = angular_gain
        self.min_angular_error_for_forward = min_angular_error_for_forward

        # Dead-reckoning yaw config
        self.control_dt = control_dt
        self.initial_yaw = initial_yaw

        # Internal state
        self.initialized = False
        self.yaw_est = initial_yaw

        # Last command actually sent (used for integration)
        self.last_cmd_linear_x = 0.0
        self.last_cmd_angular_z = 0.0

    # ------------------------------------------------------------------
    # Core strategy API
    # ------------------------------------------------------------------

    def compute_control(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,  # will be ignored
        target_x: float,
        target_y: float,
        obstacles: List[Tuple[float, float]]
    ) -> Tuple[Twist, Optional[Vector3], bool]:
        """
        Compute control to drive directly toward goal, using *internal* yaw.

        Incoming robot_yaw is ignored; instead we integrate our own yaw_est
        from the previous command.
        """

        # 1) Initialize yaw on first call
        if not self.initialized:
            self.yaw_est = self.initial_yaw
            self.initialized = True
        else:
            # 2) Integrate yaw from the previous command
            self.yaw_est += self.last_cmd_angular_z * self.control_dt
            self.yaw_est = self._wrap_angle(self.yaw_est)

        # 3) Check if goal is reached (using external x,y)
        if self.is_goal_reached(robot_x, robot_y, target_x, target_y):
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            # Reset last command so we don't keep integrating
            self.last_cmd_linear_x = 0.0
            self.last_cmd_angular_z = 0.0

            return (cmd, None, True)

        # 4) Compute vector to goal
        dx = target_x - robot_x
        dy = target_y - robot_y

        # Desired heading (angle to goal)
        target_heading = math.atan2(dy, dx)

        # 5) Heading error using our internal yaw estimate
        heading_error = self._angle_diff(target_heading, self.yaw_est)

        # 6) Angular velocity (P controller)
        angular_velocity = self.angular_gain * heading_error

        # 7) Linear velocity: only move forward if roughly facing the goal
        if abs(heading_error) < self.min_angular_error_for_forward:
            linear_velocity = self.max_linear_velocity
        else:
            linear_velocity = 0.0  # turn in place

        # 8) Build Twist command
        cmd = Twist()
        cmd.linear.x = linear_velocity
        cmd.angular.z = angular_velocity

        # Store as "last" command for next integration step
        self.last_cmd_linear_x = linear_velocity
        self.last_cmd_angular_z = angular_velocity

        # 9) Visualization vector toward goal
        heading_vec = Vector3()
        heading_vec.x = math.cos(target_heading)
        heading_vec.y = math.sin(target_heading)
        heading_vec.z = 0.0

        return (cmd, heading_vec, False)

    def reset(self) -> None:
        """Reset strategy state."""
        self.initialized = False
        self.yaw_est = self.initial_yaw
        self.last_cmd_linear_x = 0.0
        self.last_cmd_angular_z = 0.0

    def is_goal_reached(
        self,
        robot_x: float,
        robot_y: float,
        target_x: float,
        target_y: float
    ) -> bool:
        dx = target_x - robot_x
        dy = target_y - robot_y
        distance = math.sqrt(dx * dx + dy * dy)
        return distance < self.goal_tolerance

    def get_parameters(self) -> dict:
        """Get current strategy parameters (for debugging/introspection)."""
        return {
            "goal_tolerance": self.goal_tolerance,
            "max_linear_velocity": self.max_linear_velocity,
            "angular_gain": self.angular_gain,
            "min_angular_error_for_forward": self.min_angular_error_for_forward,
            "control_dt": self.control_dt,
            "initial_yaw": self.initial_yaw,
        }

    def set_parameters(self, params: dict) -> None:
        """Update strategy parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Shortest angular difference between two angles."""
        d = a - b
        return math.atan2(math.sin(d), math.cos(d))

    @staticmethod
    def _wrap_angle(a: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return math.atan2(math.sin(a), math.cos(a))
