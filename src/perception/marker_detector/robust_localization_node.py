#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry  # optional, if you later want /odom
# If you don't use Odometry yet, you can remove this import.

# This node assumes /robot_pose is published by triangulation_node
# with frame_id = "world" and x,y in field coordinates.


def yaw_to_quaternion(yaw: float):
    """Convert yaw angle (rad) to a quaternion (x,y,z,w)."""
    half = yaw * 0.5
    qz = math.sin(half)
    qw = math.cos(half)
    return 0.0, 0.0, qz, qw


class RobustLocalizationNode(Node):
    """
    Fuses triangulated pose (/robot_pose) with wheel commands (/cmd_vel).

    - Maintains a continuous estimate of [x, y, theta] in the field frame.
    - Uses /cmd_vel to propagate pose (dead reckoning) between measurements.
    - When a new triangulated pose arrives:
        * Predict where we should be at that time.
        * If measurement is close → fuse (low-pass).
        * If it's a large jump → ignore as outlier.
    - Publishes a continuous, smoothed pose on /robot_pose_filtered.
    """

    def __init__(self):
        super().__init__("robust_localization_node")

        # Parameters
        self.declare_parameter("triangulated_topic", "/robot_pose")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("output_topic", "/robot_pose_filtered")

        # Max allowed distance between prediction and triangulated measurement (m)
        self.declare_parameter("max_position_jump", 0.75)

        # How strongly to trust the measurement vs prediction (0..1)
        self.declare_parameter("measurement_alpha", 0.7)

        # How long we are willing to purely dead-reckon without any markers (s)
        self.declare_parameter("max_prediction_without_markers", 5.0)

        # Publish rate for filtered pose (Hz)
        self.declare_parameter("publish_rate", 30.0)

        triangulated_topic = self.get_parameter("triangulated_topic").get_parameter_value().string_value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value

        self.max_position_jump = self.get_parameter("max_position_jump").get_parameter_value().double_value
        self.measurement_alpha = self.get_parameter("measurement_alpha").get_parameter_value().double_value
        self.max_prediction_without_markers = (
            self.get_parameter("max_prediction_without_markers").get_parameter_value().double_value
        )
        publish_rate = self.get_parameter("publish_rate").get_parameter_value().double_value

        # Internal state
        self.has_pose = False
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0  # heading in world frame (rad)

        self.last_update_time: Time = None      # time of last state update (propagation)
        self.last_marker_time: Time = None      # time of last accepted triangulation
        self.frames_since_marker = 0

        # Current wheel command
        self.v = 0.0   # linear.x
        self.w = 0.0   # angular.z
        self.last_cmd_time: Time = None

        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped,
            triangulated_topic,
            self.triangulated_pose_callback,
            10,
        )

        self.cmd_sub = self.create_subscription(
            Twist,
            cmd_vel_topic,
            self.cmd_vel_callback,
            10,
        )

        # Publisher
        self.pose_pub = self.create_publisher(
            PoseStamped,
            output_topic,
            10,
        )

        # Timer for continuous publishing
        period = 1.0 / max(publish_rate, 1.0)
        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info(
            f"RobustLocalizationNode started. Triangulated: {triangulated_topic}, "
            f"cmd_vel: {cmd_vel_topic}, output: {output_topic}, "
            f"max_jump={self.max_position_jump:.2f} m, alpha={self.measurement_alpha:.2f}"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def cmd_vel_callback(self, msg: Twist):
        """Store latest commanded linear and angular velocity."""
        self.v = float(msg.linear.x)
        self.w = float(msg.angular.z)
        self.last_cmd_time = self.get_clock().now()

    def triangulated_pose_callback(self, msg: PoseStamped):
        """
        Handle new triangulated pose from markers.
        We:
        1. Propagate prediction to the stamp of this message.
        2. Compare measured vs predicted position; reject if jump too big.
        3. Fuse by low-pass filtering.
        """
        meas_time = Time.from_msg(msg.header.stamp)

        if not self.has_pose:
            # First-ever measurement: initialize state
            self.x = msg.pose.position.x
            self.y = msg.pose.position.y
            self.theta = 0.0  # we don't get theta from triangulation yet
            self.last_update_time = meas_time
            self.last_marker_time = meas_time
            self.has_pose = True
            self.frames_since_marker = 0

            self.get_logger().info(
                f"Initialized pose from triangulation: x={self.x:.2f}, y={self.y:.2f}"
            )
            return

        # 1. Propagate to the time of this measurement using current wheel commands
        self._propagate_to(meas_time)

        # 2. Check difference between prediction and measurement
        meas_x = msg.pose.position.x
        meas_y = msg.pose.position.y

        dx = meas_x - self.x
        dy = meas_y - self.y
        dist = math.hypot(dx, dy)

        if dist > self.max_position_jump:
            # Outlier: ignore this triangulation
            self.frames_since_marker += 1
            self.get_logger().warn(
                f"Ignoring triangulation outlier (jump={dist:.2f} m > {self.max_position_jump:.2f} m). "
                f"Pred: ({self.x:.2f}, {self.y:.2f}) Meas: ({meas_x:.2f}, {meas_y:.2f})"
            )
            return

        # 3. Fuse prediction and measurement (simple complementary filter)
        alpha = self.measurement_alpha
        self.x = (1.0 - alpha) * self.x + alpha * meas_x
        self.y = (1.0 - alpha) * self.y + alpha * meas_y
        # theta stays from motion model (we don't have heading from triangulation yet)

        self.last_update_time = meas_time
        self.last_marker_time = meas_time
        self.frames_since_marker = 0

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------

    def timer_callback(self):
        """Periodically propagate and publish the filtered pose."""
        if not self.has_pose:
            return

        now = self.get_clock().now()
        self._propagate_to(now)

        # Warn if we've been dead-reckoning for too long without markers
        if self.last_marker_time is not None:
            dt_since_marker = (now - self.last_marker_time).nanoseconds / 1e9
            if dt_since_marker > self.max_prediction_without_markers:
                self.get_logger().warn_throttle(
                    period_sec=2.0,
                    msg=(
                        f"No triangulation updates for {dt_since_marker:.1f} s; "
                        "pose is now purely dead-reckoned and may drift."
                    )
                )

        # Publish filtered pose
        out = PoseStamped()
        out.header.stamp = now.to_msg()
        out.header.frame_id = "world"

        out.pose.position.x = float(self.x)
        out.pose.position.y = float(self.y)
        out.pose.position.z = 0.0

        qx, qy, qz, qw = yaw_to_quaternion(self.theta)
        out.pose.orientation.x = qx
        out.pose.orientation.y = qy
        out.pose.orientation.z = qz
        out.pose.orientation.w = qw

        self.pose_pub.publish(out)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _propagate_to(self, target_time: Time):
        """Propagate the internal state from last_update_time to target_time."""
        if self.last_update_time is None:
            self.last_update_time = target_time
            return

        dt = (target_time - self.last_update_time).nanoseconds / 1e9
        if dt <= 0.0:
            return

        # Use simple differential-drive kinematics in world frame
        v = self.v
        w = self.w

        # Small-angle handling: straight line vs arc
        if abs(w) < 1e-6:
            # Almost straight
            self.x += v * math.cos(self.theta) * dt
            self.y += v * math.sin(self.theta) * dt
        else:
            # Integrate on a circular arc
            # This is optional; simple Euler integration would also work.
            radius = v / w
            dtheta = w * dt
            self.x += radius * (math.sin(self.theta + dtheta) - math.sin(self.theta))
            self.y -= radius * (math.cos(self.theta + dtheta) - math.cos(self.theta))
            self.theta += dtheta

        # Normalize theta to [-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        self.last_update_time = target_time


def main(args=None):
    rclpy.init(args=args)
    node = RobustLocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
