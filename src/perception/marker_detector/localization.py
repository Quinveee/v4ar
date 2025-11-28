#!/usr/bin/env python3
import math
import argparse
import numpy as np
from collections import deque
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped

# Import strategies
from .localization_strategies import *

STRATEGIES = {
    "kalman": KalmanLocalization,
    "complementary": ComplementaryLocalization,
    "robust": RobustLocalization,
    "identity": IdentityLocalization,
    "adaptive_kalman": AdaptiveKalmanLocalization,
    "particle_filter": ParticleFilterLocalization,
    "sliding_window": SlidingWindowLocalization,
}


def yaw_to_quaternion(yaw):
    half = yaw / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


def quaternion_to_yaw(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class LocalizationNode(Node):
    def __init__(self):
        super().__init__("localization_node")

        # --- Parameters ---
        self.declare_parameter("strategy_type", "identity")
        self.declare_parameter("publish_rate", 20.0)
        self.declare_parameter("buffer_size", 5)
        self.declare_parameter("stationary_threshold", 5)
        self.declare_parameter("enable_exponential_smoothing", True)
        self.declare_parameter("smoothing_alpha", 0.3)

        strategy_type = self.get_parameter("strategy_type").value
        Strategy = STRATEGIES.get(strategy_type, IdentityLocalization)
        self.localizer = Strategy()
        self.get_logger().info(
            f"Using localization strategy: {Strategy.__name__}")

        # --- Internal state ---
        self.v = 0.0
        self.w = 0.0
        self.last_update_time = None

        # Stationary detection & buffer
        self.pose_buffer = deque(
            maxlen=self.get_parameter("buffer_size").value)
        self.stationary_counter = 0
        self.stationary_threshold = self.get_parameter(
            "stationary_threshold").value
        self.v_thresh = 0.02
        self.w_thresh = 0.02

        # Exponential smoothing
        self.enable_smoothing = self.get_parameter(
            "enable_exponential_smoothing").value
        self.alpha = self.get_parameter("smoothing_alpha").value
        self.prev_smoothed_pose = None  # (x, y, yaw)

        # --- ROS setup ---
        self.pose_pub = self.create_publisher(PoseStamped, "/robot_pose", 10)
        self.create_subscription(Twist, "/cmd_vel", self.cmd_callback, 10)
        self.create_subscription(
            PoseStamped, "/robot_pose_raw", self.pose_callback, 10)

        rate = self.get_parameter("publish_rate").value
        self.create_timer(1.0 / rate, self.timer_callback)

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------
    def is_stationary(self):
        return abs(self.v) < self.v_thresh and abs(self.w) < self.w_thresh

    def _apply_exponential_smoothing(self, x, y, yaw):
        """Apply temporal exponential smoothing (optional)."""
        if self.prev_smoothed_pose is None:
            self.prev_smoothed_pose = (x, y, yaw)
            return x, y, yaw

        prev_x, prev_y, prev_yaw = self.prev_smoothed_pose
        alpha = self.alpha

        # Smooth position
        x_s = alpha * x + (1 - alpha) * prev_x
        y_s = alpha * y + (1 - alpha) * prev_y

        # Smooth yaw using circular interpolation
        delta_yaw = math.atan2(math.sin(yaw - prev_yaw),
                               math.cos(yaw - prev_yaw))
        yaw_s = prev_yaw + alpha * delta_yaw
        yaw_s = math.atan2(math.sin(yaw_s), math.cos(yaw_s))

        self.prev_smoothed_pose = (x_s, y_s, yaw_s)
        return x_s, y_s, yaw_s

    # ----------------------------------------------------------
    # ROS Callbacks
    # ----------------------------------------------------------
    def cmd_callback(self, msg):
        self.v = msg.linear.x
        self.w = msg.angular.z

    def pose_callback(self, msg: PoseStamped):
        stationary = self.is_stationary()

        if stationary:
            self.stationary_counter = min(
                self.stationary_counter + 1, self.stationary_threshold)
        else:
            self.stationary_counter = max(self.stationary_counter - 1, 0)
            self.pose_buffer.clear()

        yaw = quaternion_to_yaw(msg.pose.orientation)

        # Stationary → buffer-based smoothing
        if self.stationary_counter >= self.stationary_threshold:
            self.pose_buffer.append(
                (msg.pose.position.x, msg.pose.position.y, yaw))

            if len(self.pose_buffer) > 1:
                xs, ys, thetas = zip(*self.pose_buffer)
                x, y = np.mean(xs), np.mean(ys)
                yaw = math.atan2(np.mean(np.sin(thetas)),
                                 np.mean(np.cos(thetas)))
            else:
                x, y = msg.pose.position.x, msg.pose.position.y

            self.get_logger().debug("Stationary mode: averaging buffer")

        else:
            x, y = msg.pose.position.x, msg.pose.position.y
            self.get_logger().debug("Moving mode: raw pose")

        # Optional exponential smoothing
        if self.enable_smoothing:
            x, y, yaw = self._apply_exponential_smoothing(x, y, yaw)

        # Update message with smoothed values
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)

        # Pass to filter
        self.localizer.update(msg)

    def timer_callback(self):
        now = self.get_clock().now()
        if self.last_update_time is None:
            self.last_update_time = now
            return
        dt = (now - self.last_update_time).nanoseconds / 1e9
        self.last_update_time = now

        self.localizer.predict(self.v, self.w, dt)
        x, y, theta = self.localizer.get_pose()

        # Publish filtered output
        qx, qy, qz, qw = yaw_to_quaternion(theta)
        out = PoseStamped()
        out.header.stamp = now.to_msg()
        out.header.frame_id = "world"
        out.pose.position.x = x
        out.pose.position.y = y
        out.pose.orientation.x = qx
        out.pose.orientation.y = qy
        out.pose.orientation.z = qz
        out.pose.orientation.w = qw
        self.pose_pub.publish(out)

# ----------------------------------------------------------
# Node entrypoint
# ----------------------------------------------------------


def main(args=None):
    rclpy.init()
    node = LocalizationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
