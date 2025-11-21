#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Vector3
from perception_msgs.msg import ObjectPoseArray
import math
import numpy as np
import argparse


class NavigationWithAvoidanceNode(Node):
    def __init__(self, x=1.0, y=1.0, safe_distance=1.2, repulse_strength=1.5):
        super().__init__('navigation_with_avoidance_node')

        # Target
        self.declare_parameter("target_x", x)
        self.declare_parameter("target_y", y)
        self.declare_parameter("safe_distance", safe_distance)
        self.declare_parameter("repulse_strength", repulse_strength)
        self.target_x = self.get_parameter("target_x").value
        self.target_y = self.get_parameter("target_y").value
        self.safe_distance = self.get_parameter("safe_distance").value
        self.repulse_strength = self.get_parameter("repulse_strength").value

        # Robot pose
        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None

        # Subscriptions
        self.sub_pose = self.create_subscription(
            PoseStamped, "/robot_pose", self.pose_callback, 10
        )
        self.sub_rovers = self.create_subscription(
            ObjectPoseArray, "/detected_rovers", self.rover_callback, 10
        )

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.heading_msg = self.create_publisher(
            Vector3, "/control/heading_vector", 10)

        # Store obstacles
        self.obstacles = []  # (x, y) positions in world frame

        # Timer
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info("NavigationWithAvoidanceNode running.")

    # -------------------------------------------------------------------

    def pose_callback(self, msg: PoseStamped):
        self.robot_x = msg.pose.position.x
        self.robot_y = msg.pose.position.y

        q = msg.pose.orientation
        self.robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    # -------------------------------------------------------------------

    def rover_callback(self, msg: ObjectPoseArray):
        # Convert every detected rover to world-frame x,y
        self.obstacles = []

        for obj in msg.rovers:
            # Rover is relative to the camera — assume camera aligned with robot
            dx = obj.pose.position.z     # forward
            dy = -obj.pose.position.x    # left/right

            # Now convert to world frame
            world_x = self.robot_x + dx * \
                math.cos(self.robot_yaw) - dy * math.sin(self.robot_yaw)
            world_y = self.robot_y + dx * \
                math.sin(self.robot_yaw) + dy * math.cos(self.robot_yaw)

            self.obstacles.append((world_x, world_y))

    # -------------------------------------------------------------------

    def control_loop(self):
        if self.robot_x is None:
            return

        # ---------------------------------------------------
        # 1. Attractive force toward target
        # ---------------------------------------------------
        dx = self.target_x - self.robot_x
        dy = self.target_y - self.robot_y
        dist_to_goal = math.sqrt(dx*dx + dy*dy)

        goal_vec = np.array([dx, dy])
        goal_vec_norm = goal_vec / (np.linalg.norm(goal_vec) + 1e-6)

        # ---------------------------------------------------
        # 2. Repulsion from rovers (dynamic obstacles)
        # ---------------------------------------------------
        repulse_sum = np.array([0.0, 0.0])
        repulse_strength = self.repulse_strength
        safe_distance = self.safe_distance

        for ox, oy in self.obstacles:
            vx = self.robot_x - ox
            vy = self.robot_y - oy
            distance = math.sqrt(vx*vx + vy*vy)

            if distance < safe_distance:
                direction_norm = np.array([vx, vy]) / (distance + 1e-6)
                strength = repulse_strength * \
                    (1.0 / (distance*distance + 1e-6))
                repulse_sum += direction_norm * strength

        # ---------------------------------------------------
        # 3. Combine the vectors
        # ---------------------------------------------------
        combined = goal_vec_norm + repulse_sum
        combined_norm = combined / (np.linalg.norm(combined) + 1e-6)

        # Desired heading
        target_heading = math.atan2(combined_norm[1], combined_norm[0])

        # Compute angular error
        heading_error = self.angle_diff(target_heading, self.robot_yaw)

        # ---------------------------------------------------
        # 4. Convert to velocity commands
        # ---------------------------------------------------
        cmd = Twist()

        # Stop at goal
        if dist_to_goal < 0.1:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            self.get_logger().info("Target reached.")
            return

        cmd.angular.z = 1.5 * heading_error

        # Slow down if obstacle repulsion is large
        repulse_mag = np.linalg.norm(repulse_sum)
        speed_scale = max(0.1, 1.0 - repulse_mag)

        cmd.linear.x = 0.4 * speed_scale

        self.cmd_pub.publish(cmd)

        heading_msg = Vector3()
        heading_msg.x = combined_norm[0]
        heading_msg.y = combined_norm[1]
        heading_msg.z = 0.0
        self.heading_msg.publish(heading_msg)

    # -------------------------------------------------------------------

    @staticmethod
    def angle_diff(a, b):
        d = a - b
        return math.atan2(math.sin(d), math.cos(d))


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Navigation with Obstacle Avoidance Node")
    parser.add_argument("--target_x", type=float,
                        default=1.0, help="Target X coordinate in meters")
    parser.add_argument("--target_y", type=float,
                        default=1.0, help="Target Y coordinate in meters")
    parser.add_argument("--safe_distance", type=float,
                        default=1.2, help="Safe distance from obstacles in meters")
    parser.add_argument("--repulse_strength", type=float,
                        default=1.5, help="Repulsion strength from obstacles")
    parsed_args, unknown = parser.parse_known_args()
    rclpy.init(args=unknown)
    node = NavigationWithAvoidanceNode(
        x=parsed_args.target_x, y=parsed_args.target_y,
        safe_distance=parsed_args.safe_distance,
        repulse_strength=parsed_args.repulse_strength)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
