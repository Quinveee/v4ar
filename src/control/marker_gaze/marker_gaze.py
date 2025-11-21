#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from perception_msgs.msg import MarkerPoseArray
from .gaze_strategies.weighed_gaze_strategy import WeightedGazeStrategy
from .gaze_strategies.vibe_gaze_strategy import ActiveSearchGazeStrategy

STRATEGIES = {
    "weighted": WeightedGazeStrategy,
    "vibe": ActiveSearchGazeStrategy,
}


class MarkerGazeNode(Node):
    def __init__(self):
        super().__init__("marker_gaze")

        # --- Parameters ---
        self.declare_parameter("strategy_type", "weighted")
        self.declare_parameter("kp_pan", 0.5)
        self.declare_parameter("kp_tilt", 0.2)

        self.strategy_type = self.get_parameter("strategy_type").value
        self.kp_pan = self.get_parameter("kp_pan").value
        self.kp_tilt = self.get_parameter("kp_tilt").value

        if self.strategy_type not in STRATEGIES:
            self.get_logger().warn(
                f"Unknown strategy '{self.strategy_type}', defaulting to 'weighted'")
            self.strategy_type = "weighted"

        # --- Lazy initialization ---
        self.strategy = None
        self.img_width = None
        self.img_height = None

        # --- Subscribers and Publishers ---
        self.image_sub = self.create_subscription(
            Image, "/processed_image", self.image_callback, 1)
        self.marker_sub = self.create_subscription(
            MarkerPoseArray, "/detected_markers", self.marker_callback, 10)

        # Publish pan and tilt as joint commands
        self.joint_pub = self.create_publisher(
            JointState, "/ugv/joint_states", 10)

        self.get_logger().info(
            "MarkerGazeNode initialized. Waiting for first image to get dimensions...")

    # ---------------------------------------------------------------

    def image_callback(self, msg: Image):
        """Initialize image dimensions once and then unsubscribe."""
        if self.img_width is None or self.img_height is None:
            self.img_width = msg.width
            self.img_height = msg.height
            self.get_logger().info(
                f"Got image dimensions: {self.img_width}x{self.img_height}")

            # Initialize strategy dynamically
            strategy_cls = STRATEGIES[self.strategy_type]
            self.strategy = strategy_cls(
                img_width=self.img_width,
                img_height=self.img_height,
                kp_pan=self.kp_pan,
                kp_tilt=self.kp_tilt,
            )

            # Unsubscribe after first message to save bandwidth
            self.destroy_subscription(self.image_sub)
            self.get_logger().info("Unsubscribed from /processed_image (dimensions locked).")

    # ---------------------------------------------------------------

    def marker_callback(self, msg: MarkerPoseArray):
        """Compute pan/tilt corrections from marker detections."""

        if self.strategy is None:
            self.get_logger().warn("No image dimensions yet; waiting for /processed_image...")
            return

        if len(msg.markers) < 2:
            pan_angle, tilt_angle = self.strategy.compute_angles(msg)

            # Create and publish JointState command
            joint_msg = JointState()
            joint_msg.name = [
                "pt_base_link_to_pt_link1", "pt_link1_to_pt_link2"]
            joint_msg.position = [float(pan_angle), float(tilt_angle)]

            self.joint_pub.publish(joint_msg)
            self.get_logger().info(
                f"Published joint command: pan={pan_angle:.4f}, tilt={tilt_angle:.4f}")


def main(args=None):
    rclpy.init(args=args)
    node = MarkerGazeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
