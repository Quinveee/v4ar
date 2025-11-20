#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
from apriltag import apriltag
from perception_msgs.msg import MarkerPoseArray, MarkerPose


def rotation_matrix_to_quaternion(R: np.ndarray):
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w)."""
    q = np.zeros(4, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[3] = 0.25 / s
        q[0] = (R[2, 1] - R[1, 2]) * s
        q[1] = (R[0, 2] - R[2, 0]) * s
        q[2] = (R[1, 0] - R[0, 1]) * s
    else:
        idx = np.argmax(np.diag(R))
        if idx == 0:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[3] = (R[2, 1] - R[1, 2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[3] = (R[0, 2] - R[2, 0]) / s
            q[0] = (R[0, 1] + R[1, 0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[3] = (R[1, 0] - R[0, 1]) / s
            q[0] = (R[0, 2] + R[2, 0]) / s
            q[1] = (R[1, 2] + R[2, 1]) / s
            q[2] = 0.25 * s
    return tuple(float(x) for x in q)


class AprilTagPublisher(Node):
    """AprilTag detector that publishes MarkerPoseArray only."""

    def __init__(self):
        super().__init__("apriltag_publisher")
        self.bridge = CvBridge()
        self.tag_size = 0.162  # in meters

        # Calibration parameters
        self.K = np.array([
            [286.9896545092208, 0.0, 311.7114840273407],
            [0.0, 290.8395992360502, 249.9287049631703],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        self.P = np.array([
            [190.74984649646845, 0.0, 318.0141593176815, 0.0],
            [0.0, 247.35103262891005, 248.37293105876694, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype=np.float64)
        self.new_K = self.P[:, :3]

        # Detector
        self.detector = apriltag("tagStandard41h12")

        # ROS I/O
        self.sub = self.create_subscription(Image, "/processed_image", self.image_callback, 10)
        self.pub = self.create_publisher(MarkerPoseArray, "/detected_markers", 10)

        self.get_logger().info("AprilTagPublisher initialized and listening to /processed_image")

    def image_callback(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detections = self.detector.detect(gray)
        marker_array = MarkerPoseArray()
        marker_array.header = msg.header

        for det in detections:
            try:
                # Extract corners (lt, rt, rb, lb)
                if "lb" in det:
                    lb, rb, rt, lt = det["lb"], det["rb"], det["rt"], det["lt"]
                elif "lb-rb-rt-lt" in det:
                    lb, rb, rt, lt = det["lb-rb-rt-lt"]
                else:
                    continue

                img_pts = np.array([lt, rt, rb, lb], dtype=np.float32)
                obj_pts = np.array([
                    [-self.tag_size/2,  self.tag_size/2, 0],
                    [ self.tag_size/2,  self.tag_size/2, 0],
                    [ self.tag_size/2, -self.tag_size/2, 0],
                    [-self.tag_size/2, -self.tag_size/2, 0]
                ], dtype=np.float32)

                success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.new_K, np.zeros((4, 1)))
                if not success:
                    continue

                R, _ = cv2.Rodrigues(rvec)
                qx, qy, qz, qw = rotation_matrix_to_quaternion(R)
                t = tvec.flatten()
                distance = np.linalg.norm(t)

                marker = MarkerPose()
                marker.id = int(det["id"])
                marker.pose.position.x = float(t[0])
                marker.pose.position.y = float(t[1])
                marker.pose.position.z = float(t[2])
                marker.pose.orientation.x = qx
                marker.pose.orientation.y = qy
                marker.pose.orientation.z = qz
                marker.pose.orientation.w = qw
                marker.distance = float(distance)
                marker.center_x = float(det["center"][0])
                marker.center_y = float(det["center"][1])
                marker.corners = img_pts.flatten().tolist()

                marker_array.markers.append(marker)

            except Exception as e:
                self.get_logger().warn(f"Detection processing failed: {e}")

        self.pub.publish(marker_array)
        self.get_logger().info(f"Published {len(marker_array.markers)} markers")


def main(args=None):
    rclpy.init(args=args)
    node = AprilTagPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
