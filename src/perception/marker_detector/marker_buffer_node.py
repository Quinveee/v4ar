#!/usr/bin/env python3
import copy
import math
from typing import Dict

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from perception_msgs.msg import MarkerPoseArray, MarkerPose


class MarkerBufferNode(Node):
    """
    Temporally smooths AprilTag detections.

    - Holds on to markers for a short time window when they disappear
    - Optionally low-pass filters the distance per marker
    - Outputs a "stable" MarkerPoseArray for triangulation

    This is similar in spirit to your TrackingLineSelector for lines.
    """

    def __init__(self):
        super().__init__("marker_buffer_node")

        # Parameters
        self.declare_parameter("input_topic", "/detected_markers")
        self.declare_parameter("output_topic", "/detected_markers_buffered")
        self.declare_parameter("max_age_sec", 0.3)         # keep marker for this long after last seen
        self.declare_parameter("distance_alpha", 0.6)      # low-pass filter weight for distance

        input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.max_age_sec = self.get_parameter("max_age_sec").get_parameter_value().double_value
        self.distance_alpha = self.get_parameter("distance_alpha").get_parameter_value().double_value

        # Per-marker state: id -> {"marker": MarkerPose, "last_seen": Time}
        self.marker_state: Dict[int, Dict[str, object]] = {}

        self.sub = self.create_subscription(
            MarkerPoseArray,
            input_topic,
            self.marker_callback,
            10
        )

        self.pub = self.create_publisher(
            MarkerPoseArray,
            output_topic,
            10
        )

        self.get_logger().info(
            f"MarkerBufferNode started. Input: {input_topic}, Output: {output_topic}, "
            f"max_age_sec={self.max_age_sec:.2f}, distance_alpha={self.distance_alpha:.2f}"
        )

    # ------------------------------------------------------------

    def marker_callback(self, msg: MarkerPoseArray):
        """
        For each incoming frame:
        - Update per-marker state with new detections (with distance smoothing)
        - Keep older markers which have not expired yet
        - Publish a combined array
        """
        now = Time.from_msg(msg.header.stamp)

        # 1. Update markers we actually see in this frame
        seen_ids = set()
        for m in msg.markers:
            seen_ids.add(m.id)
            if m.id in self.marker_state:
                prev_marker: MarkerPose = self.marker_state[m.id]["marker"]
                # low-pass filter on distance
                filtered_distance = (
                    self.distance_alpha * m.distance
                    + (1.0 - self.distance_alpha) * prev_marker.distance
                )
                new_marker = copy.deepcopy(m)
                new_marker.distance = float(filtered_distance)
            else:
                # First time we see this marker
                new_marker = copy.deepcopy(m)

            self.marker_state[m.id] = {
                "marker": new_marker,
                "last_seen": now,
            }

        # 2. Build output array with all markers that are still "fresh" enough
        out = MarkerPoseArray()
        out.header = msg.header

        ids_to_delete = []
        for marker_id, state in self.marker_state.items():
            age = (now - state["last_seen"]).nanoseconds / 1e9

            if age <= self.max_age_sec:
                # Still fresh enough → keep
                out.markers.append(state["marker"])
            else:
                # Too old → forget this marker
                ids_to_delete.append(marker_id)

        for marker_id in ids_to_delete:
            del self.marker_state[marker_id]

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = MarkerBufferNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
