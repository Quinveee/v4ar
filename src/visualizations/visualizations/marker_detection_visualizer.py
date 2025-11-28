#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from perception_msgs.msg import MarkerPoseArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import math


class MarkerDetectionVisualizer(Node):
    """
    Visualizes marker detections overlaid on camera feeds.
    
    Subscribes to:
        - /processed_image (mono camera feed)
        - /oak/rgb/image_rect (OAK camera feed) 
        - /detected_markers (mono camera markers)
        - /oak/detected_markers (OAK camera markers)
    """

    def __init__(self):
        super().__init__('marker_detection_visualizer')
        
        self.bridge = CvBridge()
        
        # Image storage
        self.mono_image = None
        self.oak_image = None
        self.mono_markers = []
        self.oak_markers = []
        
        # Subscriptions - Images
        self.mono_image_sub = self.create_subscription(
            Image, '/processed_image', self.mono_image_callback, 10)
        
        self.oak_image_sub = self.create_subscription(
            Image, '/oak/rgb/image_rect', self.oak_image_callback, 10)
        
        # Subscriptions - Markers
        self.mono_markers_sub = self.create_subscription(
            MarkerPoseArray, '/detected_markers', self.mono_markers_callback, 10)
        
        self.oak_markers_sub = self.create_subscription(
            MarkerPoseArray, '/oak/detected_markers', self.oak_markers_callback, 10)
        
        # Render timer
        self.timer = self.create_timer(0.033, self.render)  # ~30 FPS
        
        self.get_logger().info("Marker Detection Visualizer started")

    def mono_image_callback(self, msg):
        try:
            self.mono_image = self.bridge.imgmsg_to_cv2(msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to convert mono image: {e}")

    def oak_image_callback(self, msg):
        try:
            self.oak_image = self.bridge.imgmsg_to_cv2(msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to convert OAK image: {e}")

    def mono_markers_callback(self, msg):
        self.mono_markers = msg.markers

    def oak_markers_callback(self, msg):
        self.oak_markers = msg.markers

    def draw_markers_on_image(self, image, markers, window_title):
        """Draw detected markers on image with annotations."""
        if image is None:
            return
        
        vis_image = image.copy()
        
        for marker in markers:
            # Draw marker corners
            if hasattr(marker, 'corners') and len(marker.corners) >= 8:
                corners = marker.corners
                pts = np.array([
                    [int(corners[0]), int(corners[1])],  # top-left
                    [int(corners[2]), int(corners[3])],  # top-right  
                    [int(corners[4]), int(corners[5])],  # bottom-right
                    [int(corners[6]), int(corners[7])]   # bottom-left
                ], dtype=np.int32)
                
                # Draw marker outline
                cv2.polylines(vis_image, [pts], True, (0, 255, 0), 2)
                
                # Draw center
                center = (int(marker.center_x), int(marker.center_y))
                cv2.circle(vis_image, center, 5, (0, 0, 255), -1)
                
                # Draw ID and distance
                cv2.putText(
                    vis_image,
                    f"ID: {marker.id}",
                    (pts[0][0], pts[0][1] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )
                
                cv2.putText(
                    vis_image,
                    f"Dist: {marker.distance:.2f}m",
                    (pts[0][0], pts[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1
                )
                
                # Draw pose arrow if available
                if hasattr(marker, 'pose'):
                    # Extract yaw from quaternion
                    q = marker.pose.orientation
                    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                   1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                    
                    # Draw orientation arrow
                    arrow_length = 40
                    end_x = int(center[0] + arrow_length * math.cos(yaw))
                    end_y = int(center[1] + arrow_length * math.sin(yaw))
                    
                    cv2.arrowedLine(vis_image, center, (end_x, end_y), 
                                  (255, 0, 0), 2, tipLength=0.3)
        
        # Add info overlay
        info_text = f"Markers: {len(markers)}"
        cv2.putText(vis_image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show window
        cv2.imshow(window_title, vis_image)

    def render(self):
        """Render both camera feeds with marker overlays."""
        
        # Mono camera visualization
        if self.mono_image is not None:
            self.draw_markers_on_image(
                self.mono_image, 
                self.mono_markers, 
                "Mono Camera - AprilTag Detection"
            )
        
        # OAK camera visualization  
        if self.oak_image is not None:
            self.draw_markers_on_image(
                self.oak_image,
                self.oak_markers,
                "OAK Camera - AprilTag Detection" 
            )
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("Shutting down visualizer...")
            rclpy.shutdown()
        elif key == ord('s'):
            self.save_screenshots()

    def save_screenshots(self):
        """Save current visualizations as screenshots."""
        import time
        timestamp = int(time.time())
        
        if self.mono_image is not None and len(self.mono_markers) > 0:
            vis_mono = self.mono_image.copy()
            self.draw_markers_on_image(vis_mono, self.mono_markers, "temp")
            filename = f"mono_detection_{timestamp}.png"
            cv2.imwrite(filename, vis_mono)
            self.get_logger().info(f"Saved mono screenshot: {filename}")
        
        if self.oak_image is not None and len(self.oak_markers) > 0:
            vis_oak = self.oak_image.copy()
            self.draw_markers_on_image(vis_oak, self.oak_markers, "temp")
            filename = f"oak_detection_{timestamp}.png"
            cv2.imwrite(filename, vis_oak)
            self.get_logger().info(f"Saved OAK screenshot: {filename}")

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    node = MarkerDetectionVisualizer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()