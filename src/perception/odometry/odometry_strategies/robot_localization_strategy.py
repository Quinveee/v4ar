import math
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from .base_strategy import BaseOdometryStrategy


class RobotLocalizationStrategy(BaseOdometryStrategy):
    """Strategy using robot_localization's EKF/UKF filtered odometry.
    
    This strategy subscribes to robot_localization's filtered odometry output
    (typically /odometry/filtered) which fuses multiple sensors:
    - rf2o_laser_odometry
    - IMU
    - Wheel odometry (if available)
    - GPS (if available)
    
    robot_localization uses an Extended/Unscented Kalman Filter to optimally
    combine all sensor measurements with proper noise models.
    
    This is the recommended approach for production systems as it's:
    - Well-tested and maintained
    - Handles sensor fusion automatically
    - Configurable via parameters
    - Supports multiple sensor types
    """

    def __init__(self):
        super().__init__()
        self.last_odom_x = 0.0
        self.last_odom_y = 0.0
        self.last_odom_theta = 0.0
        self.first_odom_received = False

    def update(self, odom_msg: Odometry):
        """Update pose using robot_localization's filtered odometry.
        
        Args:
            odom_msg: Filtered Odometry message from robot_localization
                      (typically from /odometry/filtered topic)
        """
        if not self.initialized:
            return

        # Extract pose from filtered odometry message
        odom_x = odom_msg.pose.pose.position.x
        odom_y = odom_msg.pose.pose.position.y
        
        # Extract orientation (quaternion to yaw)
        qx = odom_msg.pose.pose.orientation.x
        qy = odom_msg.pose.pose.orientation.y
        qz = odom_msg.pose.pose.orientation.z
        qw = odom_msg.pose.pose.orientation.w
        
        # Convert quaternion to yaw
        odom_theta = math.atan2(2.0 * (qw * qz + qx * qy),
                                1.0 - 2.0 * (qy * qy + qz * qz))

        if not self.first_odom_received:
            # First odometry message - store as reference
            # robot_localization typically starts at (0,0,0) in odom frame
            self.last_odom_x = odom_x
            self.last_odom_y = odom_y
            self.last_odom_theta = odom_theta
            self.first_odom_received = True
            return

        # Calculate delta in odometry frame
        dx_odom = odom_x - self.last_odom_x
        dy_odom = odom_y - self.last_odom_y
        dtheta_odom = odom_theta - self.last_odom_theta
        
        # Normalize angle difference
        dtheta_odom = math.atan2(math.sin(dtheta_odom), math.cos(dtheta_odom))

        # Transform delta from odom frame to world frame
        # Use current world heading to transform the motion delta
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)
        
        # Rotate the motion delta to world frame coordinates
        dx_world = cos_theta * dx_odom - sin_theta * dy_odom
        dy_world = sin_theta * dx_odom + cos_theta * dy_odom

        # Update world pose (anchored to triangulation initialization)
        self.x += dx_world
        self.y += dy_world
        self.theta += dtheta_odom
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        # Update last odometry values for next delta calculation
        self.last_odom_x = odom_x
        self.last_odom_y = odom_y
        self.last_odom_theta = odom_theta

