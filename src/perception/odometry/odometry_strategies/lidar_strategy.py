import math
import numpy as np
from nav_msgs.msg import Odometry
from .base_strategy import BaseOdometryStrategy


class RF2OStrategy(BaseOdometryStrategy):
    """Odometry strategy using rf2o_laser_odometry (laser scan matching).
    
    This strategy:
    1. Receives rf2o's odometry (in local 'odom' frame, starts at 0,0,0)
    2. Computes motion deltas from rf2o
    3. Transforms deltas to world frame (anchored to triangulation initialization)
    4. Accumulates world pose using rf2o's motion estimates
    """

    def __init__(self):
        super().__init__()
        self.last_odom_x = 0.0
        self.last_odom_y = 0.0
        self.last_odom_theta = 0.0
        self.first_odom_received = False
        # Store initial odom orientation to handle frame rotation
        self.initial_odom_theta = None

    def update(self, odom_msg: Odometry):
        """Update pose using rf2o laser odometry.
        
        Args:
            odom_msg: Odometry message from rf2o (in 'odom' frame)
        """
        if not self.initialized:
            return

        # Extract pose from odometry message
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
            # This is when rf2o starts (typically 0,0,0 in odom frame)
            self.last_odom_x = odom_x
            self.last_odom_y = odom_y
            self.last_odom_theta = odom_theta
            self.initial_odom_theta = odom_theta
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

