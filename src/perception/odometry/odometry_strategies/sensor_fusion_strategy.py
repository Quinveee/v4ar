import math
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
from .base_strategy import BaseOdometryStrategy


class SensorFusionStrategy(BaseOdometryStrategy):
    """Advanced sensor fusion strategy using Extended Kalman Filter (EKF).
    
    Fuses multiple sensors for accurate and stable odometry:
    - rf2o_laser_odometry: High-frequency position updates (motion estimation)
    - IMU: Orientation, angular velocity, linear acceleration
    - Triangulation: Absolute position corrections (low frequency, high accuracy)
    
    Uses EKF to optimally combine sensor measurements with proper noise models.
    """

    def __init__(self):
        super().__init__()
        
        # State: [x, y, theta, vx, vy, omega]
        self.state = np.zeros(6)
        self.state_cov = np.eye(6) * 0.1  # Initial covariance
        
        # Process noise (Q) - how much we trust our motion model
        # Lower values = more trust in model, higher = more uncertainty
        self.Q = np.diag([
            0.01,   # x position noise
            0.01,   # y position noise
            0.005,  # theta orientation noise
            0.05,   # vx velocity noise
            0.05,   # vy velocity noise
            0.02    # omega angular velocity noise
        ])
        
        # Measurement noise (R) - sensor uncertainty
        # rf2o odometry noise
        self.R_rf2o = np.diag([0.02, 0.02, 0.01])  # [x, y, theta]
        
        # IMU noise
        self.R_imu = np.diag([0.01, 0.01, 0.005])  # [theta, omega, ax]
        
        # Triangulation noise (very accurate, low noise)
        self.R_triangulation = np.diag([0.05, 0.05, 0.02])  # [x, y, theta]
        
        # Sensor data storage
        self.last_rf2o_odom = None
        self.last_imu = None
        self.last_triangulation = None
        self.last_update_time = None
        
        # IMU bias estimation (for better accuracy)
        self.imu_gyro_bias = 0.0
        self.imu_accel_bias = np.zeros(2)
        self.imu_bias_samples = []
        self.max_bias_samples = 100
        
        # Velocity estimates
        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0
        
        # Flags
        self.first_rf2o_received = False
        self.first_imu_received = False

    def update(self, *args, **kwargs):
        """Update method - should be called with specific sensor data."""
        # This is a placeholder - actual updates come from specific methods
        pass

    def update_rf2o(self, odom_msg: Odometry, current_time: float):
        """Update with rf2o laser odometry (high frequency motion estimates)."""
        if not self.initialized:
            return
        
        # Extract pose from odometry message
        odom_x = odom_msg.pose.pose.position.x
        odom_y = odom_msg.pose.pose.position.y
        
        # Extract orientation
        qx = odom_msg.pose.pose.orientation.x
        qy = odom_msg.pose.pose.orientation.y
        qz = odom_msg.pose.pose.orientation.z
        qw = odom_msg.pose.pose.orientation.w
        odom_theta = math.atan2(2.0 * (qw * qz + qx * qy),
                                1.0 - 2.0 * (qy * qy + qz * qz))
        
        # Extract velocities from twist
        vx_odom = odom_msg.twist.twist.linear.x
        vy_odom = odom_msg.twist.twist.linear.y
        omega_odom = odom_msg.twist.twist.angular.z
        
        if not self.first_rf2o_received:
            self.last_rf2o_odom = (odom_x, odom_y, odom_theta)
            self.first_rf2o_received = True
            return
        
        # Calculate delta in odom frame
        dx_odom = odom_x - self.last_rf2o_odom[0]
        dy_odom = odom_y - self.last_rf2o_odom[1]
        dtheta_odom = odom_theta - self.last_rf2o_odom[2]
        dtheta_odom = math.atan2(math.sin(dtheta_odom), math.cos(dtheta_odom))
        
        # Calculate time delta
        if self.last_update_time is not None:
            dt = current_time - self.last_update_time
            if dt > 0 and dt < 1.0:  # Sanity check
                # Predict step (motion model)
                self._predict(dt)
                
                # Update step with rf2o measurement
                # Transform delta to world frame
                cos_theta = math.cos(self.state[2])
                sin_theta = math.sin(self.state[2])
                
                dx_world = cos_theta * dx_odom - sin_theta * dy_odom
                dy_world = sin_theta * dx_odom + cos_theta * dy_odom
                
                # Measurement: predicted position + delta
                z = np.array([
                    self.state[0] + dx_world,
                    self.state[1] + dy_world,
                    self.state[2] + dtheta_odom
                ])
                
                # Measurement model: H = I (direct state observation)
                H = np.eye(3)
                H_full = np.zeros((3, 6))
                H_full[:, :3] = H
                
                # Kalman update
                self._update(z, H_full, self.R_rf2o)
        
        self.last_rf2o_odom = (odom_x, odom_y, odom_theta)
        self.last_update_time = current_time
        
        # Update velocity estimates
        self.vx = vx_odom
        self.vy = vy_odom
        self.omega = omega_odom

    def update_imu(self, imu_msg: Imu, current_time: float):
        """Update with IMU data (orientation and angular velocity)."""
        if not self.initialized:
            return
        
        # Extract orientation from quaternion
        qx = imu_msg.orientation.x
        qy = imu_msg.orientation.y
        qz = imu_msg.orientation.z
        qw = imu_msg.orientation.w
        imu_theta = math.atan2(2.0 * (qw * qz + qx * qy),
                               1.0 - 2.0 * (qy * qy + qz * qz))
        
        # Extract angular velocity (compensate for bias)
        omega_imu = imu_msg.angular_velocity.z - self.imu_gyro_bias
        
        # Extract linear acceleration (in base_link frame)
        ax = imu_msg.linear_acceleration.x
        ay = imu_msg.linear_acceleration.y
        
        # Estimate bias during first few seconds (assuming robot is stationary)
        if len(self.imu_bias_samples) < self.max_bias_samples:
            self.imu_bias_samples.append(omega_imu)
            if len(self.imu_bias_samples) == self.max_bias_samples:
                self.imu_gyro_bias = np.mean(self.imu_bias_samples)
        
        if not self.first_imu_received:
            self.first_imu_received = True
            self.last_imu = (imu_theta, omega_imu, ax)
            return
        
        # Calculate time delta
        if self.last_update_time is not None:
            dt = current_time - self.last_update_time
            if dt > 0 and dt < 1.0:
                # Predict step
                self._predict(dt)
                
                # Update with IMU orientation and angular velocity
                z = np.array([
                    imu_theta,
                    omega_imu,
                    ax  # Can use acceleration for velocity estimation
                ])
                
                # Measurement model: observe theta and omega directly
                H = np.zeros((3, 6))
                H[0, 2] = 1.0  # theta
                H[1, 5] = 1.0  # omega
                H[2, 3] = 0.0  # ax (not directly in state, but can be used)
                
                # Kalman update
                self._update(z, H, self.R_imu)
        
        self.last_imu = (imu_theta, omega_imu, ax)

    def update_triangulation(self, pose_msg: PoseStamped):
        """Update with triangulation (absolute position correction)."""
        if not self.initialized:
            return
        
        # Extract pose
        x = pose_msg.pose.position.x
        y = pose_msg.pose.position.y
        
        # Extract orientation
        qx = pose_msg.pose.orientation.x
        qy = pose_msg.pose.orientation.y
        qz = pose_msg.pose.orientation.z
        qw = pose_msg.pose.orientation.w
        theta = math.atan2(2.0 * (qw * qz + qx * qy),
                           1.0 - 2.0 * (qy * qy + qz * qz))
        
        # Triangulation is absolute measurement - use it to correct state
        z = np.array([x, y, theta])
        
        # Measurement model: direct state observation
        H = np.zeros((3, 6))
        H[0, 0] = 1.0  # x
        H[1, 1] = 1.0  # y
        H[2, 2] = 1.0  # theta
        
        # Kalman update (triangulation is very accurate)
        self._update(z, H, self.R_triangulation)
        
        self.last_triangulation = (x, y, theta)

    def _predict(self, dt: float):
        """EKF prediction step (motion model)."""
        # State: [x, y, theta, vx, vy, omega]
        x, y, theta, vx, vy, omega = self.state
        
        # Motion model (constant velocity model)
        if abs(omega) < 1e-6:
            # Straight line motion
            self.state[0] += vx * math.cos(theta) * dt - vy * math.sin(theta) * dt
            self.state[1] += vx * math.sin(theta) * dt + vy * math.cos(theta) * dt
            self.state[2] += omega * dt
        else:
            # Arc motion
            radius = vx / omega if abs(omega) > 1e-6 else 0.0
            dtheta = omega * dt
            self.state[0] += radius * (math.sin(theta + dtheta) - math.sin(theta))
            self.state[1] += radius * (-math.cos(theta + dtheta) + math.cos(theta))
            self.state[2] += dtheta
        
        # Normalize theta
        self.state[2] = math.atan2(math.sin(self.state[2]), math.cos(self.state[2]))
        
        # Jacobian of motion model (F)
        F = np.eye(6)
        F[0, 2] = -vx * math.sin(theta) * dt - vy * math.cos(theta) * dt
        F[0, 3] = math.cos(theta) * dt
        F[0, 4] = -math.sin(theta) * dt
        F[1, 2] = vx * math.cos(theta) * dt - vy * math.sin(theta) * dt
        F[1, 3] = math.sin(theta) * dt
        F[1, 4] = math.cos(theta) * dt
        F[2, 5] = dt
        
        # Update covariance
        self.state_cov = F @ self.state_cov @ F.T + self.Q

    def _update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray):
        """EKF update step (measurement update)."""
        # Predicted measurement
        z_pred = H @ self.state
        
        # Innovation (measurement residual)
        y = z - z_pred
        
        # Normalize angle difference if measuring orientation
        if H.shape[0] >= 3 and H[2, 2] != 0:
            y[2] = math.atan2(math.sin(y[2]), math.cos(y[2]))
        
        # Innovation covariance
        S = H @ self.state_cov @ H.T + R
        
        # Kalman gain
        try:
            K = self.state_cov @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudo-inverse
            K = self.state_cov @ H.T @ np.linalg.pinv(S)
        
        # Update state
        self.state += K @ y
        
        # Normalize theta
        self.state[2] = math.atan2(math.sin(self.state[2]), math.cos(self.state[2]))
        
        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(6) - K @ H
        self.state_cov = I_KH @ self.state_cov @ I_KH.T + K @ R @ K.T
        
        # Ensure covariance is positive definite
        self.state_cov = (self.state_cov + self.state_cov.T) / 2

    def initialize(self, x: float, y: float, theta: float = 0.0):
        """Initialize odometry with starting pose from triangulation."""
        self.state[0] = x
        self.state[1] = y
        self.state[2] = theta
        self.state[3] = 0.0  # vx
        self.state[4] = 0.0  # vy
        self.state[5] = 0.0  # omega
        
        # Update base class state
        self.x = x
        self.y = y
        self.theta = theta
        
        # Reset covariance
        self.state_cov = np.eye(6) * 0.1
        
        # Reset flags
        self.first_rf2o_received = False
        self.first_imu_received = False
        self.last_update_time = None
        
        super().initialize(x, y, theta)

    def get_pose(self):
        """Return current pose as (x, y, theta)."""
        # Update base class state for consistency
        self.x = self.state[0]
        self.y = self.state[1]
        self.theta = self.state[2]
        return self.state[0], self.state[1], self.state[2]

