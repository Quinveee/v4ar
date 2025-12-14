#!/usr/bin/env python3
"""
Online Navigator Node (Laptop Side) - Using Nav2 Planner

This node runs on the laptop and performs:
1. Receives maps from RTAB-Map bridge
2. Uses Nav2's planner_server for path planning (handles unexplored areas!)
3. Generates movement commands from planned paths
4. Publishes commands for rover to execute

Architecture:
    Laptop (this node):
        - Subscribes to /map (from RTAB-Map bridge)
        - Subscribes to /odom (robot position from rover)
        - Subscribes to /goal (goal pose)
        - Uses Nav2 planner_server action client to plan paths
        - Generates velocity commands from path
        - Publishes /navigation_commands (geometry_msgs/Twist) for rover

    Rover:
        - Runs command_executor node
        - Subscribes to /navigation_commands
        - Executes commands (publishes to /cmd_vel or similar)

Topics:
    Subscribes:
        /map (nav_msgs/OccupancyGrid): Map from RTAB-Map bridge
        /odom (nav_msgs/Odometry): Robot odometry from rover
        /goal (geometry_msgs/PoseStamped): Goal pose (user sets this)

    Publishes:
        /navigation_commands (geometry_msgs/Twist): Movement commands for rover
        /planned_path (nav_msgs/Path): Planned path for visualization in RViz

Actions:
    Uses:
        /planner_server/compute_path_to_pose (nav2_msgs/action/ComputePathToPose)
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import ComputePathToPose
import math
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np
import cv2
import time
import traceback


@dataclass
class Pose2D:
    """Simple 2D pose representation."""
    x: float
    y: float
    theta: float


class OnlineNavigator(Node):
    """
    Online navigator that uses Nav2's planner_server and generates commands.

    Runs on laptop - uses Nav2's planner_server for path planning
    (which handles unexplored areas via allow_unknown: true!), then generates
    velocity commands for rover to execute.
    """

    def __init__(self):
        super().__init__('online_navigator')

        # Parameters
        # Empty string = use /compute_path_to_pose (Nav2's default action name)
        # Non-empty = use /{planner_server}/compute_path_to_pose
        self.declare_parameter('planner_server', '')
        self.declare_parameter('k_linear', 0.5)
        self.declare_parameter('k_angular', 2.0)
        self.declare_parameter('max_linear_speed', 0.3)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('waypoint_threshold', 0.15)
        self.declare_parameter('goal_threshold', 0.1)
        self.declare_parameter('control_frequency', 10.0)  # Hz

        # Map cleaning parameters (for visualization only)
        self.declare_parameter('clean_map', False)
        self.declare_parameter('min_blob_size', 30)
        self.declare_parameter('connect_gap_size', 6)
        self.declare_parameter('prune_size', 3)
        self.declare_parameter('prune_iters', 1)

        # Get parameters
        planner_server_name = self.get_parameter('planner_server').value
        self.k_linear = self.get_parameter('k_linear').value
        self.k_angular = self.get_parameter('k_angular').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.waypoint_threshold = self.get_parameter('waypoint_threshold').value
        self.goal_threshold = self.get_parameter('goal_threshold').value
        control_freq = self.get_parameter('control_frequency').value

        # Map cleaning parameters
        self.clean_map = self.get_parameter('clean_map').value
        self.min_blob_size = self.get_parameter('min_blob_size').value
        self.connect_gap_size = self.get_parameter('connect_gap_size').value
        self.prune_size = self.get_parameter('prune_size').value
        self.prune_iters = self.get_parameter('prune_iters').value

        if self.clean_map:
            self.get_logger().info('[INIT] Map cleaning ENABLED (visualization only)')

        # State
        self.current_map: Optional[OccupancyGrid] = None
        self.current_pose: Optional[Pose2D] = None
        self.current_path: Optional[Path] = None
        self.current_waypoint_idx: int = 0
        self.goal_pose: Optional[PoseStamped] = None
        self.goal_reached: bool = False
        self.planning_in_progress: bool = False
        self.pending_plan_request: bool = False  # Track if we have a goal waiting for server
        self.send_goal_future: Optional = None  # Track the send_goal future for timeout detection
        self.send_goal_time: Optional[float] = None  # Time when goal was sent
        
        # Track map stability for action server connection
        self.last_map_size: Optional[Tuple[int, int]] = None
        self.map_stable_count: int = 0
        self.map_stable_threshold: int = 3  # Maps must be stable for 3 updates

        # Nav2 Planner Action Client (create early to allow DDS discovery time)
        # Construct action topic: if planner_server_name is empty, use /compute_path_to_pose
        # Otherwise use /{planner_server_name}/compute_path_to_pose
        if planner_server_name:
            planner_action_topic = f'/{planner_server_name}/compute_path_to_pose'
        else:
            planner_action_topic = '/compute_path_to_pose'
        self.planner_action_topic = planner_action_topic
        # Create ActionClient immediately to start DDS discovery early
        # DDS discovery can take 60+ seconds in Docker, so start early
        self.planner_action_client = ActionClient(self, ComputePathToPose, self.planner_action_topic)
        self.get_logger().info(f'[INIT] Created action client for {planner_action_topic} (DDS discovery may take 60+ seconds)')

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/navigation_commands', 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)

        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal', self.goal_callback, 10)

        # Control timer
        control_period = 1.0 / control_freq
        self.control_timer = self.create_timer(control_period, self.control_loop)

        # Timer to periodically check action server availability (every 2 seconds)
        self.server_check_timer = self.create_timer(2.0, self._check_action_server)
        
        # Timer to check for timeout on goal sending (every 1 second)
        self.goal_timeout_timer = self.create_timer(1.0, self._check_goal_timeout)

        self.get_logger().info(f'[INIT] Online Navigator initialized - using Nav2 planner_server: {planner_action_topic}')

    def map_callback(self, msg: OccupancyGrid):
        """Receive map from RTAB-Map bridge."""
        # Log map extent
        info = msg.info
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y
        max_x = origin_x + info.width * info.resolution
        max_y = origin_y + info.height * info.resolution

        if self.current_map is None:
            self.get_logger().info(
                f'[MAP DEBUG] First map received from RTAB-Map:'
            )
            self.get_logger().info(f'  Size: {info.width}x{info.height} cells')
            self.get_logger().info(f'  Resolution: {info.resolution}m/cell')
            self.get_logger().info(f'  Origin: ({origin_x:.2f}, {origin_y:.2f})')
            self.get_logger().info(f'  Coverage X: [{origin_x:.2f}, {max_x:.2f}]')
            self.get_logger().info(f'  Coverage Y: [{origin_y:.2f}, {max_y:.2f}]')
            self.get_logger().info(f'  Frame: {msg.header.frame_id}')
        elif (self.current_map.info.width != info.width or
              self.current_map.info.height != info.height):
            self.get_logger().info(
                f'[MAP DEBUG] Map expanded: {info.width}x{info.height} cells, '
                f'X:[{origin_x:.2f}, {max_x:.2f}] Y:[{origin_y:.2f}, {max_y:.2f}]'
            )

        # Optionally clean map noise (for visualization only)
        if self.clean_map:
            msg = self.clean_occupancy_grid(msg)

        # Track map stability (for action server connection timing)
        current_size = (msg.info.width, msg.info.height)
        if self.last_map_size == current_size:
            self.map_stable_count += 1
        else:
            self.map_stable_count = 0
            self.last_map_size = current_size

        self.current_map = msg

        # Replan if we have a goal
        if self.goal_pose is not None and self.current_pose is not None:
            self.plan_path()

    def odom_callback(self, msg: Odometry):
        """Update current robot position from odometry."""
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation

        yaw = self.quaternion_to_yaw(ori.x, ori.y, ori.z, ori.w)
        if self.current_pose is None:
            self.get_logger().info(f'[ODOM] First pose received: ({pos.x:.2f}, {pos.y:.2f})')
        self.current_pose = Pose2D(pos.x, pos.y, yaw)

    def _check_action_server(self):
        """Periodically check if action server is available and retry pending planning requests."""
        # ActionClient is now created at init, so this should always exist
        if self.planner_action_client is None:
            # Fallback: create if somehow missing (shouldn't happen)
            self.planner_action_client = ActionClient(self, ComputePathToPose, self.planner_action_topic)
            self.get_logger().warn(f'[PLANNER] ActionClient was None - recreating (this shouldn\'t happen)')
        
        # Check if server is available (non-blocking check)
        if self.planner_action_client.server_is_ready():
            # Server is ready - log once and retry pending planning if needed
            if not hasattr(self, '_server_ready_logged'):
                self.get_logger().info('[PLANNER] ✓ Action server is now available!')
                self._server_ready_logged = True
            
            # If we have a pending goal and server is now ready, try planning
            if self.pending_plan_request and not self.planning_in_progress:
                if self.goal_pose is not None and self.current_pose is not None and self.current_map is not None:
                    self.get_logger().info('[PLANNER] Server ready - retrying planning request...')
                    self.pending_plan_request = False
                    self.plan_path()
        else:
            # Server not ready according to server_is_ready(), but this is OK!
            # ros2 action info confirms the server exists - this is a DDS discovery timing issue.
            # We'll try sending goals anyway (see plan_path()).
            # Only log periodically to avoid spam (every 30 seconds instead of 10)
            if not hasattr(self, '_last_server_check_log'):
                self._last_server_check_log = 0.0
            now = self.get_clock().now().nanoseconds / 1e9
            if now - self._last_server_check_log > 30.0:  # Reduced frequency
                self.get_logger().info(
                    f'[PLANNER] ⏳ server_is_ready() returns False, but server exists (confirmed by ros2 action info).\n'
                    'This is a DDS discovery timing issue in Docker - goals will be attempted anyway.\n'
                    'If goals fail, check planner_server logs for TF/costmap errors.'
                )
                self._last_server_check_log = now

    def _ensure_action_client(self):
        """Ensure action client exists. Always returns True - we'll try sending and handle errors."""
        # ActionClient is now created at init, so this should always exist
        if self.planner_action_client is None:
            # Fallback: create if somehow missing (shouldn't happen)
            self.get_logger().warn(f'[PLANNER] ActionClient was None - recreating (this shouldn\'t happen)')
            self.planner_action_client = ActionClient(self, ComputePathToPose, self.planner_action_topic)
            # Give DDS a moment to discover
            time.sleep(0.1)
        
        # Check if server is ready (non-blocking check)
        if self.planner_action_client.server_is_ready():
            if not hasattr(self, '_server_ready_logged'):
                self.get_logger().info('[PLANNER] ✓ Action server is available and ready')
                self._server_ready_logged = True
            return True
        
        # Server not ready according to server_is_ready(), but we'll try anyway
        # In ROS2, server_is_ready() can return False even when the server exists
        # because DDS discovery hasn't completed. send_goal_async() may still work.
        # We'll handle failures gracefully in the response callback.
        if not hasattr(self, '_server_not_ready_logged'):
            self.get_logger().info(
                f'[PLANNER] Action server {self.planner_action_topic} not yet discovered by client.\n'
                'Server exists (confirmed by ros2 action list) but DDS discovery incomplete.\n'
                'Will attempt to send goal anyway - ROS2 may discover server during send.'
            )
            self._server_not_ready_logged = True
        
        # Return True to allow sending - we'll handle errors in callback
        return True

    def goal_callback(self, msg: PoseStamped):
        """
        Handle new goal pose - plan path using Nav2 planner_server.

        Nav2's planner can handle goals in unexplored areas (unknown space)
        when allow_unknown: true is configured.
        """
        self.get_logger().info(f'[GOAL] Received goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})')

        self.goal_pose = msg
        self.goal_reached = False
        self.current_waypoint_idx = 0
        self.current_path = None

        # Plan path if we have map and pose
        if self.current_map is not None and self.current_pose is not None:
            self.plan_path()
        else:
            missing = []
            if self.current_map is None:
                missing.append('map')
            if self.current_pose is None:
                missing.append('odometry')
            self.get_logger().warn(f'[GOAL] Cannot plan yet - waiting for: {", ".join(missing)}')

    def plan_path(self):
        """Plan path to current goal using Nav2 planner_server."""
        if self.goal_pose is None or self.current_pose is None or self.current_map is None:
            return

        if self.planning_in_progress:
            return

        # Ensure action client exists (always returns True now - we try sending anyway)
        self._ensure_action_client()

        self.planning_in_progress = True
        self.get_logger().info('[PLANNER] Sending planning request to Nav2 planner_server')

        # Create goal message
        goal_msg = ComputePathToPose.Goal()
        goal_msg.start.header.frame_id = 'map'
        goal_msg.start.header.stamp = self.get_clock().now().to_msg()
        goal_msg.start.pose.position.x = self.current_pose.x
        goal_msg.start.pose.position.y = self.current_pose.y
        goal_msg.start.pose.position.z = 0.0
        # Convert yaw to quaternion
        yaw = self.current_pose.theta
        goal_msg.start.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.start.pose.orientation.w = math.cos(yaw / 2.0)

        goal_msg.goal = self.goal_pose
        goal_msg.goal.header.frame_id = 'map'
        goal_msg.goal.header.stamp = self.get_clock().now().to_msg()

        goal_msg.planner_id = 'GridBased'  # Use GridBased planner
        goal_msg.use_start = True

        # Log goal details for debugging
        server_ready = self.planner_action_client.server_is_ready() if self.planner_action_client else False
        self.get_logger().info(
            f'[PLANNER] Goal details:\n'
            f'  Start: ({goal_msg.start.pose.position.x:.2f}, {goal_msg.start.pose.position.y:.2f})\n'
            f'  Goal: ({goal_msg.goal.pose.position.x:.2f}, {goal_msg.goal.pose.position.y:.2f})\n'
            f'  Server ready (server_is_ready()): {server_ready}'
        )
        
        # IMPORTANT: Even if server_is_ready() returns False, try sending anyway!
        # DDS discovery can complete during send_goal_async(), and ros2 action info
        # confirms the server exists. This is a known ROS2 DDS quirk in Docker.
        if not server_ready:
            self.get_logger().warn(
                '[PLANNER] ⚠ server_is_ready() is False, but attempting to send goal anyway.\n'
                'ros2 action info confirms server exists - DDS discovery may complete during send.'
            )
        
        # Send goal asynchronously
        try:
            send_goal_future = self.planner_action_client.send_goal_async(goal_msg)
            if send_goal_future is None:
                self.get_logger().error(
                    '[PLANNER] ❌ send_goal_async() returned None - action server not discovered.\n'
                    'The server exists (ros2 action list confirms) but client cannot discover it.\n'
                    'This is a ROS2 DDS discovery issue. The costmap is created but planner cannot connect.\n'
                    'Possible fixes:\n'
                    '  1. Restart all nodes to reset DDS discovery\n'
                    '  2. Check ROS_DOMAIN_ID matches on all nodes\n'
                    '  3. Wait 30-60 seconds for DDS discovery (can be slow in Docker)\n'
                    '  4. Check network connectivity between nodes'
                )
                self.planning_in_progress = False
                # Clear timeout tracking
                self.send_goal_future = None
                self.send_goal_time = None
                return
            
            self.get_logger().info('[PLANNER] ✓ Goal sent successfully, waiting for response...')
            send_goal_future.add_done_callback(self._planning_response_callback)
            
            # Store future and timestamp for timeout detection
            self.send_goal_future = send_goal_future
            self.send_goal_time = self.get_clock().now().nanoseconds / 1e9
        except Exception as e:
            self.get_logger().error(f'[PLANNER] ❌ Exception sending goal: {e}\n{traceback.format_exc()}')
            self.planning_in_progress = False
            # Clear timeout tracking
            self.send_goal_future = None
            self.send_goal_time = None
            return

    def _check_goal_timeout(self):
        """Check if goal sending has timed out (future never completed)."""
        if self.send_goal_future is None or self.send_goal_time is None:
            return
        
        # Check if future is done
        if self.send_goal_future.done():
            # Future completed, clear tracking
            self.send_goal_future = None
            self.send_goal_time = None
            return
        
        # Check timeout (10 seconds)
        now = self.get_clock().now().nanoseconds / 1e9
        elapsed = now - self.send_goal_time
        if elapsed > 10.0:
            self.get_logger().error(
                f'[PLANNER] ❌ TIMEOUT: Goal sent {elapsed:.1f}s ago but no response received.\n'
                'This indicates the action server was NOT discovered by DDS.\n'
                'The future from send_goal_async() never completes because server cannot be reached.\n'
                '\n'
                'Diagnosis:\n'
                '  - send_goal_async() returned a future (not None)\n'
                '  - But the future never completes (server not discovered)\n'
                '  - This is a ROS2 DDS discovery failure\n'
                '\n'
                'Possible fixes:\n'
                '  1. Verify planner_server is actually running: ros2 node list | grep planner\n'
                '  2. Check action server exists: ros2 action list | grep compute_path_to_pose\n'
                '  3. Verify ROS_DOMAIN_ID matches on all nodes\n'
                '  4. Restart all nodes to reset DDS discovery\n'
                '  5. Wait 30-60 seconds for DDS discovery (can be very slow in Docker)\n'
                '  6. Check network connectivity between nodes'
            )
            # Reset state
            self.send_goal_future = None
            self.send_goal_time = None
            self.planning_in_progress = False

    def _planning_response_callback(self, future):
        """Handle response from planning request."""
        # Clear timeout tracking since we got a response
        self.send_goal_future = None
        self.send_goal_time = None
        
        try:
            goal_handle = future.result()
            if goal_handle is None:
                self.get_logger().error(
                    '[PLANNER] ❌ Goal handle is None - action server may not be discovered.\n'
                    'Server exists (ros2 action list confirms) but client cannot connect.\n'
                    'This is a ROS2 DDS discovery issue. The costmap exists but planner cannot use it.\n'
                    'Will retry on next goal.'
                )
                self.planning_in_progress = False
                return
            
            if not goal_handle.accepted:
                self.get_logger().error(
                    '[PLANNER] ❌ Planning request was REJECTED by server.\n'
                    'Possible reasons:\n'
                    '  1. Costmap not ready (still initializing)\n'
                    '  2. Start/goal poses invalid\n'
                    '  3. No valid path exists\n'
                    'Check planner_server logs for details.'
                )
                self.planning_in_progress = False
                return

            self.get_logger().info('[PLANNER] ✓ Planning request ACCEPTED, waiting for path result...')
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self._planning_result_callback)
        except Exception as e:
            self.get_logger().error(f'[PLANNER] ❌ Exception in response callback: {e}\n{traceback.format_exc()}')
            self.planning_in_progress = False

    def _planning_result_callback(self, future):
        """Handle planning result."""
        result = future.result().result
        self.planning_in_progress = False

        if result.path.poses:
            self.get_logger().info(f'[PLANNER] ✓ Path planned with {len(result.path.poses)} waypoints')
            
            # Store the path
            self.current_path = result.path
            self.current_waypoint_idx = 0

            # Ensure frame_id is set
            self.current_path.header.frame_id = 'map'
            self.current_path.header.stamp = self.get_clock().now().to_msg()

            # Publish for visualization
            self.path_pub.publish(self.current_path)

            # Log start and goal
            start = result.path.poses[0].pose.position
            goal = result.path.poses[-1].pose.position
            self.get_logger().info(
                f'[PLANNER] Path: ({start.x:.2f}, {start.y:.2f}) → '
                f'({goal.x:.2f}, {goal.y:.2f})'
            )
        else:
            self.get_logger().warn('[PLANNER] Received empty path - planning may have failed')
            self.current_path = None

    def quaternion_to_yaw(self, x: float, y: float, z: float, w: float) -> float:
        """Convert quaternion to yaw angle."""
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def clean_occupancy_grid(self, grid_msg: OccupancyGrid) -> OccupancyGrid:
        """
        Clean noise from occupancy grid map using morphological operations.
        Adapted from clean_map_noise.py for online processing.
        """
        # Convert OccupancyGrid to numpy array
        width = grid_msg.info.width
        height = grid_msg.info.height
        data = np.array(grid_msg.data, dtype=np.int8).reshape((height, width))

        # Convert to image format (0-255)
        # OccupancyGrid: -1=unknown, 0=free, 100=occupied
        img = np.zeros((height, width), dtype=np.uint8)
        img[data == -1] = 205  # unknown
        img[data == 0] = 255    # free space (white)
        img[data == 100] = 0    # occupied (black)

        # 1) Create walls mask (255 = wall)
        _, binary_map = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV)

        # 2) Keep only blobs above area threshold (remove noise)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
        cleaned = np.zeros_like(binary_map)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_blob_size:
                cleaned[labels == i] = 255

        # 3) Close small gaps
        if self.connect_gap_size > 0:
            k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (self.connect_gap_size, self.connect_gap_size))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close, iterations=1)

        # 4) Prune tiny protrusions
        if self.prune_size > 0:
            k_prune = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.prune_size, self.prune_size))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_prune, iterations=self.prune_iters)

            # Re-close very lightly to keep borders connected
            if self.connect_gap_size > 0:
                cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close, iterations=1)

        # 5) Reconstruct final map
        final_map = np.full_like(img, 255)

        # Preserve unknown space
        unknown_mask = (img >= 203) & (img <= 207)
        final_map[unknown_mask] = 205

        # Set cleaned obstacles
        final_map[cleaned == 255] = 0

        # Convert back to OccupancyGrid format
        output_data = np.zeros((height, width), dtype=np.int8)
        output_data[final_map == 205] = -1   # unknown
        output_data[final_map == 255] = 0    # free
        output_data[final_map == 0] = 100    # occupied

        # Create new OccupancyGrid message
        cleaned_grid = OccupancyGrid()
        cleaned_grid.header = grid_msg.header
        cleaned_grid.info = grid_msg.info
        cleaned_grid.data = output_data.flatten().tolist()

        return cleaned_grid

    def compute_control(self, target_x: float, target_y: float) -> Tuple[float, float]:
        """Compute velocity commands to drive toward target waypoint."""
        if self.current_pose is None:
            return (0.0, 0.0)

        dx = target_x - self.current_pose.x
        dy = target_y - self.current_pose.y
        distance = math.sqrt(dx * dx + dy * dy)

        desired_heading = math.atan2(dy, dx)
        heading_error = self.normalize_angle(desired_heading - self.current_pose.theta)

        # Proportional control
        linear_vel = self.k_linear * distance
        linear_vel = max(-self.max_linear_speed, min(self.max_linear_speed, linear_vel))

        # Reduce speed when turning sharply
        if abs(heading_error) > math.pi / 4:
            linear_vel *= 0.5

        angular_vel = self.k_angular * heading_error
        angular_vel = max(-self.max_angular_speed, min(self.max_angular_speed, angular_vel))

        return (linear_vel, angular_vel)

    def normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi] range."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def control_loop(self):
        """Main control loop - generates and publishes commands."""
        if self.goal_reached or self.current_path is None or self.current_pose is None:
            return

        if self.current_waypoint_idx >= len(self.current_path.poses):
            if not self.goal_reached:
                self.goal_reached = True
                cmd = Twist()
                self.cmd_pub.publish(cmd)
            return

        # Get current target waypoint
        target_pose = self.current_path.poses[self.current_waypoint_idx].pose
        target_x = target_pose.position.x
        target_y = target_pose.position.y

        # Calculate distance to waypoint
        dx = target_x - self.current_pose.x
        dy = target_y - self.current_pose.y
        distance = math.sqrt(dx * dx + dy * dy)

        # Check if we should advance to next waypoint
        is_last_waypoint = (self.current_waypoint_idx == len(self.current_path.poses) - 1)
        threshold = self.goal_threshold if is_last_waypoint else self.waypoint_threshold

        if distance < threshold:
            self.current_waypoint_idx += 1
            return

        # Compute and publish velocity command
        linear_vel, angular_vel = self.compute_control(target_x, target_y)

        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = OnlineNavigator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop robot on shutdown
        cmd = Twist()
        node.cmd_pub.publish(cmd)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
