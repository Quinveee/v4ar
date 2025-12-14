#!/usr/bin/env python3
"""
RTAB-Map to Occupancy Grid Bridge

Converts RTAB-Map's map output to standard nav_msgs/OccupancyGrid for navigation.
RTAB-Map can publish occupancy grids directly, but this bridge ensures compatibility
and handles map updates dynamically.

Topics:
    Subscribes:
        /rtabmap/grid_map (nav_msgs/OccupancyGrid): RTAB-Map's 2D occupancy grid (if available)
        /rtabmap/mapData (rtabmap_msgs/MapData): RTAB-Map map data (alternative source)
    
    Publishes:
        /map (nav_msgs/OccupancyGrid): Standard occupancy grid for navigation planners
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
import numpy as np
from typing import Optional

# Optional import for MapData (may not be available)
try:
    from rtabmap_msgs.msg import MapData
    HAS_RTABMAP_MSGS = True
except ImportError:
    HAS_RTABMAP_MSGS = False
    MapData = None


class RTABMapBridge(Node):
    """Bridge RTAB-Map output to standard occupancy grid for navigation."""
    
    def __init__(self):
        super().__init__('rtabmap_bridge')
        
        # Parameters
        self.declare_parameter('map_frame_id', 'map')
        self.declare_parameter('use_rtabmap_grid', True)  # Use RTAB-Map's grid_map if available
        
        map_frame = self.get_parameter('map_frame_id').value
        use_rtabmap_grid = self.get_parameter('use_rtabmap_grid').value
        
        # State
        self.last_map: Optional[OccupancyGrid] = None
        
        # Publisher with TRANSIENT_LOCAL durability (required by Nav2 costmap)
        # This ensures late subscribers (like planner_server) can get the latest map
        map_qos = QoSProfile(
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.map_pub = self.create_publisher(
            OccupancyGrid, '/map', map_qos)
        
        # Subscribers
        if use_rtabmap_grid:
            # RTAB-Map can publish occupancy grid directly if configured with Grid/2D
            # Use TRANSIENT_LOCAL to match RTAB-Map's QoS
            grid_qos = QoSProfile(
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10
            )
            self.grid_sub = self.create_subscription(
                OccupancyGrid, '/rtabmap/grid_map', self.grid_map_callback, grid_qos)
            self.get_logger().info('Subscribed to /rtabmap/grid_map')
        else:
            if HAS_RTABMAP_MSGS:
                # Alternative: subscribe to MapData and extract grid
                self.map_data_sub = self.create_subscription(
                    MapData, '/rtabmap/mapData', self.map_data_callback, 10)
                self.get_logger().info('Subscribed to /rtabmap/mapData')
            else:
                self.get_logger().error(
                    'rtabmap_msgs not available. Cannot subscribe to MapData. '
                    'Set use_rtabmap_grid:=true or install rtabmap_msgs.'
                )
        
        self.get_logger().info('RTAB-Map bridge initialized')
        self.get_logger().info(f'Publishing to /map with frame_id: {map_frame}')
    
    def grid_map_callback(self, msg: OccupancyGrid):
        """
        Handle RTAB-Map's occupancy grid directly.
        
        This is the preferred method - RTAB-Map publishes occupancy grids
        if configured with Grid/2D parameters.
        """
        # Ensure frame_id is correct
        msg.header.frame_id = self.get_parameter('map_frame_id').value
        
        # Publish directly
        self.map_pub.publish(msg)
        self.last_map = msg
        
        self.get_logger().debug(
            f'Published occupancy grid: {msg.info.width}x{msg.info.height}, '
            f'resolution={msg.info.resolution}m'
        )
    
    def map_data_callback(self, msg: MapData):
        """
        Extract occupancy grid from RTAB-Map MapData.
        
        This is a fallback method if RTAB-Map doesn't publish grid_map directly.
        """
        # RTAB-Map's MapData contains grid information in msg.grid_ground and msg.grid_obstacles
        # For now, we'll log that this method needs implementation
        # In practice, RTAB-Map should be configured to publish grid_map directly
        
        self.get_logger().warn(
            'MapData callback received but grid extraction not implemented. '
            'Configure RTAB-Map to publish /rtabmap/grid_map instead.'
        )


def main(args=None):
    rclpy.init(args=args)
    node = RTABMapBridge()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

