#!/usr/bin/env python3
"""
Dijkstra Path Planner Node for UGV Rover

This node implements a 2D path planner using Dijkstra's algorithm on an occupancy grid map.
It subscribes to a map (either from /map topic or loaded from PGM+YAML files), receives goal
poses, and plans collision-free paths from the robot's current position to the goal.

Topics:
    Subscribes:
        /map (nav_msgs/OccupancyGrid): 2D occupancy grid map
        /goal (geometry_msgs/PoseStamped): Goal pose in map frame
        /odom (nav_msgs/Odometry): Robot odometry for start position
    
    Publishes:
        /planned_path (nav_msgs/Path): Planned path from start to goal in map frame

Parameters:
    map_file (string): Path to map YAML file (if loading from file instead of /map topic)
    occupied_threshold (int): Occupancy values >= this are considered obstacles (default: 50)
    use_map_topic (bool): If true, subscribe to /map; if false, load from file (default: true)
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped
import heapq
import math
import yaml
from PIL import Image
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set


@dataclass
class GridCell:
    """Represents a cell in the occupancy grid."""
    row: int
    col: int
    
    def __hash__(self):
        return hash((self.row, self.col))
    
    def __eq__(self, other):
        return self.row == other.row and self.col == other.col


@dataclass
class MapInfo:
    """Stores map metadata for coordinate conversions."""
    resolution: float  # meters per cell
    width: int  # cells
    height: int  # cells
    origin_x: float  # world coordinates of (0,0) cell
    origin_y: float  # world coordinates of (0,0) cell
    data: np.ndarray  # occupancy values (height x width)


class DijkstraPlanner(Node):
    """Path planner using Dijkstra's algorithm on 2D occupancy grid."""
    
    # 8-connected neighbors: [dx, dy, cost]
    # Cardinal directions (N, S, E, W) have cost 1.0
    # Diagonal directions have cost sqrt(2) ≈ 1.414
    NEIGHBORS = [
        (-1, 0, 1.0),      # North
        (1, 0, 1.0),       # South
        (0, 1, 1.0),       # East
        (0, -1, 1.0),      # West
        (-1, -1, 1.414),   # NW
        (-1, 1, 1.414),    # NE
        (1, -1, 1.414),    # SW
        (1, 1, 1.414),     # SE
    ]
    
    def __init__(self):
        super().__init__('dijkstra_planner')
        
        # Declare parameters
        self.declare_parameter('map_file', '')
        self.declare_parameter('occupied_threshold', 50)
        self.declare_parameter('use_map_topic', True)
        
        # Get parameters
        self.map_file = self.get_parameter('map_file').value
        self.occupied_threshold = self.get_parameter('occupied_threshold').value
        self.use_map_topic = self.get_parameter('use_map_topic').value
        
        # State
        self.map_info: Optional[MapInfo] = None
        self.current_pose: Optional[Tuple[float, float]] = None  # (x, y) in map frame
        
        # Publishers
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        
        # Subscribers
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal', self.goal_callback, 10)
        
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        
        # Subscribe to map topic or load from file
        if self.use_map_topic:
            self.map_sub = self.create_subscription(
                OccupancyGrid, '/map', self.map_callback, 10)
            self.get_logger().info('Waiting for map on /map topic...')
        else:
            if self.map_file:
                self.load_map_from_file(self.map_file)
            else:
                self.get_logger().error('map_file parameter not set!')
        
        self.get_logger().info('Dijkstra planner initialized')
    
    def odom_callback(self, msg: Odometry):
        """Update current robot position from odometry."""
        self.current_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )
    
    def map_callback(self, msg: OccupancyGrid):
        """Receive map from /map topic and store it."""
        self.get_logger().info(f'Received map: {msg.info.width}x{msg.info.height}, '
                               f'resolution={msg.info.resolution}m')
        
        # Convert map data to numpy array (row-major: height x width)
        data = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        
        self.map_info = MapInfo(
            resolution=msg.info.resolution,
            width=msg.info.width,
            height=msg.info.height,
            origin_x=msg.info.origin.position.x,
            origin_y=msg.info.origin.position.y,
            data=data
        )
    
    def load_map_from_file(self, yaml_path: str):
        """
        Load map from PGM + YAML files.
        
        The YAML file contains metadata (resolution, origin, etc.) and references
        the PGM image file containing occupancy data.
        """
        try:
            with open(yaml_path, 'r') as f:
                map_yaml = yaml.safe_load(f)
            
            # Load PGM image
            image_path = map_yaml['image']
            # Handle relative path (relative to YAML file location)
            if not image_path.startswith('/'):
                import os
                yaml_dir = os.path.dirname(yaml_path)
                image_path = os.path.join(yaml_dir, image_path)
            
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Convert pixel values to occupancy values
            # PGM typically: 255=free, 0=occupied, 205=unknown
            # ROS convention: 0=free, 100=occupied, -1=unknown
            # We'll map: white(255)->0(free), black(0)->100(occupied), gray(205)->-1
            occupancy = np.zeros_like(img_array, dtype=np.int8)
            occupancy[img_array == 0] = 100  # black -> occupied
            occupancy[img_array == 205] = -1  # gray -> unknown
            occupancy[img_array == 255] = 0  # white -> free
            
            self.map_info = MapInfo(
                resolution=map_yaml['resolution'],
                width=img_array.shape[1],
                height=img_array.shape[0],
                origin_x=map_yaml['origin'][0],
                origin_y=map_yaml['origin'][1],
                data=occupancy
            )
            
            self.get_logger().info(f'Loaded map from {yaml_path}: '
                                   f'{self.map_info.width}x{self.map_info.height}, '
                                   f'resolution={self.map_info.resolution}m')
        
        except Exception as e:
            self.get_logger().error(f'Failed to load map from {yaml_path}: {e}')
    
    def world_to_grid(self, x: float, y: float) -> Optional[GridCell]:
        """
        Convert world coordinates (meters) to grid indices.
        
        World coordinate system:
            - Origin is defined by map's origin (origin_x, origin_y)
            - X-axis points right, Y-axis points up
        
        Grid coordinate system:
            - (0, 0) is top-left corner
            - row increases downward
            - col increases rightward
        
        Conversion:
            col = (x - origin_x) / resolution
            row = (height - 1) - (y - origin_y) / resolution
        
        Returns None if coordinates are outside the map.
        """
        if self.map_info is None:
            return None
        
        # Calculate grid indices
        col = int((x - self.map_info.origin_x) / self.map_info.resolution)
        row = int((self.map_info.height - 1) - 
                  (y - self.map_info.origin_y) / self.map_info.resolution)
        
        # Check bounds
        if 0 <= row < self.map_info.height and 0 <= col < self.map_info.width:
            return GridCell(row, col)
        
        return None
    
    def grid_to_world(self, cell: GridCell) -> Tuple[float, float]:
        """
        Convert grid indices to world coordinates (center of cell).
        
        Returns (x, y) in world frame (meters).
        """
        x = self.map_info.origin_x + (cell.col + 0.5) * self.map_info.resolution
        y = self.map_info.origin_y + \
            (self.map_info.height - 1 - cell.row + 0.5) * self.map_info.resolution
        
        return (x, y)
    
    def is_free(self, cell: GridCell) -> bool:
        """
        Check if a grid cell is free (not occupied or unknown).
        
        A cell is free if its occupancy value is below the occupied threshold.
        Unknown cells (-1) and occupied cells (>= threshold) are not free.
        """
        if self.map_info is None:
            return False
        
        if cell.row < 0 or cell.row >= self.map_info.height:
            return False
        if cell.col < 0 or cell.col >= self.map_info.width:
            return False
        
        occupancy = self.map_info.data[cell.row, cell.col]
        
        # Unknown cells are not free
        if occupancy < 0:
            return False
        
        # Check against threshold
        return occupancy < self.occupied_threshold
    
    def get_neighbors(self, cell: GridCell) -> List[Tuple[GridCell, float]]:
        """
        Get valid neighbor cells and their movement costs.
        
        Returns list of (neighbor_cell, cost) tuples.
        Only includes neighbors that are:
            - Within map bounds
            - Not occupied
            - Not unknown
        """
        neighbors = []
        
        for drow, dcol, cost in self.NEIGHBORS:
            neighbor = GridCell(cell.row + drow, cell.col + dcol)
            
            if self.is_free(neighbor):
                neighbors.append((neighbor, cost))
        
        return neighbors
    
    def dijkstra(self, start: GridCell, goal: GridCell) -> Optional[List[GridCell]]:
        """
        Run Dijkstra's algorithm to find shortest path from start to goal.
        
        Algorithm:
            1. Initialize distance to all cells as infinity, except start (distance = 0)
            2. Use a priority queue (min-heap) to process cells by increasing distance
            3. For each cell, explore all neighbors:
                - Calculate tentative distance = current distance + edge cost
                - If tentative distance < neighbor's distance, update it
            4. Continue until goal is reached or queue is empty
            5. Reconstruct path by backtracking from goal to start
        
        Returns:
            List of GridCell from start to goal, or None if no path exists.
        """
        if not self.is_free(start):
            self.get_logger().warn('Start cell is not free!')
            return None
        
        if not self.is_free(goal):
            self.get_logger().warn('Goal cell is not free!')
            return None
        
        # Priority queue: (distance, cell)
        # heapq is a min-heap, so smallest distance is popped first
        pq = [(0.0, start)]
        
        # Distance from start to each cell
        distances = {start: 0.0}
        
        # Parent pointers for path reconstruction
        parents = {}
        
        # Visited set for efficiency
        visited: Set[GridCell] = set()
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            # Skip if already visited
            if current in visited:
                continue
            
            visited.add(current)
            
            # Goal reached!
            if current == goal:
                # Reconstruct path
                path = []
                node = goal
                while node in parents:
                    path.append(node)
                    node = parents[node]
                path.append(start)
                path.reverse()
                return path
            
            # Explore neighbors
            for neighbor, edge_cost in self.get_neighbors(current):
                if neighbor in visited:
                    continue
                
                # Calculate tentative distance through current cell
                tentative_dist = current_dist + edge_cost
                
                # Update if this is a better path
                if neighbor not in distances or tentative_dist < distances[neighbor]:
                    distances[neighbor] = tentative_dist
                    parents[neighbor] = current
                    heapq.heappush(pq, (tentative_dist, neighbor))
        
        # No path found
        self.get_logger().warn('No path found from start to goal!')
        return None
    
    def goal_callback(self, msg: PoseStamped):
        """
        Handle new goal pose.
        
        Plans a path from current robot position to the goal using Dijkstra's algorithm,
        then publishes the path as nav_msgs/Path.
        """
        self.get_logger().info(f'Received goal: ({msg.pose.position.x:.2f}, '
                               f'{msg.pose.position.y:.2f})')
        
        # Check if map is available
        if self.map_info is None:
            self.get_logger().warn('Map not available yet!')
            return
        
        # Check if current position is known
        if self.current_pose is None:
            self.get_logger().warn('Current pose not available yet!')
            return
        
        # Convert start and goal to grid coordinates
        start_world = self.current_pose
        goal_world = (msg.pose.position.x, msg.pose.position.y)
        
        start_cell = self.world_to_grid(start_world[0], start_world[1])
        goal_cell = self.world_to_grid(goal_world[0], goal_world[1])
        
        if start_cell is None:
            self.get_logger().error(f'Start position ({start_world[0]:.2f}, '
                                    f'{start_world[1]:.2f}) is outside map!')
            return
        
        if goal_cell is None:
            self.get_logger().error(f'Goal position ({goal_world[0]:.2f}, '
                                    f'{goal_world[1]:.2f}) is outside map!')
            return
        
        self.get_logger().info(f'Planning path from grid({start_cell.row}, {start_cell.col}) '
                               f'to grid({goal_cell.row}, {goal_cell.col})...')
        
        # Run Dijkstra's algorithm
        grid_path = self.dijkstra(start_cell, goal_cell)
        
        if grid_path is None:
            self.get_logger().error('Failed to find path!')
            # Publish empty path
            empty_path = Path()
            empty_path.header.stamp = self.get_clock().now().to_msg()
            empty_path.header.frame_id = 'map'
            self.path_pub.publish(empty_path)
            return
        
        # Convert grid path to world coordinates
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        
        for cell in grid_path:
            x, y = self.grid_to_world(cell)
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # No rotation
            path_msg.poses.append(pose)
        
        # Publish path
        self.path_pub.publish(path_msg)
        self.get_logger().info(f'Published path with {len(path_msg.poses)} waypoints')


def main(args=None):
    rclpy.init(args=args)
    node = DijkstraPlanner()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
