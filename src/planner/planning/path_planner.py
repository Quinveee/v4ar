#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, MapMetaData
import time

import numpy as np
import yaml
from PIL import Image
import os

# -------------------------
# Import Planners
# -------------------------
from .algos import DijkstraPlanner, AStarPlanner, Planner

# -------------------------
# Import Inflators
# -------------------------
from .inflators import *


class PathPlannerNode(Node):

    def __init__(self):
        super().__init__("path_planner")

        # --------------------------
        # Declare parameters
        # --------------------------
        self.declare_parameter("map_yaml", "")
        self.declare_parameter("planner", "dijkstra")
        self.declare_parameter("inflation", "none")
        self.declare_parameter("inflation_radius", 0.5)
        self.declare_parameter("inflation_scaling", 3.0)
        self.declare_parameter("use_nav2_costmap", False)
        self.declare_parameter("costmap_topic", "/local_costmap/costmap")

        yaml_path = self.get_parameter("map_yaml").value
        planner_name = self.get_parameter("planner").value
        inflation_name = self.get_parameter("inflation").value
        inflation_radius = self.get_parameter("inflation_radius").value
        inflation_scaling = self.get_parameter("inflation_scaling").value
        use_nav2_costmap = bool(self.get_parameter("use_nav2_costmap").value)
        costmap_topic = str(self.get_parameter("costmap_topic").value)

        # --------------------------
        # Choose planner strategy
        # --------------------------
        if planner_name == "astar":
            self.planner: Planner = AStarPlanner()
        else:
            self.planner: Planner = DijkstraPlanner()

        self.get_logger().info(f"Planner selected: {planner_name}")

        # --------------------------
        # Load map (same as Nav2)
        # --------------------------
        self.occ_grid = self.load_map_from_yaml(yaml_path)
        self.H, self.W = self.occ_grid.shape

        # --------------------------
        # Choose inflation strategy (only used when not subscribing to Nav2 costmap)
        # --------------------------
        if inflation_name == "euclidean":
            self.inflator = EuclideanInflation(
                inflation_radius=inflation_radius,
                resolution=self.resolution
            )
        elif inflation_name == "nav2":
            self.inflator = Nav2Inflation(
                inflation_radius=inflation_radius,
                cost_scaling=inflation_scaling,
                resolution=self.resolution
            )
        elif inflation_name == "none":
            self.inflator = NoInflation()
        else:
            self.inflator = NoInflation()

        self.get_logger().info(f"Inflation strategy: {inflation_name}")

        # If requested, subscribe to an external Nav2 costmap topic instead of using our inflated one.
        self.use_nav2_costmap = use_nav2_costmap
        self.costmap_topic = costmap_topic

        if self.use_nav2_costmap:
            # subscribe to OccupancyGrid topic published by Nav2 (or other) costmap
            self.get_logger().info(f"Subscribing to Nav2 costmap topic: {self.costmap_topic}")
            self.create_subscription(
                OccupancyGrid,
                self.costmap_topic,
                self.nav2_costmap_callback,
                10,
            )
            # create an initial costmap fallback from our inflator until we receive external costmap
            self.costmap = self.inflator.inflate(self.occ_grid)
        else:
            # --------------------------
            # Inflate costmap (local fallback)
            # --------------------------
            self.costmap = self.inflator.inflate(self.occ_grid)

        # --- Debugging: report costmap stats and save image ---
        try:
            unique_vals = np.unique(self.costmap)
            self.get_logger().info(f"Costmap values (unique): {unique_vals}")
            self.get_logger().info(f"Costmap min/max: {int(self.costmap.min())}/{int(self.costmap.max())}")

            # histogram of values (coarse bins)
            vals, counts = np.unique(self.costmap, return_counts=True)
            top = list(zip(vals.tolist(), counts.tolist()))[:30]
            self.get_logger().info(f"Costmap sample histogram (value,count) first 30: {top}")

            # save visualization for quick inspection
            try:
                from PIL import Image
                img = Image.fromarray(self.costmap)
                out_path = "planner_costmap.png"
                img.save(out_path)
                self.get_logger().info(f"Saved costmap image to {out_path}")
            except Exception as e:
                self.get_logger().warn(f"Failed to save costmap image: {e}")
        except Exception as e:
            # keep node working even if debug fails
            self.get_logger().warn(f"Costmap debug failed: {e}")

        # --------------------------
        # Start & goal from RViz
        # --------------------------
        self.start_world = None
        self.goal_world = None

        # --------------------------
        # Publisher
        # --------------------------
        self.path_pub = self.create_publisher(Path, "planned_path", 10)

        # --------------------------
        # Subscribers (RViz inputs)
        # --------------------------
        self.create_subscription(
            PoseWithCovarianceStamped,
            "/initialpose",
            self.initialpose_callback,
            10,
        )

        self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goalpose_callback,
            10,
        )

        self.map_pub = self.create_publisher(OccupancyGrid, "planner_map", 10)
        self.publish_map()

        self.get_logger().info(str(np.unique(self.costmap)))
        self.get_logger().info("Ready. Click '2D Pose Estimate' and '2D Goal Pose' in RViz.")


    # ======================================================================
    # Load Nav2-style YAML map
    # ======================================================================
    def load_map_from_yaml(self, yaml_path):
        if yaml_path == "":
            raise RuntimeError("map_yaml parameter is required!")

        with open(yaml_path, "r") as f:
            info = yaml.safe_load(f)

        image_path = info["image"]
        resolution = float(info["resolution"])
        origin = info["origin"]

        negate = int(info.get("negate", 0))
        occ_thresh = float(info.get("occupied_thresh", 0.65))
        free_thresh = float(info.get("free_thresh", 0.25))

        # store metadata
        self.resolution = resolution
        self.origin_x = origin[0]
        self.origin_y = origin[1]

        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(yaml_path), image_path)

        img = Image.open(image_path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0

        if negate == 1:
            arr = 1.0 - arr

        occ = np.full(arr.shape, -1, dtype=np.int8)
        occ[arr >= occ_thresh] = 1
        occ[arr <= free_thresh] = 0

        self.get_logger().info(
            f"Loaded map {image_path}, size={occ.shape}, res={resolution}"
        )

        return occ

    def publish_map(self):
        msg = OccupancyGrid()

        # --- Header ---
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()

        # --- Metadata ---
        info = MapMetaData()
        info.resolution = self.resolution
        info.width = self.W
        info.height = self.H
        info.origin.position.x = self.origin_x
        info.origin.position.y = self.origin_y
        info.origin.position.z = 0.0
        info.origin.orientation.w = 1.0

        msg.info = info

        # --- Convert occupancy grid from [-1,0,1] to conventional [-1,0,100] ---
        flat = np.zeros(self.occ_grid.size, dtype=np.int8)

        # Unknown
        flat[self.occ_grid.flatten() < 0] = -1
        # Free
        flat[self.occ_grid.flatten() == 0] = 0
        # Occupied
        flat[self.occ_grid.flatten() == 1] = 100

        msg.data = flat.tolist()

        self.map_pub.publish(msg)
        self.get_logger().info("Published planner_map.")


    # ------------------------------------------------------------------
    # Callback to receive external Nav2 costmap (OccupancyGrid)
    # ------------------------------------------------------------------
    def nav2_costmap_callback(self, msg: OccupancyGrid):
        try:
            # msg.data is a flat list of size height*width in row-major order
            arr = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))

            # Convert occupancy values (0..100, -1) to planner internal costmap (0..254)
            # Map 0 -> 0, 100 -> 254, unknown(-1) -> 127
            conv = np.zeros_like(arr, dtype=np.uint8)
            # known free
            conv[arr == 0] = 0
            # known occupied
            conv[arr == 100] = 254
            # unknown
            conv[arr < 0] = 127

            # For any intermediate values (rare), scale linearly
            mask_mid = (arr > 0) & (arr < 100)
            conv[mask_mid] = np.clip((arr[mask_mid].astype(np.float32) / 100.0) * 254.0, 0, 254).astype(np.uint8)

            # assign to current costmap (note: arr shape is height x width -> [y,x])
            self.costmap = conv
            self.get_logger().info(f"Received Nav2 costmap ({msg.info.width}x{msg.info.height}), min/max: {int(self.costmap.min())}/{int(self.costmap.max())}")
        except Exception as e:
            self.get_logger().warn(f"Failed to process Nav2 costmap: {e}")


    # ======================================================================
    # Coordinate Conversions
    # ======================================================================
    def world_to_grid(self, x, y):
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        x = gx * self.resolution + self.origin_x
        y = gy * self.resolution + self.origin_y
        return x, y


    # ======================================================================
    # RViz callbacks
    # ======================================================================
    def initialpose_callback(self, msg: PoseWithCovarianceStamped):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.start_world = (x, y)
        self.get_logger().info(f"Start pose set to {self.start_world}")
        self.try_compute_path()

    def goalpose_callback(self, msg: PoseStamped):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.goal_world = (x, y)
        self.get_logger().info(f"Goal pose set to {self.goal_world}")
        self.try_compute_path()


    # ======================================================================
    # Run planner once both poses available
    # ======================================================================
    def try_compute_path(self):
        if self.start_world is None or self.goal_world is None:
            return

        start = self.world_to_grid(*self.start_world)
        goal = self.world_to_grid(*self.goal_world)

        self.get_logger().info("Planning path...")

        path = self.planner.plan(self.costmap, start, goal)

        if path is None or len(path) == 0:
            self.get_logger().warn("No path found.")
            return

        # build RViz message
        msg = Path()
        msg.header.frame_id = "map"

        for gx, gy in path:
            x, y = self.grid_to_world(gx, gy)
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.path_pub.publish(msg)
        self.get_logger().info(f"Published path with {len(msg.poses)} poses.")


def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()