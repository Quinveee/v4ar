#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from perception_msgs.msg import ObjectPoseArray
import numpy as np
import networkx as nx
import time

# ==============================================================
# Configuration Constants
# ==============================================================
FIELD_X = 6.0       # Field width (m)
FIELD_Y = 9.0       # Field height (m)
GRID_RES = 0.3      # Grid cell size (m)

NX = int(FIELD_X / GRID_RES)
NY = int(FIELD_Y / GRID_RES)


# ==============================================================
# Utility Functions
# ==============================================================
def world_to_grid(x: float, y: float):
    """Convert world coordinates (m) to grid indices (i, j)."""
    i = int(np.clip(x / GRID_RES, 0, NX - 1))
    j = int(np.clip(y / GRID_RES, 0, NY - 1))
    return i, j


def grid_to_world(i: int, j: int):
    """Convert grid indices (i, j) to world coordinates (center of cell)."""
    x = i * GRID_RES + GRID_RES / 2.0
    y = j * GRID_RES + GRID_RES / 2.0
    return x, y


# ==============================================================
# Path Planner Node
# ==============================================================
class PathPlannerNode(Node):
    def __init__(self):
        super().__init__("path_planner_node")

        # ---------------- Parameters ----------------
        self.declare_parameter("waypoint_tolerance", 0.1)
        self.declare_parameter("localization_pause", 1.5)
        self.declare_parameter("replan_interval", 2.0)
        self.declare_parameter("replan_on_local_goal", True)

        self.waypoint_tolerance = self.get_parameter("waypoint_tolerance").value
        self.localization_pause = self.get_parameter("localization_pause").value
        self.replan_interval = self.get_parameter("replan_interval").value
        self.replan_on_local_goal = self.get_parameter("replan_on_local_goal").value

        # ---------------- ROS I/O ----------------
        self.create_subscription(PoseStamped, "/robot_pose", self.pose_callback, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self.goal_callback, 10)
        self.create_subscription(ObjectPoseArray, "/detected_rovers", self.obstacle_callback, 10)

        self.local_goal_pub = self.create_publisher(PoseStamped, "/local_goal", 10)
        self.goal_reached_pub = self.create_publisher(Bool, "/goal_reached", 10)

        # ---------------- Internal State ----------------
        self.robot_pose = None
        self.goal_pose = None
        self.obstacles = set()
        self.path = []
        self.current_idx = 0
        self.state = "IDLE"  # IDLE, NAVIGATING, LOCALIZING, DONE
        self.localization_start_time = None
        self.last_plan_time = 0.0

        # ---------------- Main Timer ----------------
        self.create_timer(0.05, self.update_loop)  # 20 Hz

        self.get_logger().info(
            f"PathPlannerNode initialized | Grid: {NX}x{NY} | Resolution: {GRID_RES:.2f} m"
        )

    # ------------------------------------------------------------
    # ROS Callbacks
    # ------------------------------------------------------------
    def pose_callback(self, msg: PoseStamped):
        self.robot_pose = msg.pose

    def goal_callback(self, msg: PoseStamped):
        self.goal_pose = msg.pose
        self.get_logger().info("🎯 New global goal received — computing path.")
        self.compute_path()

    def obstacle_callback(self, msg: ObjectPoseArray):
        """Update obstacles based on detected rovers."""
        new_obstacles = set()
        for obj in msg.rovers:
            x = obj.pose.position.x
            y = obj.pose.position.y
            i, j = world_to_grid(x, y)
            new_obstacles.add((i, j))
        self.obstacles = new_obstacles

    # ------------------------------------------------------------
    # Path Planning
    # ------------------------------------------------------------
    def compute_path(self):
        if self.robot_pose is None or self.goal_pose is None:
            self.get_logger().warn("Cannot compute path — pose or goal missing.")
            return

        start = world_to_grid(self.robot_pose.position.x, self.robot_pose.position.y)
        goal = world_to_grid(self.goal_pose.position.x, self.goal_pose.position.y)

        # Build grid graph
        G = nx.grid_2d_graph(NX, NY)

        # Remove obstacle nodes
        for o in self.obstacles:
            if o in G:
                G.remove_node(o)

        try:
            path = nx.shortest_path(G, start, goal)
            self.path = path
            self.current_idx = 0
            self.state = "NAVIGATING"
            self.get_logger().info(
                f"🧭 New path planned with {len(path)} waypoints | Start: {start} | Goal: {goal}"
            )
        except nx.NetworkXNoPath:
            self.get_logger().warn("🚫 No path found to goal.")
            self.path = []
            self.state = "IDLE"

    # ------------------------------------------------------------
    # Main Control Loop
    # ------------------------------------------------------------
    def update_loop(self):
        if self.robot_pose is None or self.goal_pose is None:
            return

        now = time.time()

        # Periodic replanning
        if now - self.last_plan_time > self.replan_interval and self.state not in ("LOCALIZING", "DONE"):
            self.compute_path()
            self.last_plan_time = now

        if self.state == "IDLE" or not self.path:
            return

        if self.state == "NAVIGATING":
            self.navigate_step()

        elif self.state == "LOCALIZING":
            if time.time() - self.localization_start_time > self.localization_pause:
                self.advance_waypoint()

    # ------------------------------------------------------------
    # Navigation Step
    # ------------------------------------------------------------
    def navigate_step(self):
        """Publish next local goal; check if robot entered the next bin."""
        if self.current_idx >= len(self.path):
            self.finish_goal()
            return

        target_i, target_j = self.path[self.current_idx]
        x_goal, y_goal = grid_to_world(target_i, target_j)
        robot_x, robot_y = self.robot_pose.position.x, self.robot_pose.position.y
        robot_i, robot_j = world_to_grid(robot_x, robot_y)

        # Log current robot position and local goal
        self.get_logger().info(
            f"📍 Robot: world=({robot_x:.2f}, {robot_y:.2f}) grid=({robot_i}, {robot_j}) | "
            f"Target: world=({x_goal:.2f}, {y_goal:.2f}) grid=({target_i}, {target_j})"
        )

        # Check if robot entered current target bin
        if (robot_i, robot_j) == (target_i, target_j):
            self.get_logger().info(
                f"✅ Entered local goal cell ({target_i}, {target_j}) → switching to LOCALIZING."
            )
            self.state = "LOCALIZING"
            self.localization_start_time = time.time()
            return

        # Publish local goal
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "world"
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = float(x_goal)
        goal_msg.pose.position.y = float(y_goal)
        goal_msg.pose.orientation.w = 1.0
        self.local_goal_pub.publish(goal_msg)

    # ------------------------------------------------------------
    # Waypoint progression
    # ------------------------------------------------------------
    def advance_waypoint(self):
        """Move to the next waypoint or finish the plan."""
        self.current_idx += 1
        if self.current_idx >= len(self.path):
            self.finish_goal()
            return

        next_i, next_j = self.path[self.current_idx]
        x_next, y_next = grid_to_world(next_i, next_j)

        if self.replan_on_local_goal:
            self.get_logger().info("♻️ Replanning after localization refinement.")
            self.compute_path()
        else:
            self.get_logger().info(
                f"➡️ Proceeding to next waypoint: grid=({next_i}, {next_j}) world=({x_next:.2f}, {y_next:.2f})"
            )

        self.state = "NAVIGATING"

    # ------------------------------------------------------------
    # Goal Completion
    # ------------------------------------------------------------
    def finish_goal(self):
        """Publish goal reached event."""
        self.state = "DONE"
        self.path = []
        msg = Bool()
        msg.data = True
        self.goal_reached_pub.publish(msg)
        self.get_logger().info("🏁 GLOBAL GOAL REACHED — navigation complete!")


# ==============================================================
# Main Entry Point
# ==============================================================
def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()