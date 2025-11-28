#!/usr/bin/env python3
"""
Launch file to start both the global planner and the follow planner.

Nodes:
- planner: Global hierarchical planner computing paths across the field.
- follow_planner: Local controller that executes the planner's goals sequentially.

They communicate through /local_goal and /goal_reached.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # --- Launch arguments ---
    replan_interval = LaunchConfiguration("replan_interval", default="2.0")
    localization_pause = LaunchConfiguration("localization_pause", default="1.5")
    grid_resolution = LaunchConfiguration("grid_resolution", default="0.3")
    strategy_type = LaunchConfiguration("strategy_type", default="direct_goal")
    planner_name = LaunchConfiguration("planner_name", default="planner")
    follow_name = LaunchConfiguration("follow_planner", default="follow_planner")

    # --- Global planner node ---
    planner_node = Node(
        package="control",
        executable="planner",
        name="planner_node",
        output="screen",
        parameters=[
            {
                "replan_interval": replan_interval,
                "grid_resolution": grid_resolution,
            }
        ],
    )

    # --- Follow planner node ---
    follow_planner_node = Node(
        package="control",
        executable="follow_planner",
        name="follow_planner_node",
        output="screen",
        parameters=[
            {
                "strategy_type": strategy_type,
                "replan_interval": replan_interval,
                "localization_pause": localization_pause,
                "grid_resolution": grid_resolution,
            }
        ],
    )

    # --- Launch description ---
    return LaunchDescription([
        DeclareLaunchArgument(
            "strategy_type",
            default_value="direct_goal",
            description="Navigation strategy (direct_goal, dwa, potential_field, etc.)"
        ),
        DeclareLaunchArgument(
            "replan_interval",
            default_value="2.0",
            description="Interval between replanning steps in seconds"
        ),
        DeclareLaunchArgument(
            "localization_pause",
            default_value="1.5",
            description="Pause duration (seconds) for localization refinement at bins"
        ),
        DeclareLaunchArgument(
            "grid_resolution",
            default_value="0.3",
            description="Size of grid cells (meters)"
        ),
        planner_node,
        follow_planner_node,
    ])