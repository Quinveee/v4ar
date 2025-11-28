#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    LogInfo,
    IncludeLaunchDescription,
    GroupAction,
    SetEnvironmentVariable,
    ExecuteProcess
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
import os


def generate_launch_description():

    # ------------------------------------------------------------
    # Launch Arguments (must be strings)
    # ------------------------------------------------------------
    use_gui_arg = DeclareLaunchArgument(
        "use_gui",
        default_value="false",
        description="Toggle GUI for detector"
    )
    use_tilt_arg = DeclareLaunchArgument(
        "use_tilt",
        default_value="false",
        description="Enable pan/tilt camera"
    )
    use_cameras_only_arg = DeclareLaunchArgument(
        "use_cameras_only",
        default_value="false",
        description="Run only cameras + detector"
    )
    use_markers_only_arg = DeclareLaunchArgument(
        "use_markers_only",
        default_value="false",
        description="Run only marker visualization"
    )
    use_localization_only_arg = DeclareLaunchArgument(
        "use_localization_only",
        default_value="false",
        description="Run only localization stack"
    )
    use_display_field_arg = DeclareLaunchArgument(
        "use_display_field",
        default_value="false",
        description="Display field visualization"
    )

    gui = LaunchConfiguration("use_gui")
    tilt = LaunchConfiguration("use_tilt")
    cameras_only = LaunchConfiguration("use_cameras_only")
    markers_only = LaunchConfiguration("use_markers_only")
    localization_only = LaunchConfiguration("use_localization_only")
    display_field = LaunchConfiguration("use_display_field")

    ld = LaunchDescription([
        use_gui_arg,
        use_tilt_arg,
        use_cameras_only_arg,
        use_markers_only_arg,
        use_localization_only_arg,
        use_display_field_arg
    ])

    # ------------------------------------------------------------
    # Set ROS_DOMAIN_ID
    ld.add_action(SetEnvironmentVariable('ROS_DOMAIN_ID', '7'))
    ld.add_action(LogInfo(msg="ROS_DOMAIN_ID = 7"))

    # # Kill any existing ugv_bringup processes
    # ld.add_action(ExecuteProcess(
    #     cmd="pkill -f 'ros2 run ugv_bringup ugv_bringup' || true",
    #     output='screen',
    #     shell=True
    # ))

    # # Start ugv_bringup
    # ld.add_action(ExecuteProcess(
    #     cmd=['ros2', 'run', 'ugv_bringup', 'ugv_bringup'],
    #     output='screen',
    #     on_exit=LogInfo(msg="ugv_bringup exited")
    # ))

    # # Kill any existing ugv_driver processes
    # ld.add_action(ExecuteProcess(
    #     cmd="pkill -f 'ros2 run ugv_bringup ugv_driver' || true",
    #     output='screen',
    #     shell=True
    # ))

    # # Start ugv_driver
    # ld.add_action(ExecuteProcess(
    #     cmd=['ros2', 'run', 'ugv_bringup', 'ugv_driver'],
    #     output='screen',
    #     on_exit=LogInfo(msg="ugv_driver exited")
    # ))

    # # Kill any existing camera.launch.py processes
    # ld.add_action(ExecuteProcess(
    #     cmd="pkill -f 'ros2 launch ugv_vision camera.launch.py' || true",
    #     output='screen',
    #     shell=True
    # ))

    # # Start camera.launch.py
    # ld.add_action(ExecuteProcess(
    #     cmd=['ros2', 'launch', 'ugv_vision', 'camera.launch.py'],
    #     output='screen',
    #     on_exit=LogInfo(msg="camera.launch.py exited")
    # ))

    # Environment variables (only when GUI enabled)
    # ------------------------------------------------------------
    ld.add_action(SetEnvironmentVariable(
        'DISPLAY', os.environ.get('DISPLAY', ':0'),
        condition=IfCondition(gui)
    ))
    ld.add_action(SetEnvironmentVariable(
        'WAYLAND_DISPLAY', os.environ.get('WAYLAND_DISPLAY', ''),
        condition=IfCondition(gui)
    ))
    ld.add_action(SetEnvironmentVariable(
        'QT_QPA_PLATFORM', 'xcb',
        condition=IfCondition(gui)
    ))

    # ------------------------------------------------------------
    # OAK Camera: optional include
    # ------------------------------------------------------------
    try:
        ugv_share = get_package_share_directory('ugv_vision')
        oak_launch = os.path.join(ugv_share, 'launch', 'oak_d_lite.launch.py')

        if os.path.exists(oak_launch):
            ld.add_action(LogInfo(msg=f"Including OAK launch: {oak_launch}"))
            ld.add_action(
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(oak_launch))
            )
        else:
            ld.add_action(
                LogInfo(msg="ugv_vision found but no oak_d_lite.launch.py"))
    except PackageNotFoundError:
        ld.add_action(
            LogInfo(msg="ugv_vision not found; skipping OAK include"))

    try:
        ugv_share = get_package_share_directory('ugv_bringup')
        lidar_launch = os.path.join(
            ugv_share, 'launch', 'bringup_lidar.launch.py')
        if os.path.exists(lidar_launch):
            ld.add_action(
                LogInfo(msg=f"Including LIDAR launch: {lidar_launch}"))
            ld.add_action(
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(lidar_launch))
            )
        else:
            ld.add_action(
                LogInfo(msg="ugv_bringup found but no bringup_lidar.launch.py"))
    except PackageNotFoundError:
        ld.add_action(
            LogInfo(msg="ugv_bringup not found; skipping LIDAR include"))

    # ------------------------------------------------------------
    # Nodes (shared definitions)
    # ------------------------------------------------------------
    image_tools_img = Node(
        package='image_tools',
        executable='img',
        name='image_tools_img',
        output='screen',
        arguments=['--custom_rect', '--topic',
                   '/image_raw', '--frame-rate', '5']
    )

    apriltag_vis = Node(
        package='perception',
        executable='apriltag_vis_node_2',
        name='apriltag_vis',
        output='screen',
        parameters=[{
            'enable_gui': gui,
            'enable_multiscale': True,
            'multiscale_scales': "0.7, 1.0, 1.4, 2.5"
        }]
    )

    triangulation_node = Node(
        package='perception',
        executable='triangulation_uni',
        name='triangulation',
        output='screen'
    )

    localization_node = Node(
        package='perception',
        executable='localization',
        name='localization',
        output='screen',
        parameters=[{'strategy_type': 'kalman'}]
    )

    oak_detector = Node(
        package='perception',
        executable='oak_apriltag',
        name='oak_apriltag',
        output='screen',
        parameters=[{'enable_gui': gui}]
    )

    field_vis = Node(
        package='perception',
        executable='field_visualization',
        name='field_visualization',
        output='screen'
    )

    camera_tilt_node = Node(
        package='control',
        executable='marker_gaze',
        name='pan_tilt',
        output='screen',
        parameters=[{'pan': 0.15, 'tilt': 0.35}]
    )

    # ------------------------------------------------------------
    # MODES (Mutually exclusive via conditions)
    # ------------------------------------------------------------

    # Cameras only
    ld.add_action(
        GroupAction(
            condition=IfCondition(cameras_only),
            actions=[
                image_tools_img,
                oak_detector,
            ]
        )
    )

    # Markers only
    ld.add_action(
        GroupAction(
            condition=IfCondition(markers_only),
            actions=[
                apriltag_vis
            ]
        )
    )

    # Localization only
    ld.add_action(
        GroupAction(
            condition=IfCondition(localization_only),
            actions=[
                triangulation_node,
                localization_node,
                GroupAction(
                    condition=IfCondition(display_field),
                    actions=[field_vis]
                ),
            ]
        )
    )

    # ------------------------------------------------------------
    # Default full pipeline (when none of the "only" flags are true)
    # ------------------------------------------------------------
    ld.add_action(
        GroupAction(
            condition=UnlessCondition(cameras_only),
            actions=[
                GroupAction(
                    condition=UnlessCondition(markers_only),
                    actions=[
                        GroupAction(
                            condition=UnlessCondition(localization_only),
                            actions=[
                                image_tools_img,
                                apriltag_vis,
                                triangulation_node,
                                localization_node,
                                oak_detector,
                            ]
                        ),
                        GroupAction(
                            condition=IfCondition(display_field),
                            actions=[field_vis]
                        ),
                        GroupAction(
                            condition=IfCondition(tilt),
                            actions=[camera_tilt_node]
                        )
                    ]
                )
            ]
        )
    )

    return ld
