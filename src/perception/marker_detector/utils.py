import math
import numpy as np
from scipy.spatial.transform import Rotation as R


def compute_yaw_from_apriltag(tag_msg, marker_map):
    """
    Compute robot yaw based on:
    - tag orientation in camera frame (tag_msg.pose.orientation)
    - known tag orientation in world frame (marker_map[id][2])
    """
    # 1. Rotation matrix from tag → camera
    q = [
        tag_msg.pose.orientation.x,
        tag_msg.pose.orientation.y,
        tag_msg.pose.orientation.z,
        tag_msg.pose.orientation.w
    ]
    R_ct = R.from_quat(q).as_matrix()

    # 2. Tag normal direction in camera frame (tag's +Z axis)
    tag_normal_cam = R_ct @ np.array([0, 0, 1.0])

    # 3. Camera → tag direction (reverse normal)
    cam_to_tag = -tag_normal_cam

    # 4. Bearing angle to tag in camera/robot frame
    cam_yaw_to_tag = math.atan2(cam_to_tag[1], cam_to_tag[0])

    # 5. Tag yaw in world frame (from YAML config)
    _, _, tag_yaw_world = marker_map[tag_msg.id]

    # 6. Robot yaw = difference
    yaw_robot = tag_yaw_world - cam_yaw_to_tag

    # Normalize
    return math.atan2(math.sin(yaw_robot), math.cos(yaw_robot))


def compute_average_yaw_from_markers(robot_x, robot_y, visible_markers, marker_map,
                                     oak_weight=1.0, mono_weight=1.0,
                                     oak_ids=None, mono_ids=None):
    """
    Estimate yaw as the weighted average direction from robot to all visible markers.

    Args:
        robot_x, robot_y: Estimated robot position
        visible_markers: List of marker objects with .id attribute, or list of (marker_id, distance, weight) tuples
        marker_map: Dictionary mapping marker_id -> (x, y, yaw)
        oak_weight: Weight multiplier for OAK camera markers (0 to ignore)
        mono_weight: Weight multiplier for mono camera markers (0 to ignore)
        oak_ids: Set of marker IDs detected by OAK camera (optional)
        mono_ids: Set of marker IDs detected by mono camera (optional)

    Returns:
        Average yaw in radians, or None if insufficient markers
    """
    if len(visible_markers) < 2:
        return None

    sin_sum = 0.0
    cos_sum = 0.0
    total_weight = 0.0

    for m in visible_markers:
        # Handle both marker objects and tuples
        if hasattr(m, 'id'):
            marker_id = m.id
            distance = getattr(m, 'distance', 1.0)
            # Determine source weight
            if oak_ids is not None and marker_id in oak_ids:
                source_weight = oak_weight
            elif mono_ids is not None and marker_id in mono_ids:
                source_weight = mono_weight
            else:
                source_weight = 1.0  # Default weight
        else:
            # Tuple format: (marker_id, distance, weight)
            marker_id, distance, source_weight = m

        if marker_id not in marker_map:
            continue

        # Skip if source weight is zero
        if source_weight <= 0:
            continue

        mx, my = marker_map[marker_id][:2]
        angle_to_marker = math.atan2(my - robot_y, mx - robot_x)
        # Robot faces opposite direction of the markers it sees
        robot_facing = angle_to_marker + math.pi

        # Weight by inverse distance and source weight
        weight = source_weight / max(distance, 0.1)

        sin_sum += weight * math.sin(robot_facing)
        cos_sum += weight * math.cos(robot_facing)
        total_weight += weight

    if total_weight <= 0:
        return None

    # Circular mean
    avg_yaw = math.atan2(sin_sum / total_weight, cos_sum / total_weight)

    # Normalize to [-pi, pi] and adjust
    avg_yaw = math.atan2(math.sin(avg_yaw), math.cos(avg_yaw))
    return avg_yaw - math.pi
