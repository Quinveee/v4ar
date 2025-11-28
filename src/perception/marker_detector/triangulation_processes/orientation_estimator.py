import math


def compute_yaw_from_apriltag(marker, marker_map):
    """Compute robot yaw from AprilTag pose (fallback method)."""
    # This is a simplified version - you may need to adjust based on your marker message structure
    if hasattr(marker, 'pose') and hasattr(marker.pose, 'orientation'):
        q = marker.pose.orientation
        # Convert quaternion to yaw
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                        1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return yaw
    else:
        # Fallback: assume robot faces the marker
        return 0.0


class OrientationEstimator:
    def __init__(self, marker_map, axis_markers=None):
        self.marker_map = marker_map
        self.axis_markers = axis_markers or {}

    def estimate_yaw(self, marker):
        """Estimate robot yaw from marker detection."""
        # Try axis-based snap first
        yaw_axis = self._axis_based(marker)
        if yaw_axis is not None:
            return yaw_axis

        # Fallback to tag orientation method
        return compute_yaw_from_apriltag(marker, self.marker_map)

    def _axis_based(self, marker):
        """Get yaw from axis markers if available."""
        tag_id = marker.id
        for axis_name, ids in self.axis_markers.items():
            if tag_id in ids:
                return {
                    'axis_0_deg': 0.0,
                    'axis_90_deg': math.pi/2,
                    'axis_180_deg': math.pi,
                    'axis_270_deg': -math.pi/2
                }.get(axis_name, None)
        return None
