import numpy as np
from .base_solver import BaseTriangulationSolver


class LeastSquaresSolver(BaseTriangulationSolver):
    """Exact two-marker triangulation solver (two-circle intersection)."""

    def solve(self, detections, marker_map, *args, **kwargs):
        # Must have at least two detections
        if len(detections) < 2:
            raise ValueError("Need at least two markers for triangulation")

        (id1, d1), (id2, d2) = detections[:2]
        x1, y1, _ = marker_map[id1]
        x2, y2, _ = marker_map[id2]

        dx, dy = x2 - x1, y2 - y1
        d = np.hypot(dx, dy)

        if d == 0:
            raise ValueError("Markers have identical positions")

        # Distance from marker 1 along the line between markers
        a = (d1**2 - d2**2 + d**2) / (2 * d)

        # Height from the line (triangle height)
        h_sq = d1**2 - a**2
        if h_sq < 0:
            raise ValueError("No real intersection (circles do not meet)")

        h = np.sqrt(h_sq)

        # Base point on the line between markers
        x3 = x1 + (a * dx) / d
        y3 = y1 + (a * dy) / d

        # Two intersection points
        x_int1 = x3 + (h * dy) / d
        y_int1 = y3 - (h * dx) / d
        x_int2 = x3 - (h * dy) / d
        y_int2 = y3 + (h * dx) / d

        # Prefer intersection where both x and y are positive
        p1_positive = (x_int1 > 0.0) and (y_int1 > 0.0)
        p2_positive = (x_int2 > 0.0) and (y_int2 > 0.0)

        if p1_positive and not p2_positive:
            chosen = (x_int1, y_int1)
        elif p2_positive and not p1_positive:
            chosen = (x_int2, y_int2)
        else:
            # fallback: choose the intersection with the higher y
            if y_int1 >= y_int2:
                chosen = (x_int1, y_int1)
            else:
                chosen = (x_int2, y_int2)

        x_r, y_r = chosen

        # --- Step 3: Estimate yaw based on marker geometry ---
        # In world frame: direction from marker 1 to marker 2
        world_angle = np.arctan2(y2 - y1, x2 - x1)
        # In robot frame: direction from robot to marker 2
        robot_angle = np.arctan2(y2 - y_r, x2 - x_r)
        # The yaw difference between world and robot geometry
        yaw = world_angle - robot_angle
        yaw = np.arctan2(np.sin(yaw), np.cos(yaw))  # normalize

        return np.array([x_r, y_r, yaw])
