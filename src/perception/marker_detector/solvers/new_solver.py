import numpy as np
from .base_solver import BaseTriangulationSolver


class LeastSquaresSolver(BaseTriangulationSolver):
    """
    Two-marker triangulation (circle intersection) + 
    bounded least-squares selection using all visible markers.

    Field is assumed to be a rectangle:
        x ∈ [x_min, x_max]
        y ∈ [y_min, y_max]
    """

    def __init__(self,
                 x_bounds=(0.0, 6.0),
                 y_bounds=(0.0, 9.0)):
        super().__init__()
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds

    def _in_bounds(self, x, y):
        return (self.x_min <= x <= self.x_max and
                self.y_min <= y <= self.y_max)

    def _residual_error(self, x, y, detections, marker_map, penalty_outside=1e6):
        """
        Compute sum of squared range residuals to all detected markers.
        If (x,y) is outside the field bounds, add a big penalty.
        """
        err = 0.0
        for mid, dist_meas in detections:
            mx, my, _ = marker_map[mid]
            dist_pred = np.hypot(x - mx, y - my)
            err += (dist_pred - dist_meas) ** 2

        if not self._in_bounds(x, y):
            err += penalty_outside

        return err

    def solve(self, detections, marker_map, *args, **kwargs):
        # Need at least two markers for circle intersection
        if len(detections) < 2:
            raise ValueError("Need at least two markers for triangulation")

        # Use the first two detections to define the two-circle intersection
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
            # numerical noise: treat tiny negatives as zero
            if h_sq > -1e-6:
                h_sq = 0.0
            else:
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

        # Evaluate both candidates against ALL detections and bounds
        cand1 = (x_int1, y_int1)
        cand2 = (x_int2, y_int2)

        err1 = self._residual_error(x_int1, y_int1, detections, marker_map)
        err2 = self._residual_error(x_int2, y_int2, detections, marker_map)

        # Choose candidate with smaller error
        if err1 < err2:
            chosen = cand1
        else:
            chosen = cand2

        # Optional: final safety clamp into bounds (if you really want that)
        x, y = chosen
        x = float(np.clip(x, self.x_min, self.x_max))
        y = float(np.clip(y, self.y_min, self.y_max))

        # --- Step 3: Estimate yaw based on marker geometry ---
        # In world frame: direction from marker 1 to marker 2
        world_angle = np.arctan2(y2 - y1, x2 - x1)
        # In robot frame: direction from robot to marker 2
        robot_angle = np.arctan2(y2 - y, x2 - x)
        # The yaw difference between world and robot geometry
        yaw = world_angle - robot_angle
        yaw = np.arctan2(np.sin(yaw), np.cos(yaw))  # normalize

        return np.array([x, y, yaw], dtype=float)
