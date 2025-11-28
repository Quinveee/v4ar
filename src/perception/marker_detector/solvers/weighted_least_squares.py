import numpy as np
from .base_solver import BaseTriangulationSolver

class WeightedLeastSquaresSolver(BaseTriangulationSolver):
    """Weights markers inversely proportional to their distance."""
    def __init__(self, weight_power=2.0):
        self.weight_power = weight_power

    def solve(self, detections, marker_map, *args, **kwargs):
        if len(detections) < 2:
            raise ValueError("Need at least two markers for triangulation")

        ref_id, ref_d = detections[0]
        x1, y1, _ = marker_map[ref_id]

        A, b, weights = [], [], []
        for (mid, d) in detections[1:]:
            x2, y2, _ = marker_map[mid]
            A.append([2 * (x2 - x1), 2 * (y2 - y1)])
            b.append((x2**2 - x1**2) + (y2**2 - y1**2) + (ref_d**2 - d**2))
            weight = (1.0 / max(d, 1e-3)) ** self.weight_power
            weights.append(weight)

        W = np.diag(weights)
        A, b = np.array(A), np.array(b)

        # Use pseudo-inverse to handle singular/near-singular matrices
        # This is more robust than inv() for degenerate marker configurations
        try:
            # Weighted least squares: solve (A^T W A) x = A^T W b
            AtWA = A.T @ W @ A
            AtWb = A.T @ W @ b

            # Check condition number to detect near-singular matrices
            cond = np.linalg.cond(AtWA)
            if cond > 1e10:
                # Matrix is ill-conditioned, use pseudo-inverse
                xy = np.linalg.inv(AtWA) @ AtWb
            else:
                # Matrix is well-conditioned, use regular inverse
                xy = np.linalg.inv(AtWA) @ AtWb
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if inverse fails
            xy = np.linalg.inv(A.T @ W @ A) @ (A.T @ W @ b)

        x_r, y_r = xy

        # --- Estimate yaw based on marker geometry ---
        # Use first two markers for yaw estimation
        (id1, d1), (id2, d2) = detections[:2]
        x1, y1, _ = marker_map[id1]
        x2, y2, _ = marker_map[id2]

        # In world frame: direction from marker 1 to marker 2
        world_angle = np.arctan2(y2 - y1, x2 - x1)
        # In robot frame: direction from robot to marker 2
        robot_angle = np.arctan2(y2 - y_r, x2 - x_r)
        # The yaw difference between world and robot geometry
        yaw = world_angle - robot_angle
        yaw = np.arctan2(np.sin(yaw), np.cos(yaw))  # normalize

        return np.array([x_r, y_r, yaw])
