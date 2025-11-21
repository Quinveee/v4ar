import numpy as np
from .base_solver import BaseTriangulationSolver

class WeightedLeastSquaresSolver(BaseTriangulationSolver):
    """Weights markers inversely proportional to their distance."""
    def __init__(self, weight_power=2.0):
        self.weight_power = weight_power

    def solve(self, detections, marker_map, *args, **kwargs):
        ref_id, ref_d = detections[0]
        x1, y1 = marker_map[ref_id]

        A, b, weights = [], [], []
        for (mid, d) in detections[1:]:
            x2, y2 = marker_map[mid]
            A.append([2 * (x2 - x1), 2 * (y2 - y1)])
            b.append((x2**2 - x1**2) + (y2**2 - y1**2) + (ref_d**2 - d**2))
            weight = (1.0 / max(d, 1e-3)) ** self.weight_power
            weights.append(weight)
            
        W = np.diag(weights)
        A, b = np.array(A), np.array(b)
        xy = np.linalg.inv(A.T @ W @ A) @ (A.T @ W @ b)
        return xy
