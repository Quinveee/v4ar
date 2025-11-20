import numpy as np
from .base_solver import BaseTriangulationSolver

class LeastSquaresSolver(BaseTriangulationSolver):
    def solve(self, detections, marker_map):
        ref_id, ref_d = detections[0]
        x1, y1 = marker_map[ref_id]

        A, b = [], []
        for (mid, d) in detections[1:]:
            x2, y2 = marker_map[mid]
            A.append([2 * (x2 - x1), 2 * (y2 - y1)])
            b.append((x2**2 - x1**2) + (y2**2 - y1**2) + (ref_d**2 - d**2))

        A, b = np.array(A), np.array(b)
        xy, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return xy
