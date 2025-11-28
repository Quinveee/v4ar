class PositionEstimator:
    def __init__(self, marker_map, solver):
        self.marker_map = marker_map
        self.solver = solver

    def estimate_position(self, fused_detections):
        if hasattr(self.solver, 'solve_weighted'):
            return self.solver.solve_weighted(fused_detections, self.marker_map)
        else:
            detections = [(m_id, dist) for m_id, dist, _ in fused_detections]
            return self.solver.solve(detections, self.marker_map)
