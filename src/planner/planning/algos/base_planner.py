class Planner:
    """
    Strategy interface for path planning algorithms.
    """

    def plan(self, occ_grid, start, goal):
        """
        Compute a path from start to goal.
        
        Parameters:
            occ_grid: 2D np.array of 0 (free) and 1 (obstacle)
            start: (gx, gy) grid coordinates
            goal: (gx, gy) grid coordinates

        Returns:
            List of (gx, gy) grid coordinates representing the path
        """
        raise NotImplementedError("Planner must implement plan()")
