class CostmapInflator:
    def inflate(self, occ_grid):
        """
        Input:  occ_grid (H×W) with values {-1, 0, 1}
        Output: costmap (H×W) with cost values [0..254]

        Must be implemented in subclasses.
        """
        raise NotImplementedError
