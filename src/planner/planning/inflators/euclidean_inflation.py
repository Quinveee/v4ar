import numpy as np
from scipy.ndimage import distance_transform_edt
from .base_inflator import CostmapInflator

class EuclideanInflation(CostmapInflator):
    def __init__(self, inflation_radius=0.5, resolution=0.05):
        self.inflation_cells = int(inflation_radius / resolution)

    def inflate(self, occ_grid):
        # binary map: obstacles = True
        obstacles = (occ_grid == 1)

        # compute distance field
        dist = distance_transform_edt(~obstacles)

        costmap = np.zeros_like(dist, dtype=np.uint8)

        for y in range(dist.shape[0]):
            for x in range(dist.shape[1]):
                d = dist[y, x]
                if d == 0:
                    costmap[y, x] = 254        # occupied
                elif d <= self.inflation_cells:
                    # Nav2-inspired cost formula
                    costmap[y, x] = int(254 * (1 - d / self.inflation_cells))
                else:
                    costmap[y, x] = 0

        return costmap