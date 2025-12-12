import numpy as np
from scipy.ndimage import distance_transform_edt
from .base_inflator import CostmapInflator

class Nav2Inflation(CostmapInflator):
    def __init__(self, inflation_radius=0.5, cost_scaling=3.0, resolution=0.05):
        self.inflation_cells = int(inflation_radius / resolution)
        self.cost_scaling = cost_scaling

    def inflate(self, occ_grid):
        obstacles = (occ_grid == 1)
        dist = distance_transform_edt(~obstacles)

        costmap = np.zeros(dist.shape, dtype=np.uint8)

        for y in range(dist.shape[0]):
            for x in range(dist.shape[1]):
                d = dist[y, x]

                if d == 0:
                    costmap[y, x] = 254
                    continue

                if d <= self.inflation_cells:
                    # SAME FORMULA AS NAV2 InflationLayer
                    cost = 253 * np.exp(-self.cost_scaling * (d / self.inflation_cells))
                    costmap[y, x] = int(cost)
                else:
                    costmap[y, x] = 0

        return costmap
