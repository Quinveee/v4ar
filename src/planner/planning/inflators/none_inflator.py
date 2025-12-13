import numpy as np
from .base_inflator import CostmapInflator

class NoInflation(CostmapInflator):
    def inflate(self, occ_grid):
        # Convert kown free (0) → cost 0
        # occupied (1) → cost 254
        # unknown (-1) → cost 128 (Nav2 uses 255 but let's keep mid)
        costmap = np.zeros_like(occ_grid, dtype=np.uint8)
        costmap[occ_grid == 1] = 254
        costmap[occ_grid == -1] = 128
        return costmap
