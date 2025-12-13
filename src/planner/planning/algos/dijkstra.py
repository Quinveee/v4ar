import numpy as np
from queue import PriorityQueue
from .base_planner import Planner


class DijkstraPlanner(Planner):

    def __init__(self, verbose: bool = False):
        # verbose will enable simple print debugging
        self.verbose = verbose

    def neighbors(self, grid, x, y):
        # grid is indexed [y, x] (H, W)
        H, W = grid.shape
        nbrs = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        valid = []

        for nx, ny in nbrs:
            if 0 <= nx < W and 0 <= ny < H:
                # treat 254 (or 255) as occupied in our costmap
                if int(grid[ny, nx]) < 254:
                    valid.append((nx, ny))
        return valid


    def plan(self, occ_grid, start, goal):
        # occ_grid may be an occupancy array (0,1) or a costmap (0..254)
        sx, sy = start
        gx, gy = goal

        H, W = occ_grid.shape

        # distance array indexed [y, x]
        dist = np.full((H, W), np.inf)
        parent = {}

        pq = PriorityQueue()
        pq.put((0.0, (sx, sy)))
        dist[sy, sx] = 0.0

        while not pq.empty():
            d, current = pq.get()
            x, y = current

            if (x, y) == (gx, gy):
                break

            # If we've already found a better path to this node, skip
            if d > dist[y, x]:
                continue

            for nx, ny in self.neighbors(occ_grid, x, y):
                # base step cost = 1. Add additional penalty proportional to cell cost
                cell_cost = float(occ_grid[ny, nx])
                # Treat binary occupancy (0/1) naturally; treat costmap values (0..254) as weights
                if cell_cost >= 254:
                    continue

                # Normalize cell_cost into a small penalty
                penalty = 0.0
                if cell_cost > 1.0:
                    penalty = (cell_cost / 254.0) * 10.0  # tunable

                new_cost = d + 1.0 + penalty

                if new_cost < dist[ny, nx]:
                    dist[ny, nx] = new_cost
                    parent[(nx, ny)] = (x, y)
                    pq.put((new_cost, (nx, ny)))

        # Reconstruct path
        path = []
        p = (gx, gy)

        # If goal was never reached, return empty path
        if np.isinf(dist[gy, gx]):
            return []

        while p != (sx, sy):
            path.append(p)
            p = parent.get(p)
            if p is None:
                # failed to reconstruct
                return []

        path.append((sx, sy))
        return path[::-1]