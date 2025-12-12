import numpy as np
from queue import PriorityQueue
from .base_planner import Planner


class DijkstraPlanner(Planner):

    def neighbors(self, occ, x, y):
        H, W = occ.shape
        nbrs = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
        valid = []

        for nx, ny in nbrs:
            if 0 <= nx < W and 0 <= ny < H:
                if occ[ny, nx] == 0:
                    valid.append((nx, ny))
        return valid


    def plan(self, occ_grid, start, goal):

        sx, sy = start
        gx, gy = goal
        H, W = occ_grid.shape
        
        dist = np.full((W, H), np.inf)
        parent = {}

        pq = PriorityQueue()
        pq.put((0, (sx, sy)))
        dist[sx, sy] = 0

        while not pq.empty():
            d, current = pq.get()
            if current == (gx, gy):
                break

            x, y = current

            for nx, ny in self.neighbors(occ_grid, x, y):
                new_cost = d + 1
                if new_cost < dist[nx, ny]:
                    dist[nx, ny] = new_cost
                    parent[(nx, ny)] = (x, y)
                    pq.put((new_cost, (nx, ny)))

        # Reconstruct path
        path = []
        p = (gx, gy)

        while p != (sx, sy):
            path.append(p)
            p = parent.get(p, (sx, sy))
            if p == (sx, sy):
                path.append(p)
                break

        return path[::-1]