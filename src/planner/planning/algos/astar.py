import numpy as np
from heapq import heappush, heappop
from .base_planner import Planner


class AStarPlanner(Planner):

    def h(self, a, b):
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


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

        open_set = []
        heappush(open_set, (0, (sx, sy)))

        came_from = {}
        g_score = np.full((W, H), np.inf)
        g_score[sx, sy] = 0

        f_score = np.full((W, H), np.inf)
        f_score[sx, sy] = self.h((sx, sy), (gx, gy))

        while open_set:
            _, current = heappop(open_set)
            x, y = current

            if current == (gx, gy):
                break

            for nx, ny in self.neighbors(occ_grid, x, y):
                tentative_g = g_score[x, y] + 1

                if tentative_g < g_score[nx, ny]:
                    came_from[(nx, ny)] = (x, y)
                    g_score[nx, ny] = tentative_g
                    f_score[nx, ny] = tentative_g + self.h((nx, ny), (gx, gy))
                    heappush(open_set, (f_score[nx, ny], (nx, ny)))

        # Reconstruct path
        path = []
        p = (gx, gy)

        while p != (sx, sy):
            path.append(p)
            p = came_from.get(p, (sx, sy))
            if p == (sx, sy):
                path.append(p)
                break

        return path[::-1]
