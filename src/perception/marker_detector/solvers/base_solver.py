import numpy as np
from abc import ABC, abstractmethod

class BaseTriangulationSolver(ABC):
    """Abstract base class for all triangulation solvers."""
    @abstractmethod
    def solve(self, detections: list[tuple[int, float]], marker_map: dict[int, tuple[float, float]]) -> np.ndarray:
        """Compute robot position from detections and known marker coordinates."""
        raise NotImplementedError
