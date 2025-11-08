import cv2
import numpy as np
from abc import ABC, abstractmethod


class BaseLineDetector(ABC):
    """
    Abstract base class for line detectors.
    Subclasses must implement the `detect` method.
    """
    @abstractmethod
    def detect(self, image: np.ndarray) -> list:
        """
        Detect lines in the given image.
        Returns a list of (x1, y1, x2, y2) tuples.
        """
        pass
