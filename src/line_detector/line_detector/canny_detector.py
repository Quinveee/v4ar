import cv2
import numpy as np
from .base_detector import BaseLineDetector

class CannyLineDetector(BaseLineDetector):
    """
    Simple line detector using Canny edge detection and Hough transform.
    Returns a list of (x1, y1, x2, y2) tuples.
    """

    def __init__(self, *args, **kwargs):
        pass

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

        # Optional intensity thresholding (helps under bright light)
        max_intensity = np.max(blurred)
        threshold = max_intensity * 0.8
        _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

        edges = cv2.Canny(thresh, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=50,
            maxLineGap=10
        )

        if lines is None:
            return []

        return [tuple(line[0]) for line in lines]
