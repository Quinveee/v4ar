import cv2
import numpy as np
from .base_detector import BaseLineDetector

class GradientLineDetector(BaseLineDetector):
    """
    Line detector using Sobel gradient magnitude + Hough transform.
    Useful when edges are well defined but not high-contrast enough for Canny.
    Returns a list of (x1, y1, x2, y2) tuples.
    """

    def __init__(self, *args, **kwargs):
        pass

    def detect(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

        # Compute gradients
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Normalize and threshold to get binary edges
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        _, edges = cv2.threshold(gradient_magnitude.astype(np.uint8), 50, 255, cv2.THRESH_BINARY)

        # Detect lines with Hough transform
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
