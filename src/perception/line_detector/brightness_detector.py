import cv2
import numpy as np
from .base_detector import BaseLineDetector

class BrightnessLineDetector(BaseLineDetector):
    """
    Brightness-based line detector designed for ceiling light tracking.
    Uses overexposure filtering and Hough transform on bright regions.
    Returns a list of (x1, y1, x2, y2) tuples.
    """

    def __init__(self, *args, **kwargs):
        pass

    def detect(self, image):
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Handle overexposed regions
        overexposed_ratio = np.sum(blurred >= 255) / (height * width)
        if overexposed_ratio > 0.30:
            overexposure_threshold = 250
            mask = blurred < overexposure_threshold
            blurred[~mask] = 0

        # Threshold bright regions
        brightness_threshold = np.mean(blurred) + 0.5 * np.std(blurred)
        _, bright_mask = cv2.threshold(blurred, brightness_threshold, 255, cv2.THRESH_BINARY)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)

        # Detect bright lines
        lines = cv2.HoughLinesP(
            bright_mask,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=30,
            maxLineGap=20
        )

        if lines is None:
            return []

        # Optionally filter for vertical lines (common in ceiling fixtures)
        vertical_lines = []
        for (x1, y1, x2, y2) in lines[:, 0]:
            angle_deg = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            if angle_deg > 60:  # mostly vertical
                vertical_lines.append((x1, y1, x2, y2))

        return vertical_lines
