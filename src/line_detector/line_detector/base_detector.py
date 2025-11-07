import cv2
import numpy as np
from .base_detector import BaseLineDetector

class CannyLineDetector(BaseLineDetector):
    def detect(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=50, maxLineGap=10)
        return [tuple(line[0]) for line in lines] if lines is not None else []
