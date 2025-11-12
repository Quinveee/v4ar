import cv2
import numpy as np
from .base_detector import BaseLineDetector

class CustomLineDetector(BaseLineDetector):
    """
    Line detector using Sobel gradient magnitude + Hough transform.
    Useful when edges are well defined but not high-contrast enough for Canny.
    Returns a list of (x1, y1, x2, y2) tuples.
    """

    def __init__(self, vignette=False, *args, **kwargs):
        self.vignette = vignette

    def detect(self, image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(image_gray, (5, 5), 1.5)

        h, w = blurred.shape
        center_x, center_y = w / 2, h / 2

        # --- Create a Gaussian vignette mask ---
        if self.vignette:
            # Generate coordinate grids
            x = np.linspace(0, w - 1, w)
            y = np.linspace(0, h - 1, h)
            X, Y = np.meshgrid(x, y)

            # Compute distance from center for each pixel
            sigma_x = w / 3   # controls how quickly it fades horizontally
            sigma_y = h / 3   # controls how quickly it fades vertically
            mask = np.exp(-(((X - center_x) ** 2) / (2 * sigma_x ** 2) +
                            ((Y - center_y) ** 2) / (2 * sigma_y ** 2)))

            # Normalize mask to 0�1 range
            mask = (mask - mask.min()) / (mask.max() - mask.min())

            # Apply mask (center bright, edges darker)
            weighted = blurred * mask
        else:
            mask = np.ones_like(blurred)
            weighted = blurred

        # Continue with your existing thresholding
        max_intensity = np.max(weighted)
        threshold = max_intensity * 0.9
        _, threshold_image = cv2.threshold(
            weighted, threshold, 255, cv2.THRESH_BINARY)

        # Derivative of Gaussian for edges
        dog_filter_x = cv2.Sobel(threshold_image, cv2.CV_64F, 1, 0, ksize=5)
        dog_filter_y = cv2.Sobel(threshold_image, cv2.CV_64F, 0, 1, ksize=5)
        dog_magnitude = cv2.magnitude(dog_filter_x, dog_filter_y)
        _, dog_edges = cv2.threshold(dog_magnitude, 50, 255, cv2.THRESH_BINARY)
        dog_edges = dog_edges.astype('uint8')

        lines = cv2.HoughLinesP(
            dog_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=150,
            minLineLength=100,
            maxLineGap=8
        )
        if lines is None:
            return []

        return [tuple(line[0]) for line in lines]
