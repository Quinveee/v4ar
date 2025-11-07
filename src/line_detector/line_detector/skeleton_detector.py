# skeleton_detector.py
import cv2
import numpy as np
import math

try:
    from .base_detector import BaseLineDetector
except Exception:
    class BaseLineDetector:  # fallback if base exists elsewhere
        def detect(self, image):
            raise NotImplementedError

class SkeletonLineDetector(BaseLineDetector):
    """
    Percentile-threshold → closing → skeleton → Hough → parallel grouping.
    Returns [(x1,y1,x2,y2), ...]
    """
    def __init__(self,
                 percentile=90.0,
                 hough_threshold=50,
                 min_line_length=80,
                 max_line_gap=20,
                 angle_tolerance_deg=5,
                 min_group_size=3):
        self.percentile = percentile
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.angle_tol = angle_tolerance_deg
        self.min_group = min_group_size

    # ---- public API ----
    def detect(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1) blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

        # 2) percentile threshold (top X% brightest)
        thr = np.percentile(blurred, self.percentile)
        _, bright = cv2.threshold(blurred, thr, 255, cv2.THRESH_BINARY)

        # 3) small closing (don’t grow blobs too much)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)

        # 4) skeletonize
        skeleton = self._thinning(closed)

        # 5) Hough on skeleton
        lines = cv2.HoughLinesP(
            skeleton,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )
        lines = [tuple(l[0]) for l in lines] if lines is not None else []

        # 6) keep largest parallel group (angle clustering)
        lines = self._filter_parallel(lines,
                                      angle_tolerance=self.angle_tol,
                                      min_group_size=self.min_group)
        return lines

    # ---- helpers ----
    def _thinning(self, img_bin):
        # Prefer ximgproc if available
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            return cv2.ximgproc.thinning(img_bin)
        # Fallback morphological skeleton
        skel = np.zeros_like(img_bin)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        work = img_bin.copy()
        while True:
            eroded = cv2.erode(work, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(work, temp)
            skel = cv2.bitwise_or(skel, temp)
            work = eroded
            if cv2.countNonZero(work) == 0:
                break
        return skel

    def _filter_parallel(self, lines, angle_tolerance=5, min_group_size=3):
        if not lines or len(lines) < min_group_size:
            return lines
        tol = angle_tolerance
        # compute angle in [0,180)
        items = []
        for (x1,y1,x2,y2) in lines:
            ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
            length = math.hypot(x2 - x1, y2 - y1)
            items.append((ang, length, (x1,y1,x2,y2)))

        # greedy grouping by angle
        groups = []
        for ang, length, L in items:
            placed = False
            for g in groups:
                if abs(ang - g['ref']) < tol:
                    g['lines'].append((length, L))
                    g['sumlen'] += length
                    placed = True
                    break
            if not placed:
                groups.append({'ref': ang, 'lines': [(length, L)], 'sumlen': length})

        if not groups:
            return lines

        # choose group with max total length (more stable than count)
        best = max(groups, key=lambda g: g['sumlen'])
        if len(best['lines']) < min_group_size:
            return lines
        best['lines'].sort(key=lambda t: -t[0])
        return [L for _, L in best['lines']]
