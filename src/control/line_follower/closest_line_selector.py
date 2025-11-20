from .base_selector import BaseLineSelector
from line_msgs.msg import DetectedLines, DetectedLine

class ClosestLineSelector(BaseLineSelector):
    __NAME__ = "closest_line_selector"
    def __init__(self, *args, **kwargs):
        pass

    def select_line(self, lines):
        """
        Selects which line to follow.
        Currently picks the one with smallest |offset_x| (closest to center).
        """
        best_line = None
        min_offset = float('inf')

        for line in lines:
            offset = abs(line.offset_x)
            if offset < min_offset:
                min_offset = offset
                best_line = line

        return best_line