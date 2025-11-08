from .base_selector import BaseLineSelector
from line_msgs.msg import DetectedLine
import statistics


class BufferedLineSelector(BaseLineSelector):
    """
    Wrapper selector that adds temporal buffering to any line selector.
    Collects N frames of selected lines, then returns an aggregated line.
    Caches the last valid result and keeps returning it until a new one is computed.
    """
    
    def __init__(self, inner_selector: BaseLineSelector, frame_buffer_size: int, 
                 image_width: int = 640, use_median: bool = False):
        """
        Args:
            inner_selector: The base selector to wrap (e.g., ClosestLineSelector)
            frame_buffer_size: Number of frames to buffer before aggregating
            image_width: Width of the image (for offset calculation)
            use_median: If True, use median aggregation; otherwise use mean
        """
        self.inner_selector = inner_selector
        self.frame_buffer_size = frame_buffer_size
        self.image_width = image_width
        self.use_median = use_median
        
        self.line_history = []
        self.frame_count = 0
        self.last_valid_line = None  # Cache the last aggregated line
    
    def select_line(self, lines):
        """
        Buffer lines and return aggregated result every N frames.
        Between aggregations, return the cached last valid line.
        """
        # Use inner selector to pick best line from current frame
        selected = self.inner_selector.select_line(lines)
        
        if selected is None:
            # If no line selected, return cached line (or None if we don't have one yet)
            return self.last_valid_line
        
        # Add to history
        self.line_history.append(selected)
        self.frame_count += 1
        
        # Check if it's time to aggregate
        if self.frame_count >= self.frame_buffer_size:
            # Compute aggregated line
            if self.use_median:
                aggregated_line = self._compute_median_line()
            else:
                aggregated_line = self._compute_mean_line()
            
            # Cache the result
            self.last_valid_line = aggregated_line
            
            # Reset buffer
            self.line_history = []
            self.frame_count = 0
            
            return aggregated_line
        else:
            # Not ready to aggregate yet - return cached line
            return self.last_valid_line
    
    def _compute_mean_line(self):
        """Compute the mean of all buffered lines."""
        if not self.line_history:
            return None
        
        n = len(self.line_history)
        avg_x1 = sum(line.x1 for line in self.line_history) / n
        avg_y1 = sum(line.y1 for line in self.line_history) / n
        avg_x2 = sum(line.x2 for line in self.line_history) / n
        avg_y2 = sum(line.y2 for line in self.line_history) / n
        
        # Create new line with averaged coordinates
        mean_line = DetectedLine()
        mean_line.header = self.line_history[-1].header
        mean_line.x1 = int(avg_x1)
        mean_line.y1 = int(avg_y1)
        mean_line.x2 = int(avg_x2)
        mean_line.y2 = int(avg_y2)
        
        # Recalculate offset_x and angle
        import math
        line_center_x = (avg_x1 + avg_x2) / 2
        mean_line.offset_x = float(line_center_x - self.image_width / 2)
        dx, dy = avg_x2 - avg_x1, avg_y2 - avg_y1
        mean_line.angle = math.atan2(dy, dx)
        
        return mean_line
    
    def _compute_median_line(self):
        """Compute the median of all buffered lines."""
        if not self.line_history:
            return None
        
        # Get median of each coordinate
        x1_values = [line.x1 for line in self.line_history]
        y1_values = [line.y1 for line in self.line_history]
        x2_values = [line.x2 for line in self.line_history]
        y2_values = [line.y2 for line in self.line_history]
        
        median_x1 = statistics.median(x1_values)
        median_y1 = statistics.median(y1_values)
        median_x2 = statistics.median(x2_values)
        median_y2 = statistics.median(y2_values)
        
        # Create new line with median coordinates
        median_line = DetectedLine()
        median_line.header = self.line_history[-1].header
        median_line.x1 = int(median_x1)
        median_line.y1 = int(median_y1)
        median_line.x2 = int(median_x2)
        median_line.y2 = int(median_y2)
        
        # Recalculate offset_x and angle
        import math
        line_center_x = (median_x1 + median_x2) / 2
        median_line.offset_x = float(line_center_x - self.image_width / 2)
        dx, dy = median_x2 - median_x1, median_y2 - median_y1
        median_line.angle = math.atan2(dy, dx)
        
        return median_line
