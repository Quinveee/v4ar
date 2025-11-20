from .base_selector import BaseLineSelector
from line_msgs.msg import DetectedLine


class MeanLineSelector(BaseLineSelector):
    """
    Selects a line by computing the mean of all detected lines.
    Returns a new line with averaged x1, y1, x2, y2, offset_x, and angle.
    
    This approach is useful when you want to follow the "average" direction
    of multiple detected lines, reducing the impact of outliers and noise.
    """
    
    __NAME__ = "mean_line_selector"
    
    def __init__(self, *args, **kwargs):
        """Initialize the mean line selector."""
        pass
    
    def select_line(self, lines):
        """
        Compute the mean line from all detected lines.
        
        Args:
            lines: List of DetectedLine messages
            
        Returns:
            DetectedLine message representing the mean of all lines,
            or None if no lines are provided
        """
        if not lines:
            return None
        
        # Calculate mean of all line coordinates
        mean_x1 = sum(line.x1 for line in lines) / len(lines)
        mean_y1 = sum(line.y1 for line in lines) / len(lines)
        mean_x2 = sum(line.x2 for line in lines) / len(lines)
        mean_y2 = sum(line.y2 for line in lines) / len(lines)
        
        # Calculate mean offset_x and angle
        mean_offset_x = sum(line.offset_x for line in lines) / len(lines)
        mean_angle = sum(line.angle for line in lines) / len(lines)
        
        # Create a new DetectedLine with mean values
        mean_line = DetectedLine()
        
        # Use the header from the first line
        mean_line.header = lines[0].header
        
        # Set mean coordinates (convert to int for x1, y1, x2, y2)
        mean_line.x1 = int(mean_x1)
        mean_line.y1 = int(mean_y1)
        mean_line.x2 = int(mean_x2)
        mean_line.y2 = int(mean_y2)
        
        # Set mean offset and angle
        mean_line.offset_x = float(mean_offset_x)
        mean_line.angle = float(mean_angle)
        
        return mean_line

