from .base_selector import BaseLineSelector
from line_msgs.msg import DetectedLine
import math


class TrackingLineSelector(BaseLineSelector):
    """
    Sophisticated line selector that tracks a single line over time using temporal continuity.
    
    This selector "locks onto" one line and consistently tracks it even when:
    - The line temporarily disappears
    - Other lines have higher confidence
    - There's noise in the detections
    
    It works by:
    1. Predicting where the tracked line should be based on recent history
    2. Matching detected lines to the prediction using position and angle similarity
    3. Only switching to a different line if the current one is lost for multiple frames
    """
    
    def __init__(self, 
                 max_position_jump=100.0,    # Max pixels the line can jump between frames
                 max_angle_change=0.3,        # Max radians (~17°) angle can change between frames
                 history_size=5,              # Number of frames to keep in history for prediction
                 lost_frames_threshold=10,    # Frames to wait before considering line truly lost
                 image_width=640):            # Image width for offset calculation
        """
        Args:
            max_position_jump: Maximum distance (pixels) a line can move between frames
            max_angle_change: Maximum angle change (radians) between frames
            history_size: Number of recent frames to use for prediction
            lost_frames_threshold: How many frames to tolerate without seeing the tracked line
            image_width: Width of image for calculating offsets
        """
        self.max_position_jump = max_position_jump
        self.max_angle_change = max_angle_change
        self.history_size = history_size
        self.lost_frames_threshold = lost_frames_threshold
        self.image_width = image_width
        
        # Tracking state
        self.tracked_line = None        # The line we're currently following
        self.line_history = []          # Recent history of tracked line positions
        self.frames_since_seen = 0      # How many frames since we saw our tracked line
        self.predicted_position = None  # Where we expect the line to be next frame
        
    def select_line(self, lines):
        """
        Select line using temporal tracking with prediction.
        """
        if not lines:
            return self._handle_no_lines()
        
        # First frame - just pick closest to center
        if self.tracked_line is None:
            return self._initialize_tracking(lines)
        
        # Try to find the tracked line among current detections
        matched_line = self._find_matching_line(lines)
        
        if matched_line is not None:
            # Found our tracked line - update tracking
            self._update_tracking(matched_line)
            return matched_line
        else:
            # Didn't find tracked line - handle loss
            return self._handle_lost_line(lines)
    
    def _initialize_tracking(self, lines):
        """Initialize tracking by selecting the line closest to image center."""
        # Sort by distance from center
        center_x = self.image_width / 2
        lines_with_distance = [
            (line, abs((line.x1 + line.x2) / 2 - center_x))
            for line in lines
        ]
        lines_with_distance.sort(key=lambda x: x[1])
        
        # Start tracking the closest line
        best_line = lines_with_distance[0][0]
        self.tracked_line = best_line
        self.line_history = [self._extract_line_features(best_line)]
        self.frames_since_seen = 0
        self.predicted_position = self._extract_line_features(best_line)
        
        return best_line
    
    def _find_matching_line(self, lines):
        """
        Find which detected line (if any) matches our tracked line.
        Uses predicted position and angle to score candidates.
        """
        if self.predicted_position is None:
            return None
        
        pred_center_x, pred_center_y, pred_angle = self.predicted_position
        
        best_match = None
        best_score = float('inf')
        
        for line in lines:
            # Extract features
            center_x = (line.x1 + line.x2) / 2
            center_y = (line.y1 + line.y2) / 2
            angle = line.angle
            
            # Calculate position distance
            position_dist = math.sqrt(
                (center_x - pred_center_x)**2 + 
                (center_y - pred_center_y)**2
            )
            
            # Calculate angle difference (handle wraparound at ±π)
            angle_diff = abs(angle - pred_angle)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            
            # Check if within acceptable thresholds
            if position_dist > self.max_position_jump:
                continue  # Too far away
            if angle_diff > self.max_angle_change:
                continue  # Angle changed too much
            
            # Score this candidate (lower is better)
            # Weight position more heavily than angle for ceiling lines (they're parallel)
            score = position_dist + (angle_diff * 50.0)
            
            if score < best_score:
                best_score = score
                best_match = line
        
        return best_match
    
    def _update_tracking(self, line):
        """Update tracking state with newly matched line."""
        features = self._extract_line_features(line)
        
        # Add to history
        self.line_history.append(features)
        if len(self.line_history) > self.history_size:
            self.line_history.pop(0)
        
        # Update tracked line
        self.tracked_line = line
        self.frames_since_seen = 0
        
        # Update prediction for next frame
        self._update_prediction()
    
    def _update_prediction(self):
        """
        Predict where the line will be in the next frame based on recent motion.
        Uses simple linear extrapolation of recent position changes.
        """
        if len(self.line_history) < 2:
            # Not enough history - predict same position
            self.predicted_position = self.line_history[-1]
            return
        
        # Calculate average velocity over recent history
        velocities_x = []
        velocities_y = []
        angle_changes = []
        
        for i in range(1, len(self.line_history)):
            prev = self.line_history[i-1]
            curr = self.line_history[i]
            
            velocities_x.append(curr[0] - prev[0])
            velocities_y.append(curr[1] - prev[1])
            angle_changes.append(curr[2] - prev[2])
        
        # Average velocity
        avg_vx = sum(velocities_x) / len(velocities_x)
        avg_vy = sum(velocities_y) / len(velocities_y)
        avg_angle_change = sum(angle_changes) / len(angle_changes)
        
        # Predict next position
        current = self.line_history[-1]
        self.predicted_position = (
            current[0] + avg_vx,
            current[1] + avg_vy,
            current[2] + avg_angle_change
        )
    
    def _handle_lost_line(self, lines):
        """
        Handle case where we didn't find our tracked line.
        Keep returning predicted position for a few frames, then re-initialize.
        """
        self.frames_since_seen += 1
        
        if self.frames_since_seen <= self.lost_frames_threshold:
            # Still within tolerance - return last known line with predicted adjustment
            # This keeps the robot moving smoothly even if line disappears briefly
            if self.tracked_line is not None:
                # Create a virtual line at the predicted position
                predicted_line = self._create_predicted_line()
                return predicted_line
        
        # Line has been lost too long - re-initialize with best available line
        self.tracked_line = None
        self.line_history = []
        self.predicted_position = None
        return self._initialize_tracking(lines)
    
    def _handle_no_lines(self):
        """Handle case where no lines detected at all."""
        self.frames_since_seen += 1
        
        if self.frames_since_seen <= self.lost_frames_threshold:
            # Keep using prediction for a bit
            if self.tracked_line is not None:
                return self._create_predicted_line()
        
        # Completely lost - reset tracking
        self.tracked_line = None
        self.line_history = []
        self.predicted_position = None
        return None
    
    def _create_predicted_line(self):
        """
        Create a virtual DetectedLine at the predicted position.
        Used when the actual line isn't detected but we have a good prediction.
        """
        if self.predicted_position is None or self.tracked_line is None:
            return self.tracked_line
        
        pred_center_x, pred_center_y, pred_angle = self.predicted_position
        
        # Create a line segment at the predicted position
        # Use same length as original line
        original_length = math.sqrt(
            (self.tracked_line.x2 - self.tracked_line.x1)**2 +
            (self.tracked_line.y2 - self.tracked_line.y1)**2
        )
        
        # Calculate endpoints from center, angle, and length
        half_length = original_length / 2
        x1 = int(pred_center_x - half_length * math.cos(pred_angle))
        y1 = int(pred_center_y - half_length * math.sin(pred_angle))
        x2 = int(pred_center_x + half_length * math.cos(pred_angle))
        y2 = int(pred_center_y + half_length * math.sin(pred_angle))
        
        # Create predicted line
        predicted_line = DetectedLine()
        predicted_line.header = self.tracked_line.header
        predicted_line.x1 = x1
        predicted_line.y1 = y1
        predicted_line.x2 = x2
        predicted_line.y2 = y2
        predicted_line.offset_x = float(pred_center_x - self.image_width / 2)
        predicted_line.angle = pred_angle
        
        return predicted_line
    
    def _extract_line_features(self, line):
        """Extract features (center_x, center_y, angle) from a DetectedLine."""
        center_x = (line.x1 + line.x2) / 2
        center_y = (line.y1 + line.y2) / 2
        angle = line.angle
        return (center_x, center_y, angle)
