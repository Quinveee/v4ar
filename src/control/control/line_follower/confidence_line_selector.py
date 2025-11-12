from .base_selector import BaseLineSelector
from line_msgs.msg import DetectedLines, DetectedLine


class ConfidenceLineSelector(BaseLineSelector):
    """
    Line selector that maintains confidence scores for tracked lines across frames.
    
    Lines are tracked based on their features (offset_x, angle) and assigned confidence
    scores that decay over time. When a tracked line is matched with a new detection,
    its confidence increases. The line with the highest confidence is selected.
    
    This approach provides temporal stability and reduces jitter compared to
    frame-by-frame selection strategies.
    """
    
    __NAME__ = "confidence_line_selector"
    
    def __init__(self, decay_factor=0.9, match_threshold=50.0, angle_weight=100.0,
                 initial_confidence=0.2, confidence_boost=0.1, min_confidence=0.1,
                 *args, **kwargs):
        """
        Initialize the confidence-based line selector.
        
        Args:
            decay_factor: Multiplicative decay applied to all tracked lines each frame (default: 0.9)
            match_threshold: Maximum distance for matching a detected line to a tracked line (default: 50.0)
            angle_weight: Weight for angle difference in distance calculation (default: 100.0)
            initial_confidence: Confidence score for newly detected lines (default: 0.2)
            confidence_boost: Amount to increase confidence when a line is re-detected (default: 0.1)
            min_confidence: Minimum confidence threshold; lines below this are removed (default: 0.1)
        """
        self.decay_factor = decay_factor
        self.match_threshold = match_threshold
        self.angle_weight = angle_weight
        self.initial_confidence = initial_confidence
        self.confidence_boost = confidence_boost
        self.min_confidence = min_confidence
        
        # List of tracked lines, each element is a dict:
        # {
        #     "offset_x": float,
        #     "angle": float,
        #     "confidence": float,
        #     "header": std_msgs/Header (from last detection),
        #     "x1": int, "y1": int, "x2": int, "y2": int (from last detection)
        # }
        self.tracked_lines = []
    
    def select_line(self, lines):
        """
        Select the line with the highest confidence from tracked lines.

        Process:
        1. Apply confidence decay to all tracked lines
        2. Match new detections to tracked lines or create new tracked lines
        3. Remove lines with confidence below threshold
        4. Return the line with highest confidence as a DetectedLine message

        Args:
            lines: List of DetectedLine messages

        Returns:
            DetectedLine message for the highest-confidence line, or None if no lines tracked
        """
        # Step 1: Apply confidence decay to all tracked lines
        for tracked in self.tracked_lines:
            tracked["confidence"] *= self.decay_factor

        # Step 2: Process each new detected line
        for line in lines:
            # Extract features from the detected line
            offset_x = line.offset_x
            angle = line.angle
            
            # Try to match with existing tracked lines
            best_match_idx = None
            best_match_distance = float('inf')
            
            for idx, tracked in enumerate(self.tracked_lines):
                # Compute distance between detected line and tracked line
                delta_offset = abs(offset_x - tracked["offset_x"])
                delta_angle = abs(angle - tracked["angle"])
                distance = delta_offset + self.angle_weight * delta_angle
                
                # Check if this is the best match so far
                if distance < best_match_distance:
                    best_match_distance = distance
                    best_match_idx = idx
            
            # If a close match is found, update it; otherwise create new tracked line
            if best_match_idx is not None and best_match_distance < self.match_threshold:
                # Update the matched tracked line
                tracked = self.tracked_lines[best_match_idx]
                tracked["offset_x"] = offset_x
                tracked["angle"] = angle
                tracked["confidence"] = min(1.0, tracked["confidence"] + self.confidence_boost)
                
                # Store the full line information for later reconstruction
                tracked["header"] = line.header
                tracked["x1"] = line.x1
                tracked["y1"] = line.y1
                tracked["x2"] = line.x2
                tracked["y2"] = line.y2
            else:
                # Create a new tracked line
                new_tracked = {
                    "offset_x": offset_x,
                    "angle": angle,
                    "confidence": self.initial_confidence,
                    "header": line.header,
                    "x1": line.x1,
                    "y1": line.y1,
                    "x2": line.x2,
                    "y2": line.y2,
                }
                self.tracked_lines.append(new_tracked)
        
        # Step 3: Remove lines with confidence below minimum threshold
        self.tracked_lines = [
            tracked for tracked in self.tracked_lines
            if tracked["confidence"] >= self.min_confidence
        ]
        
        # Step 4: Select and return the line with highest confidence
        if not self.tracked_lines:
            return None
        
        # Find the tracked line with maximum confidence
        best_tracked = max(self.tracked_lines, key=lambda t: t["confidence"])
        
        # Reconstruct a DetectedLine message from the tracked line
        selected_line = DetectedLine()
        selected_line.header = best_tracked["header"]
        selected_line.x1 = best_tracked["x1"]
        selected_line.y1 = best_tracked["y1"]
        selected_line.x2 = best_tracked["x2"]
        selected_line.y2 = best_tracked["y2"]
        selected_line.offset_x = best_tracked["offset_x"]
        selected_line.angle = best_tracked["angle"]
        
        return selected_line

