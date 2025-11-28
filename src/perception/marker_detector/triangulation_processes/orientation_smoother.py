import math
import time


class OrientationSmoother:
    def __init__(self, buffer_size=10, timeout=0.5, alpha=0.3):
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.alpha = alpha  # Smoothing factor
        self.orientation_buffer = []
        self.current_smoothed_yaw = None

    def update(self, new_yaw, distance):
        """Update orientation with smoothing."""
        now = time.time()
        
        # Add new measurement to buffer
        self.orientation_buffer.append((new_yaw, distance, now))
        
        # Remove old measurements
        self.orientation_buffer = [
            (yaw, dist, ts) for yaw, dist, ts in self.orientation_buffer
            if (now - ts) <= self.timeout
        ]
        
        # Keep buffer size manageable
        if len(self.orientation_buffer) > self.buffer_size:
            self.orientation_buffer = self.orientation_buffer[-self.buffer_size:]
        
        # Compute weighted average
        if not self.orientation_buffer:
            return new_yaw
            
        weights = []
        yaws = []
        
        for yaw, distance, timestamp in self.orientation_buffer:
            # Weight inversely proportional to distance (closer = higher weight)
            distance_weight = 1.0 / (distance + 0.1)
            
            # Time decay (more recent = higher weight)
            age = now - timestamp
            time_weight = math.exp(-age / self.timeout)
            
            total_weight = distance_weight * time_weight
            weights.append(total_weight)
            yaws.append(yaw)
        
        # Weighted circular mean for angles
        if weights:
            # Convert to unit vectors for circular averaging
            sin_sum = sum(w * math.sin(yaw) for w, yaw in zip(weights, yaws))
            cos_sum = sum(w * math.cos(yaw) for w, yaw in zip(weights, yaws))
            weight_sum = sum(weights)
            
            if weight_sum > 0:
                smoothed_yaw = math.atan2(sin_sum / weight_sum, cos_sum / weight_sum)
            else:
                smoothed_yaw = new_yaw
        else:
            smoothed_yaw = new_yaw
        
        # Apply additional smoothing if we have a previous value
        if self.current_smoothed_yaw is not None:
            # Handle angle wraparound
            diff = smoothed_yaw - self.current_smoothed_yaw
            if diff > math.pi:
                diff -= 2 * math.pi
            elif diff < -math.pi:
                diff += 2 * math.pi
            
            smoothed_yaw = self.current_smoothed_yaw + self.alpha * diff
        
        self.current_smoothed_yaw = smoothed_yaw
        
        # Normalize to [-pi, pi]
        return math.atan2(math.sin(smoothed_yaw), math.cos(smoothed_yaw))
