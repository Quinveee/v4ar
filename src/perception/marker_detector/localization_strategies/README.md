# Localization Strategies

This directory contains multiple localization strategies for fusing odometry (from `/cmd_vel`) with vision measurements (from `/robot_pose_raw`).

## Available Strategies

### 1. Identity (`identity`)

**Simple pass-through strategy** - just uses raw measurements without any filtering.

**Use when:**

- Testing vision system
- Vision measurements are extremely accurate
- Debugging

**Parameters:** None

---

### 2. Kalman Filter (`kalman`)

**Standard Extended Kalman Filter (EKF)** - optimal for linear-Gaussian systems.

**How it works:**

- Predicts pose using odometry motion model
- Updates using vision measurements
- Assumes Gaussian noise with fixed covariances

**Use when:**

- Measurements are reasonably reliable
- You want a well-tested, standard approach
- Computational resources are limited

**Parameters:**

- Process noise Q: Controls trust in odometry (default: diagonal [0.02, 0.02, 0.01])
- Measurement noise R: Controls trust in vision (default: diagonal [0.05, 0.05])

**Implementation:** [kalman_localization.py](kalman_localization.py)

---

### 3. Complementary Filter (`complementary`)

**Simple weighted average** between odometry and vision.

**How it works:**

```python
x = (1 - alpha) * x_odometry + alpha * x_vision
```

**Use when:**

- You want something simple and fast
- Don't need optimal estimates
- Tuning one parameter (alpha) is easier than Q/R matrices

**Parameters:**

- `alpha`: Weight for vision (0-1), default: 0.7

**Implementation:** [complementary_localization.py](complementary_localization.py)

---

### 4. Robust Filter (`robust`)

**Complementary filter with outlier rejection.**

**How it works:**

- Similar to complementary filter
- Rejects measurements that jump too far from current estimate
- Good for handling intermittent marker detections

**Use when:**

- Vision measurements have occasional outliers
- Markers sometimes misdetected
- You want simple outlier handling

**Parameters:**

- `max_jump`: Maximum allowed measurement jump (meters), default: 0.75
- `alpha`: Fusion weight, default: 0.7

**Implementation:** [robust_localization.py](robust_localization.py)

---

### 5. Adaptive Kalman Filter (`adaptive_kalman`) ⭐ **NEW**

**Kalman filter that automatically adjusts measurement noise based on innovation statistics.**

**How it works:**

1. Runs standard Kalman filter
2. Monitors innovation (measurement residuals)
3. If innovations are large → increases R (trusts measurements less)
4. If innovations are small → decreases R (trusts measurements more)
5. Uses Chi-squared test to detect outliers

**Use when:**

- Marker visibility varies (occlusions, distance, lighting)
- You want automatic tuning
- Vision quality is unpredictable

**Key Features:**

- **Innovation monitoring**: Tracks how wrong predictions are
- **Outlier detection**: Uses Mahalanobis distance and Chi-squared test
- **Adaptive R matrix**: Scales measurement noise from 0.1x to 10x baseline
- **Statistics tracking**: Logs outlier rate, R scale factor

**Parameters:**

- `process_noise_scale`: Scale for Q matrix, default: 1.0
- `measurement_noise_base`: Base R value, default: 0.05
- `innovation_window`: Samples for statistics, default: 10
- `adaptation_rate`: How fast to adapt (0-1), default: 0.1
- `min_r_scale`: Minimum R scaling, default: 0.1
- `max_r_scale`: Maximum R scaling, default: 10.0

**Diagnostics available:**

```python
localizer.get_diagnostics()
# Returns: {
#   'r_scale': 1.5,           # Current R scaling factor
#   'outlier_rate': 0.05,     # Percentage of outliers detected
#   'uncertainty': 0.12       # Trace of covariance matrix
# }
```

**Implementation:** [adaptive_kalman_localization.py](adaptive_kalman_localization.py)

---

### 6. Particle Filter (`particle_filter`) ⭐ **NEW**

**Monte Carlo Localization** - maintains hundreds of pose hypotheses.

**How it works:**

1. Represents pose as 300 weighted particles
2. **Prediction**: Propagates each particle with noisy odometry
3. **Update**: Weights particles based on measurement likelihood
4. **Resampling**: Removes low-weight particles, duplicates high-weight ones
5. **Estimation**: Weighted average gives final pose

**Use when:**

- Need to handle multi-modal distributions (multiple hypotheses)
- Dealing with non-Gaussian noise
- Want robustness to outliers
- Need global localization capability
- Have spare CPU (more expensive than Kalman)

**Key Features:**

- **Multi-hypothesis**: Can track multiple possible locations
- **Non-parametric**: No Gaussian assumption needed
- **Robust**: Naturally handles outliers
- **Recoverable**: Can recover from bad initializations

**Parameters:**

- `n_particles`: Number of particles, default: 300 (100-1000 range)
- `process_noise_std`: Process noise (x, y, theta), default: (0.05, 0.05, 0.05)
- `measurement_noise_std`: Measurement noise, default: 0.1
- `resample_threshold`: When to resample, default: 0.5
- `initial_noise_std`: Initial spread, default: (0.2, 0.2, 0.3)

**Diagnostics available:**

```python
localizer.get_diagnostics()
# Returns: {
#   'n_effective': 250.3,        # Effective number of particles
#   'max_weight': 0.05,          # Highest particle weight
#   'position_uncertainty': 0.15 # Positional uncertainty
# }

# Get particle cloud for visualization
particles, weights = localizer.get_particle_cloud()
```

**Tuning Tips:**

- More particles = more accurate but slower
- Increase `process_noise_std` if odometry is unreliable
- Increase `measurement_noise_std` if vision is noisy
- Lower `resample_threshold` = less frequent resampling

**Implementation:** [particle_filter_localization.py](particle_filter_localization.py)

---

### 7. Sliding Window Smoother (`sliding_window`) ⭐ **NEW**

**Rauch-Tung-Striebel (RTS) smoother** - uses future measurements to refine past estimates.

**How it works:**

1. **Forward pass**: Runs standard Kalman filter, saves all states
2. **Backward pass**: Smooths estimates using future information
3. **Output**: Returns lagged but smoother trajectory

**Use when:**

- Accuracy is more important than real-time
- You can tolerate 100-200ms delay
- Want smooth trajectories for plotting/analysis
- Doing offline processing or validation

**Key Features:**

- **Smoother trajectories**: No jitter or noise
- **Better accuracy**: Uses "future" info to correct "past"
- **Delayed output**: Lag controlled by parameter
- **Fixed memory**: Maintains sliding window

**Parameters:**

- `window_size`: States in window, default: 10
- `process_noise_scale`: Scale for Q, default: 1.0
- `measurement_noise`: R value, default: 0.05
- `lag`: Output delay in steps, default: 5

**Diagnostics available:**

```python
localizer.get_diagnostics()
# Returns: {
#   'window_size': 10,
#   'lag': 5,
#   'uncertainty': 0.08
# }

# Get full smoothed trajectory
trajectory = localizer.get_smoothed_trajectory()
# Returns: (N x 3) array of [x, y, theta]
```

**Trade-offs:**

- ✅ Most accurate of all strategies
- ✅ Smoothest trajectories
- ❌ Latency = lag _ dt (e.g., 5 steps _ 50ms = 250ms)
- ❌ Higher memory usage

**Use cases:**

- Offline trajectory analysis
- Ground truth generation
- Smooth path visualization
- Post-processing logged data

**Implementation:** [sliding_window_localization.py](sliding_window_localization.py)

---

## Quick Comparison

| Strategy            | Accuracy      | Robustness    | CPU Cost   | Latency  | Complexity |
| ------------------- | ------------- | ------------- | ---------- | -------- | ---------- |
| Identity            | Low           | Low           | Minimal    | None     | Trivial    |
| Kalman              | Medium        | Medium        | Low        | None     | Low        |
| Complementary       | Low-Medium    | Low           | Minimal    | None     | Trivial    |
| Robust              | Medium        | Medium-High   | Low        | None     | Low        |
| **Adaptive Kalman** | **High**      | **High**      | **Low**    | **None** | **Medium** |
| **Particle Filter** | **High**      | **Very High** | **High**   | **None** | **High**   |
| **Sliding Window**  | **Very High** | **Medium**    | **Medium** | **High** | **Medium** |

## Usage Examples

### Command Line

```bash
# Use adaptive Kalman filter
ros2 run marker_detector localization_node --ros-args -p strategy_type:=adaptive_kalman

# Use particle filter with 500 particles
ros2 run marker_detector localization_node --ros-args -p strategy_type:=particle_filter

# Use sliding window smoother
ros2 run marker_detector localization_node --ros-args -p strategy_type:=sliding_window

# Standard Kalman (default)
ros2 run marker_detector localization_node --ros-args -p strategy_type:=kalman
```

### Launch File

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='marker_detector',
            executable='localization_node',
            parameters=[{
                'strategy_type': 'adaptive_kalman',
                'publish_rate': 20.0
            }]
        )
    ])
```

### Programmatic

```python
from localization_strategies import AdaptiveKalmanLocalization, ParticleFilterLocalization

# Create adaptive Kalman with custom parameters
localizer = AdaptiveKalmanLocalization(
    measurement_noise_base=0.08,
    adaptation_rate=0.15,
    max_r_scale=15.0
)

# Create particle filter with more particles
localizer = ParticleFilterLocalization(
    n_particles=500,
    process_noise_std=(0.03, 0.03, 0.04)
)

# Use in your node
node = LocalizationNode()
node.localizer = localizer
```

## Recommendations

### For Curling Robot

**Primary choice: Adaptive Kalman Filter**

- Handles varying marker visibility automatically
- No latency issues for real-time control
- Low computational cost
- Good balance of accuracy and robustness

**Alternative: Particle Filter**

- Use if you have multiple identical markers
- Better for global localization
- Good if you need multi-hypothesis tracking
- Worth trying if Kalman struggles with outliers

**For Analysis: Sliding Window**

- Perfect for offline trajectory analysis
- Use to validate other strategies
- Great for generating reference trajectories
- Not recommended for real-time control due to lag

### Tuning Guide

**Start with defaults**, then:

1. **If localization jumps around:**

   - Adaptive Kalman: Decrease `adaptation_rate`
   - Particle Filter: Increase `n_particles`
   - Sliding Window: Increase `lag`

2. **If localization is too slow to respond:**

   - Adaptive Kalman: Increase `adaptation_rate`
   - Particle Filter: Decrease `measurement_noise_std`
   - Sliding Window: Not suitable (inherent lag)

3. **If markers often occluded:**

   - Use Adaptive Kalman (best choice)
   - Or Particle Filter with higher `process_noise_std`

4. **If you see many outliers:**
   - Adaptive Kalman: Check `outlier_rate` in diagnostics
   - Particle Filter: Increase `measurement_noise_std`
   - Consider using Robust filter instead

## Testing & Validation

### Compare Strategies

Run all strategies simultaneously and compare:

```bash
# Terminal 1: Ground truth (sliding window)
ros2 run marker_detector localization_node --ros-args -p strategy_type:=sliding_window

# Terminal 2: Test strategy
ros2 run marker_detector localization_node --ros-args -p strategy_type:=adaptive_kalman

# Compare /robot_pose topics
ros2 topic echo /robot_pose
```

### Monitor Diagnostics

Extend the localization node to publish diagnostics for monitoring strategy performance.

## Implementation Notes

All strategies inherit from `BaseLocalizationStrategy` with this interface:

```python
class BaseLocalizationStrategy:
    def predict(self, v, w, dt):
        """Propagate state using odometry."""
        pass

    def update(self, measurement: PoseStamped):
        """Update state using vision measurement."""
        pass

    def get_pose(self):
        """Return current pose estimate as (x, y, theta)."""
        return self.x, self.y, self.theta
```

Additional methods in new strategies:

- `get_diagnostics()`: Return performance metrics
- `get_particle_cloud()`: Particle filter only
- `get_smoothed_trajectory()`: Sliding window only
