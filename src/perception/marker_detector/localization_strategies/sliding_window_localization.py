"""
Sliding Window Smoother Localization Strategy.

This strategy maintains a fixed-size window of past poses and optimizes
them jointly using all available measurements. This results in smoother
and more accurate trajectories at the cost of some latency.

Uses Rauch-Tung-Striebel (RTS) smoother - a backward pass over the
forward Kalman filter estimates.
"""

import numpy as np
from collections import deque
from .base_strategy import BaseLocalizationStrategy


class SlidingWindowLocalization(BaseLocalizationStrategy):
    """
    Sliding window smoother localization strategy.

    Uses RTS (Rauch-Tung-Striebel) smoother with a fixed lag.
    Maintains a window of past states and smooths them using future measurements.
    """

    def __init__(
        self,
        window_size=10,
        process_noise_scale=1.0,
        measurement_noise=0.05,
        lag=5
    ):
        """
        Initialize sliding window smoother.

        Args:
            window_size: Number of poses to keep in window
            process_noise_scale: Scale factor for process noise Q
            measurement_noise: Measurement noise standard deviation (meters)
            lag: Output lag (poses) - output is delayed by this many steps
        """
        super().__init__()

        self.window_size = window_size
        self.lag = min(lag, window_size - 1)

        # Noise parameters
        self.Q = np.diag([0.02, 0.02, 0.01]) * process_noise_scale
        self.R = np.diag([measurement_noise, measurement_noise])

        # State history (for smoothing)
        # Each entry: (state, covariance, predicted_state, predicted_cov)
        self.state_history = deque(maxlen=window_size)

        # Control history (v, w, dt)
        self.control_history = deque(maxlen=window_size)

        # Measurement history
        self.measurement_history = deque(maxlen=window_size)

        # Smoothed states
        self.smoothed_states = deque(maxlen=window_size)

        # Current forward estimate
        self.P = np.eye(3) * 0.1

        # Track if initialized
        self.initialized = False
        self.step_count = 0

    def predict(self, v, w, dt):
        """
        Prediction step: run forward Kalman filter and store states.
        """
        if dt <= 0:
            return

        # Store control for later smoothing
        self.control_history.append((v, w, dt))

        if not self.initialized:
            return

        # Save current state before prediction
        state_before = np.array([self.x, self.y, self.theta])
        P_before = self.P.copy()

        # Motion model
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += w * dt

        # Jacobian
        F = np.array([
            [1, 0, -v * np.sin(self.theta) * dt],
            [0, 1,  v * np.cos(self.theta) * dt],
            [0, 0,  1]
        ])

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q

        # Store prediction for smoothing
        predicted_state = np.array([self.x, self.y, self.theta])
        predicted_P = self.P.copy()

        # Store state history (before_state, before_P, predicted_state, predicted_P, F)
        self.state_history.append({
            'state': state_before,
            'P': P_before,
            'predicted_state': predicted_state,
            'predicted_P': predicted_P,
            'F': F
        })

    def update(self, measurement):
        """
        Update step: incorporate measurement and run smoother.
        """
        # Extract measurement
        z = np.array([measurement.pose.position.x, measurement.pose.position.y])

        # Initialize on first measurement
        if not self.initialized:
            self.x = z[0]
            self.y = z[1]
            self.theta = 0.0
            self.P = np.eye(3) * 0.1
            self.initialized = True

            # Initialize history
            state = np.array([self.x, self.y, self.theta])
            self.state_history.append({
                'state': state,
                'P': self.P.copy(),
                'predicted_state': state,
                'predicted_P': self.P.copy(),
                'F': np.eye(3)
            })
            self.smoothed_states.append(state)
            self.measurement_history.append(z)
            return

        # Store measurement
        self.measurement_history.append(z)

        # Measurement model
        H = np.array([[1, 0, 0],
                      [0, 1, 0]])

        # Innovation
        z_pred = H @ np.array([self.x, self.y, self.theta])
        innovation = z - z_pred

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Singular matrix, skip update
            return

        # Update state
        state_update = K @ innovation
        self.x += state_update[0]
        self.y += state_update[1]
        self.theta += state_update[2]

        # Update covariance
        self.P = (np.eye(3) - K @ H) @ self.P

        # Update the last state in history with corrected values
        if len(self.state_history) > 0:
            self.state_history[-1]['state'] = np.array([self.x, self.y, self.theta])
            self.state_history[-1]['P'] = self.P.copy()

        # Run RTS smoother backward pass
        self._smooth_backward()

        # Update output state (with lag)
        self._update_output_state()

        self.step_count += 1

    def _smooth_backward(self):
        """
        RTS (Rauch-Tung-Striebel) backward smoothing pass.

        Starting from the most recent (filtered) state, propagate smoothed
        estimates backward through the state history.
        """
        if len(self.state_history) < 2:
            # Not enough states to smooth
            self.smoothed_states = deque([h['state'] for h in self.state_history],
                                        maxlen=self.window_size)
            return

        # Initialize smoothed states list
        smoothed = [None] * len(self.state_history)

        # Last state is just the filtered state
        smoothed[-1] = {
            'state': self.state_history[-1]['state'].copy(),
            'P': self.state_history[-1]['P'].copy()
        }

        # Backward pass
        for k in range(len(self.state_history) - 2, -1, -1):
            # Get filtered estimate at k
            x_k = self.state_history[k]['state']
            P_k = self.state_history[k]['P']

            # Get prediction at k+1
            x_pred_kp1 = self.state_history[k + 1]['predicted_state']
            P_pred_kp1 = self.state_history[k + 1]['predicted_P']

            # Get transition matrix
            F_k = self.state_history[k + 1]['F']

            # Smoother gain
            try:
                C_k = P_k @ F_k.T @ np.linalg.inv(P_pred_kp1)
            except np.linalg.LinAlgError:
                # Singular matrix, use filtered estimate
                smoothed[k] = {
                    'state': x_k.copy(),
                    'P': P_k.copy()
                }
                continue

            # Smoothed state
            x_smooth_kp1 = smoothed[k + 1]['state']
            x_smooth_k = x_k + C_k @ (x_smooth_kp1 - x_pred_kp1)

            # Smoothed covariance
            P_smooth_kp1 = smoothed[k + 1]['P']
            P_smooth_k = P_k + C_k @ (P_smooth_kp1 - P_pred_kp1) @ C_k.T

            smoothed[k] = {
                'state': x_smooth_k,
                'P': P_smooth_k
            }

        # Store smoothed states
        self.smoothed_states = deque([s['state'] for s in smoothed],
                                      maxlen=self.window_size)

    def _update_output_state(self):
        """
        Update the output state with appropriate lag.

        The output is the smoothed estimate from 'lag' steps ago.
        """
        if len(self.smoothed_states) <= self.lag:
            # Not enough history, use most recent
            if len(self.smoothed_states) > 0:
                state = self.smoothed_states[-1]
                self.x = state[0]
                self.y = state[1]
                self.theta = state[2]
        else:
            # Use lagged smoothed state
            lag_idx = -(self.lag + 1)
            state = self.smoothed_states[lag_idx]
            self.x = state[0]
            self.y = state[1]
            self.theta = state[2]

    def get_smoothed_trajectory(self):
        """
        Get the full smoothed trajectory in the window.

        Returns:
            numpy.ndarray: Array of smoothed states (N x 3)
        """
        if len(self.smoothed_states) == 0:
            return np.array([])

        return np.array(list(self.smoothed_states))

    def get_diagnostics(self):
        """
        Get diagnostic information.

        Returns:
            dict: Diagnostic statistics
        """
        uncertainty = np.trace(self.P) if self.initialized else 0

        return {
            'window_size': len(self.state_history),
            'smoothed_count': len(self.smoothed_states),
            'lag': self.lag,
            'step_count': self.step_count,
            'uncertainty': uncertainty,
            'initialized': self.initialized
        }
