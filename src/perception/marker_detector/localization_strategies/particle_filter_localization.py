"""
Particle Filter (Monte Carlo Localization) Strategy.

This strategy represents the robot's pose as a set of weighted particles.
Each particle is a hypothesis about the robot's location.

Key advantages:
- Can represent multi-modal distributions (multiple location hypotheses)
- No assumption about Gaussian noise
- Robust to outliers and non-linear dynamics
- Can recover from "kidnapping"
"""

import numpy as np
from .base_strategy import BaseLocalizationStrategy


class ParticleFilterLocalization(BaseLocalizationStrategy):
    """
    Particle filter localization strategy.

    Maintains N particles, each representing a possible robot pose.
    Uses sequential importance resampling (SIR) algorithm.
    """

    def __init__(
        self,
        n_particles=300,
        process_noise_std=(0.05, 0.05, 0.05),  # (x, y, theta)
        measurement_noise_std=0.1,
        resample_threshold=0.5,
        initial_noise_std=(0.2, 0.2, 0.3)
    ):
        """
        Initialize particle filter.

        Args:
            n_particles: Number of particles to maintain
            process_noise_std: Standard deviation of process noise (x, y, theta)
            measurement_noise_std: Standard deviation of measurement noise (meters)
            resample_threshold: Resample when N_eff < threshold * n_particles
            initial_noise_std: Std for initial particle distribution
        """
        super().__init__()

        self.n_particles = n_particles
        self.process_noise_std = np.array(process_noise_std)
        self.measurement_noise_std = measurement_noise_std
        self.resample_threshold = resample_threshold
        self.initial_noise_std = np.array(initial_noise_std)

        # Particles: each row is [x, y, theta]
        self.particles = np.zeros((n_particles, 3))

        # Weights: probability of each particle
        self.weights = np.ones(n_particles) / n_particles

        # State
        self.initialized = False
        self.measurement_count = 0

    def predict(self, v, w, dt):
        """
        Prediction step: propagate all particles using motion model with noise.

        Each particle gets slightly different motion due to added noise,
        representing uncertainty in odometry.
        """
        if dt <= 0 or not self.initialized:
            return

        # Add noise to control inputs (different for each particle)
        v_noisy = v + np.random.normal(0, self.process_noise_std[0], self.n_particles)
        w_noisy = w + np.random.normal(0, self.process_noise_std[2], self.n_particles)

        # Motion model for each particle
        for i in range(self.n_particles):
            theta = self.particles[i, 2]

            # Update position based on noisy control
            self.particles[i, 0] += v_noisy[i] * np.cos(theta) * dt
            self.particles[i, 1] += v_noisy[i] * np.sin(theta) * dt
            self.particles[i, 2] += w_noisy[i] * dt

            # Add additional process noise
            self.particles[i, 0] += np.random.normal(0, self.process_noise_std[0] * dt)
            self.particles[i, 1] += np.random.normal(0, self.process_noise_std[1] * dt)

            # Normalize angle
            self.particles[i, 2] = np.arctan2(
                np.sin(self.particles[i, 2]),
                np.cos(self.particles[i, 2])
            )

    def update(self, measurement):
        """
        Update step: weight particles based on measurement likelihood.

        Particles closer to the measurement get higher weights.
        """
        # Extract measurement
        z_x = measurement.pose.position.x
        z_y = measurement.pose.position.y

        # Initialize particles on first measurement
        if not self.initialized:
            self._initialize_particles(z_x, z_y)
            self.initialized = True
            self.measurement_count += 1
            return

        # Compute likelihood for each particle
        for i in range(self.n_particles):
            # Expected measurement for this particle
            expected_x = self.particles[i, 0]
            expected_y = self.particles[i, 1]

            # Measurement residual
            dx = z_x - expected_x
            dy = z_y - expected_y
            distance = np.sqrt(dx**2 + dy**2)

            # Likelihood: Gaussian probability
            # p(z|x) = exp(-distance^2 / (2*sigma^2))
            likelihood = np.exp(
                -distance**2 / (2 * self.measurement_noise_std**2)
            )

            # Update weight
            self.weights[i] *= likelihood

        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 1e-10:
            self.weights /= weight_sum
        else:
            # All weights are zero (catastrophic failure)
            # Reset to uniform distribution
            self.weights = np.ones(self.n_particles) / self.n_particles

        # Compute effective number of particles
        n_eff = 1.0 / (np.sum(self.weights**2) + 1e-10)

        # Resample if effective particles too low
        if n_eff < self.resample_threshold * self.n_particles:
            self._resample()

        # Update state estimate
        self._update_state_estimate()

        self.measurement_count += 1

    def _initialize_particles(self, x_init, y_init):
        """Initialize particles around first measurement."""
        # Generate particles with Gaussian noise around initial position
        self.particles[:, 0] = x_init + np.random.normal(
            0, self.initial_noise_std[0], self.n_particles
        )
        self.particles[:, 1] = y_init + np.random.normal(
            0, self.initial_noise_std[1], self.n_particles
        )
        self.particles[:, 2] = np.random.normal(
            0, self.initial_noise_std[2], self.n_particles
        )

        # Uniform weights
        self.weights = np.ones(self.n_particles) / self.n_particles

        # Set initial state
        self.x = x_init
        self.y = y_init
        self.theta = 0.0

    def _resample(self):
        """
        Low variance resampling algorithm.

        Particles with higher weights are more likely to be duplicated.
        This prevents particle degeneracy.
        """
        # Cumulative sum of weights
        cumsum = np.cumsum(self.weights)

        # Low variance resampling
        new_particles = np.zeros_like(self.particles)
        r = np.random.uniform(0, 1.0 / self.n_particles)

        i = 0
        for m in range(self.n_particles):
            u = r + m * (1.0 / self.n_particles)

            # Find particle to sample
            while u > cumsum[i]:
                i += 1

            # Copy particle
            new_particles[m] = self.particles[i]

        # Replace old particles
        self.particles = new_particles

        # Reset weights to uniform
        self.weights = np.ones(self.n_particles) / self.n_particles

    def _update_state_estimate(self):
        """
        Compute state estimate as weighted average of particles.

        For theta, we use circular mean to handle angle wraparound.
        """
        # Weighted mean for x, y
        self.x = np.sum(self.weights * self.particles[:, 0])
        self.y = np.sum(self.weights * self.particles[:, 1])

        # Circular weighted mean for theta
        sin_sum = np.sum(self.weights * np.sin(self.particles[:, 2]))
        cos_sum = np.sum(self.weights * np.cos(self.particles[:, 2]))
        self.theta = np.arctan2(sin_sum, cos_sum)

    def get_particle_cloud(self):
        """
        Get particle cloud for visualization.

        Returns:
            tuple: (particles array, weights array)
        """
        return self.particles.copy(), self.weights.copy()

    def get_state_uncertainty(self):
        """
        Compute state uncertainty as weighted covariance.

        Returns:
            numpy.ndarray: 3x3 covariance matrix
        """
        # Mean state
        mean = np.array([self.x, self.y, self.theta])

        # Compute weighted covariance
        diff = self.particles - mean
        cov = np.zeros((3, 3))

        for i in range(self.n_particles):
            cov += self.weights[i] * np.outer(diff[i], diff[i])

        return cov

    def get_diagnostics(self):
        """
        Get diagnostic information.

        Returns:
            dict: Diagnostic statistics
        """
        # Effective number of particles
        n_eff = 1.0 / (np.sum(self.weights**2) + 1e-10)

        # Max weight
        max_weight = np.max(self.weights)

        # State uncertainty
        cov = self.get_state_uncertainty()
        position_uncertainty = np.sqrt(cov[0, 0] + cov[1, 1])

        return {
            'n_effective': n_eff,
            'max_weight': max_weight,
            'position_uncertainty': position_uncertainty,
            'measurement_count': self.measurement_count,
            'initialized': self.initialized
        }
