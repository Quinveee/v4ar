import math
import numpy as np
from .base_strategy import BaseGazeStrategy
from .weighed_gaze_strategy import WeightedGazeStrategy


class ActiveSearchGazeStrategy(BaseGazeStrategy):
    """
    Gaze strategy with two goals:
      1. When markers are visible: keep as many as possible in view and
         gently pan side-to-side to refresh the buffer.
      2. When markers are lost: do a local search, then a wider sweep.

    Pan/tilt outputs are small *incremental* commands (same semantics as
    WeightedGazeStrategy).
    """

    def __init__(
        self,
        img_width=640,
        img_height=480,
        center_x=None,
        center_y=None,
        kp_pan=0.002,
        kp_tilt=0.002,
        # Search behaviour when no markers:
        local_search_frames=15,     # frames of 'wiggle near last direction'
        global_sweep_step=0.01,     # pan increment per frame during global sweep
        global_sweep_range=1.0,     # approx. sweep range in servo units
        local_search_step=0.005,    # pan increment per frame during local search
        tilt_home=0.0,              # default tilt when searching
        # NEW: scanning while markers visible:
        max_center_offset_px=None,  # max horizontal shift of target center (pixels)
        center_offset_step_px=2.0,  # how many pixels per frame we move the target
        safe_margin_px=60.0,        # required margin to edges before using full scan amplitude
    ):
        self.weighted = WeightedGazeStrategy(
            img_width=img_width,
            img_height=img_height,
            center_x=center_x,
            center_y=center_y,
            kp_pan=kp_pan,
            kp_tilt=kp_tilt,
        )

        self.img_width = img_width
        self.img_height = img_height

        # Parameters for "no markers" search
        self.local_search_frames = local_search_frames
        self.global_sweep_step = global_sweep_step
        self.global_sweep_range = global_sweep_range
        self.local_search_step = local_search_step
        self.tilt_home = tilt_home

        # Parameters for scanning while markers ARE visible
        self.safe_margin_px = safe_margin_px
        self.center_offset_step_px = center_offset_step_px
        self.max_center_offset_px = (
            max_center_offset_px
            if max_center_offset_px is not None
            else img_width * 0.25  # default: ±1/4 of image width
        )

        # State
        self.frames_since_seen = 0
        self.last_track_cmd = (0.0, 0.0)

        # Scan state in "target-center" space (pixels)
        self.center_offset_x = 0.0
        self.scan_dir = 1.0  # +1 or -1

        # For local search (no markers)
        self.local_offset = 0.0
        self.local_dir = 1.0

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def compute_angles(self, msg):
        markers = msg.markers

        if markers:
            return self._track_with_scanning(markers)

        # No markers detected this frame
        self.frames_since_seen += 1
        # Reset weighted center to image center when we're blind
        self.weighted.center_x = self.img_width / 2.0

        if 0 < self.frames_since_seen <= self.local_search_frames:
            return self._local_search()
        else:
            return self._global_sweep()

    # ------------------------------------------------------------------
    # Tracking + scanning when markers are visible
    # ------------------------------------------------------------------
    def _track_with_scanning(self, markers):
        self.frames_since_seen = 0

        # --- 1) Update scan offset in pixel space (slow, smooth) ---
        self.center_offset_x += self.center_offset_step_px * self.scan_dir

        if abs(self.center_offset_x) > self.max_center_offset_px:
            # Hit one side of allowed offset -> clamp & flip direction
            self.center_offset_x = math.copysign(
                self.max_center_offset_px, self.center_offset_x
            )
            self.scan_dir *= -1.0

        # --- 2) Check how close marker cluster is to image edges ---
        xs = [m.center_x for m in markers]
        x_min = min(xs)
        x_max = max(xs)

        margin_left = x_min
        margin_right = self.img_width - x_max
        margin = min(margin_left, margin_right)

        # If markers are close to an edge, reduce scan amplitude so we don't lose them
        # margin >= safe_margin_px -> full scan
        # margin <= 0 -> no scan
        safety_factor = max(0.0, min(1.0, margin / self.safe_margin_px))

        effective_offset = self.center_offset_x * safety_factor

        # --- 3) Apply dynamic target center to WeightedGazeStrategy ---
        base_center = self.img_width / 2.0
        self.weighted.center_x = base_center + effective_offset

        # WeightedGazeStrategy now tries to pull the cluster towards this *moving* center
        pan_cmd, tilt_cmd = self.weighted.compute_angles_from_markers(markers)

        self.last_track_cmd = (pan_cmd, tilt_cmd)
        return pan_cmd, tilt_cmd

    # ------------------------------------------------------------------
    # Local search (short loss of markers)
    # ------------------------------------------------------------------
    def _local_search(self):
        """
        Small oscillation around last direction when we just lost the markers.
        """
        step = self.local_search_step * self.local_dir
        self.local_offset += step

        # Reverse direction when we've gone 'too far' from the origin
        if abs(self.local_offset) > self.global_sweep_range * 0.25:
            self.local_dir *= -1.0

        pan_cmd = step
        # Small bias towards home tilt
        tilt_cmd = (self.tilt_home - 0.0) * 0.1

        return pan_cmd, tilt_cmd

    # ------------------------------------------------------------------
    # Global sweep (no markers for a while)
    # ------------------------------------------------------------------
    def _global_sweep(self):
        """
        Wider left-right sweep when we've been blind for longer.
        """
        step = self.global_sweep_step * self.scan_dir
        self.center_offset_x += step

        if abs(self.center_offset_x) > self.global_sweep_range:
            self.center_offset_x = math.copysign(
                self.global_sweep_range, self.center_offset_x
            )
            self.scan_dir *= -1.0

        pan_cmd = step
        tilt_cmd = (self.tilt_home - 0.0) * 0.1

        return pan_cmd, tilt_cmd
