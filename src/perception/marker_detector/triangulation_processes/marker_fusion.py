import time


class MarkerFusion:
    def __init__(self, oak_weight=2.0, mono_weight=1.0, timeout=0.5):
        self.oak_weight = oak_weight
        self.mono_weight = mono_weight
        self.timeout = timeout
        self.oak_detections = {}
        self.mono_detections = {}

    def add_detections(self, detections, source, timestamp):
        """Add detections from a specific source."""
        store = self.oak_detections if source == "oak" else self.mono_detections
        for m in detections:
            store[m.id] = (m, timestamp)

    def fuse(self):
        """Fuse detections from both sources, prioritizing OAK."""
        now = time.time()
        fused = []

        # OAK detections take priority
        for m_id, (m, t) in self.oak_detections.items():
            if now - t <= self.timeout:
                fused.append((m_id, m.distance, self.oak_weight))

        # Add mono detections if not covered by OAK
        for m_id, (m, t) in self.mono_detections.items():
            if now - t <= self.timeout and m_id not in [f[0] for f in fused]:
                fused.append((m_id, m.distance, self.mono_weight))

        # Prune old detections
        self.oak_detections = {k: v for k, v in self.oak_detections.items() 
                              if now - v[1] <= self.timeout}
        self.mono_detections = {k: v for k, v in self.mono_detections.items() 
                               if now - v[1] <= self.timeout}
        
        return fused
