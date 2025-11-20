# Detected rover information
int32 id                        # Unique ID for tracking
geometry_msgs/Pose pose         # 3D pose in camera frame
float32 distance                # Euclidean distance from camera (meters)
float32 angle_horizontal        # Horizontal angle from camera center (radians, + = right)
float32 angle_vertical          # Vertical angle from camera center (radians, + = up)
float32 center_x                # Pixel coordinates of center
float32 center_y
float32 bbox_x_min              # Bounding box in pixels
float32 bbox_y_min
float32 bbox_x_max
float32 bbox_y_max
float32 confidence              # Detection confidence [0-1]
float32 area                    # Area in pixels