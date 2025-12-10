# SfM Reconstruction Pipeline for ROS2

A complete **Structure-from-Motion (SfM)** reconstruction system integrated into your v4ar ROS2 repository. This module extracts 3D point clouds from camera data recorded in ROS2 rosbags using **COLMAP**.

## Overview

```
   ROS2 Rosbag (with images)
        ↓
   Extract Images & Camera Info
        ↓
   COLMAP Feature Extraction
        ↓
   Feature Matching (SIFT)
        ↓
   Sparse Reconstruction (SfM)
        ↓
   Image Undistortion
        ↓
   Dense Stereo Matching
        ↓
   Stereo Fusion
        ↓
   Dense Point Cloud (PLY/PCD)
```

## Installation

### 1. Build the Package

```bash
cd /home/jetson/ugv_ws/v4ar
colcon build --packages-select sfm_reconstruction
source install/setup.bash
```

### 2. Install Dependencies

```bash
# Install COLMAP, PCL tools, and visualization software
sudo bash src/sfm_reconstruction/scripts/install_dependencies.sh
```

Or manually install:

```bash
# COLMAP - the main SfM engine
sudo apt-get install colmap

# Point Cloud Library tools - for format conversion
sudo apt-get install pcl-tools

# Meshlab - for point cloud visualization
sudo apt-get install meshlab

# Python dependencies
pip3 install pyyaml opencv-python
```

## Quick Start

### Option 1: Using the Shell Script (Recommended)

The easiest way to run the complete pipeline:

```bash
bash src/sfm_reconstruction/scripts/sfm_pipeline.sh \
    -b /path/to/rosbag.db3 \
    -o ./reconstruction \
    -i /camera/color/image_raw \
    -c /camera/color/camera_info
```

### Option 2: Using ROS2 CLI

**Step 1: Extract images from rosbag**

```bash
ros2 run sfm_reconstruction bag_extractor /path/to/rosbag.db3 \
    -o ./sfm_data \
    -i /camera/color/image_raw \
    -c /camera/color/camera_info
```

**Step 2: Run SfM reconstruction**

```bash
ros2 run sfm_reconstruction sfm_processor ./sfm_data
```

### Option 3: Using Launch Files

```bash
ros2 launch sfm_reconstruction bag_extraction.launch.py \
    bag_path:=/path/to/rosbag.db3 \
    output_dir:=./sfm_data \
    image_topic:=/camera/color/image_raw

ros2 launch sfm_reconstruction sfm_reconstruction.launch.py \
    project_dir:=./sfm_data
```

## Detailed Usage

### Image Extraction

Extract images and camera calibration from a rosbag:

```bash
ros2 run sfm_reconstruction bag_extractor \
    <rosbag_path> \
    -o <output_directory> \
    -i <image_topic> \
    -c <camera_info_topic> \
    [--no-camera-info]
```

**Arguments:**
- `rosbag_path`: Path to ROS2 rosbag (required)
- `-o, --output`: Output directory (default: `./sfm_data`)
- `-i, --image-topic`: Image ROS topic (default: `/camera/color/image_raw`)
- `-c, --camera-info-topic`: Camera info topic (default: `/camera/color/camera_info`)
- `--no-camera-info`: Skip camera info extraction

**Output:**
- `images/`: Extracted PNG frames
- `camera_info.yaml`: Camera intrinsics in YAML format
- `camera_info.json`: Camera intrinsics in JSON format
- `cameras.txt`: COLMAP format camera parameters

### SfM Processing

Run Structure-from-Motion reconstruction:

```bash
ros2 run sfm_reconstruction sfm_processor \
    <project_directory> \
    -m [exhaustive|sequential] \
    --sparse-only
```

**Arguments:**
- `project_directory`: Directory with `images/` subdirectory (required)
- `-m, --matcher`: Feature matching method:
  - `exhaustive`: Matches all image pairs (slower, more accurate)
  - `sequential`: Matches only temporally adjacent images (faster)
- `--sparse-only`: Only compute sparse point cloud (skip dense)
- `--dense-only`: Skip to dense reconstruction (assumes sparse exists)

**Output:**
- `sparse/0/`: Sparse reconstruction (points3D.txt, cameras.txt, images.txt)
- `dense/`: Dense reconstruction data
- `dense/fused.ply`: Final dense point cloud (if dense reconstruction enabled)

## Common Workflows

### Workflow 1: Quick Sparse Reconstruction (Fast)

For quick preview of structure:

```bash
ros2 run sfm_reconstruction bag_extractor my_bag.db3 -o ./sfm_data -i /oak/rgb/image_raw

ros2 run sfm_reconstruction sfm_processor ./sfm_data --sparse-only

# View sparse points
meshlab ./sfm_data/sparse/0/points3D.txt
```

**Time:** ~5-10 minutes for 100 images

### Workflow 2: High-Quality Dense Reconstruction (Thorough)

For detailed dense point clouds:

```bash
ros2 run sfm_reconstruction bag_extractor my_bag.db3 -o ./sfm_data

ros2 run sfm_reconstruction sfm_processor ./sfm_data

# Convert and view
pcl_ply2pcd ./sfm_data/dense/fused.ply ./cloud.pcd
pcl_viewer ./cloud.pcd
```

**Time:** ~30-60 minutes for 100 images

### Workflow 3: With Custom Camera Topics

For OAK-D cameras:

```bash
bash src/sfm_reconstruction/scripts/sfm_pipeline.sh \
    -b my_bag.db3 \
    -o ./reconstruction \
    -i /oak/rgb/image_raw \
    -c /oak/rgb/camera_info \
    -m sequential  # Sequential is faster for video-like sequences
```

## Understanding Camera Topics

Common camera ROS topics:

| Camera | Image Topic | Camera Info Topic |
|--------|------------|-----------------|
| Intel RealSense D435 | `/camera/color/image_raw` | `/camera/color/camera_info` |
| OAK-D RGB | `/oak/rgb/image_raw` | `/oak/rgb/camera_info` |
| Generic USB Camera | `/camera/image_raw` | `/camera/camera_info` |
| Realsense Depth | `/camera/depth/image_rect_raw` | `/camera/depth/camera_info` |

**To find the correct topics:**

```bash
# List all topics in rosbag
ros2 bag info /path/to/bag

# Or play bag and list active topics
ros2 bag play /path/to/bag &
ros2 topic list
ros2 topic info <topic_name>
```

## Point Cloud Visualization and Conversion

### View PLY Point Cloud (Native COLMAP Format)

```bash
# Using Meshlab (GUI)
meshlab ./sfm_data/dense/fused.ply

# Using PCL command line
pcl_viewer ./sfm_data/dense/fused.ply
```

### Convert to PCD Format

PCL's native format, useful for integration with other ROS2 tools:

```bash
pcl_ply2pcd ./sfm_data/dense/fused.ply ./cloud.pcd
```

### Convert to XYZ ASCII

Simple text format for other tools:

```bash
pcl_ply2pcd ./sfm_data/dense/fused.ply ./cloud.pcd
pcl_pcd2xyz ./cloud.pcd ./cloud.xyz
```

### Integrate with ROS2

Create a node to publish point clouds:

```python
import rclpy
from sensor_msgs.msg import PointCloud2
from pcl_msgs import PointCloud2 as PCL_PointCloud2
import pcl

node = rclpy.create_node('point_cloud_publisher')
pub = node.create_publisher(PointCloud2, '/reconstruction/cloud', 10)

cloud = pcl.load('./cloud.pcd')
msg = pcl.to_ros_msg(cloud)
pub.publish(msg)
```

## Advanced Options

### Feature Matching Methods

**Exhaustive Matching (Default)**
- Matches features between all image pairs
- Higher accuracy
- Much slower: O(n²) comparisons
- Best for: Small sequences (<100 images), high-quality results

**Sequential Matching**
- Only matches consecutive/nearby frames
- Much faster: O(n) comparisons
- Good for: Long sequences, video-like data
- Use flag: `-m sequential`

```bash
# For a video-like sequence (many frames of same scene)
ros2 run sfm_reconstruction sfm_processor ./sfm_data -m sequential
```

### Camera Models

COLMAP supports multiple camera models. Default is PINHOLE (standard perspective):

- `PINHOLE`: Standard pinhole camera model (recommended)
- `SIMPLE_PINHOLE`: Pinhole without separate fx/fy
- `OPENCV`: With radial/tangential distortion
- `FULL_OPENCV`: Full OpenCV distortion model
- `FISHEYE`: Fisheye/wide-angle cameras

## Troubleshooting

### Issue: "No images extracted from rosbag"

Check available topics:
```bash
ros2 bag info /path/to/bag.db3

# Then verify topic exists:
ros2 bag play /path/to/bag.db3 &
ros2 topic list | grep image
```

Use correct topic with `-i` flag.

### Issue: "Feature extraction succeeded but no matches found"

**Causes:**
- Images lack distinctive features (blank walls, poor texture)
- Camera moved too fast between frames
- Too much motion blur
- Images are too different from each other

**Solutions:**
- Ensure camera moves slowly with good overlap
- Record in well-lit environments
- Avoid fast pans or rotations

### Issue: "Feature matching failed - no database"

Run feature extraction first:
```bash
# Make sure you're in the right directory
ros2 run sfm_reconstruction sfm_processor ./sfm_data
# Don't skip feature extraction
```

### Issue: "Sparse reconstruction produced no points"

The initial pair selection failed. This means:
- Images don't have enough distinctive features
- Camera didn't move significantly between images
- Images are too different in appearance

Try:
- Record with slower camera movement
- Ensure good lighting and texture
- Verify camera intrinsics are correct

### Issue: "Dense reconstruction failed or very slow"

Dense reconstruction is computationally expensive. Options:

1. **Reduce image resolution** (before feature extraction):
```bash
# Manually resize images in the images/ directory
```

2. **Use sparse-only mode**:
```bash
ros2 run sfm_reconstruction sfm_processor ./sfm_data --sparse-only
```

3. **Increase timeout** if using scripts

### Issue: "COLMAP not found"

Install COLMAP:
```bash
sudo apt-get update
sudo apt-get install colmap
```

Verify installation:
```bash
colmap --help
```

## File Structure

```
sfm_reconstruction/
├── sfm_reconstruction/
│   ├── __init__.py
│   ├── bag_extractor.py          # Rosbag → images extraction
│   ├── sfm_processor.py           # COLMAP wrapper for SfM
│   └── utils.py                   # Utility functions
├── launch/
│   ├── bag_extraction.launch.py   # Launch image extraction
│   └── sfm_reconstruction.launch.py # Launch SfM processing
├── scripts/
│   ├── sfm_pipeline.sh            # Complete pipeline script
│   ├── install_dependencies.sh    # Install required tools
│   └── test_pipeline.sh           # Test with synthetic data
├── config/
│   ├── camera_configs.yaml        # Camera topic configurations
│   └── colmap_params.yaml         # COLMAP parameters
├── package.xml
├── setup.py
└── README.md
```

## Output Structure

After running the pipeline:

```
sfm_data/
├── images/                        # Extracted camera frames
│   ├── 1234567890123456.png
│   ├── 1234567890123457.png
│   └── ...
├── camera_info.yaml              # Camera intrinsics (YAML)
├── camera_info.json              # Camera intrinsics (JSON)
├── cameras.txt                   # Camera parameters (COLMAP format)
├── database.db                   # COLMAP feature database
├── sparse/                       # Sparse reconstruction
│   └── 0/
│       ├── cameras.txt
│       ├── images.txt
│       ├── points3D.txt          # Sparse point cloud
│       └── project.ini
└── dense/                        # Dense reconstruction
    ├── images/                   # Undistorted images
    ├── stereo/                   # Depth maps per image
    ├── fused.ply                 # Dense point cloud (final output!)
    └── consistency_graph.bin
```

## Performance Notes

### Approximate Runtimes (on Jetson AGX Orin)

| Stage | 50 Images | 100 Images | 200 Images |
|-------|-----------|-----------|-----------|
| Feature Extraction | 30s | 60s | 2m |
| Feature Matching (Exhaustive) | 2m | 8m | 30m |
| Sparse Reconstruction | 1m | 2m | 5m |
| Image Undistortion | 30s | 1m | 2m |
| Dense Stereo | 5m | 15m | 45m |
| Stereo Fusion | 1m | 2m | 5m |
| **Total (Dense)** | **10m** | **30m** | **90m** |
| **Total (Sparse Only)** | **4m** | **11m** | **40m** |

**Tips for faster processing:**
- Use `--sparse-only` for quick results
- Use `-m sequential` for video sequences
- Reduce image resolution before processing
- Use `--matcher sequential` instead of exhaustive

## Integration with Other v4ar Modules

### Publishing Reconstructed Point Cloud

Create a ROS2 node to publish the reconstructed point cloud:

```python
# In your perception module
import rclpy
from sensor_msgs.msg import PointCloud2
import open3d as o3d

class PointCloudPublisher(rclpy.Node):
    def __init__(self):
        super().__init__('point_cloud_pub')
        self.pub = self.create_publisher(PointCloud2, '/reconstruction/points', 10)
        
        # Load reconstructed point cloud
        pcd = o3d.io.read_point_cloud('./sfm_data/dense/fused.ply')
        
        # Convert and publish
        self.publish_cloud(pcd)
```

### Using for SLAM/Mapping

The reconstructed point cloud can be used as:
- **Loop closure detection**: Compare new observations with reconstruction
- **Place recognition**: Localize in reconstructed environment
- **Dense mapping**: Fuse with RTAB-Map for improved maps

## References

- **COLMAP**: https://colmap.github.io/
- **SfM Paper**: Schönberger & Frahm (2016) - "Structure-from-Motion Revisited"
- **PCL Tools**: https://pointclouds.org/
- **Open3D**: http://www.open3d.org/

## Future Enhancements

- [ ] Integration with depth sensors for hybrid SfM
- [ ] GPU acceleration for feature matching
- [ ] Multi-view stereo fusion with depth data
- [ ] Real-time incremental reconstruction
- [ ] Loop closure detection
- [ ] Bundle adjustment refinement
- [ ] Mesh generation from point cloud

## License

Apache License 2.0

## Contributing

To contribute improvements:
1. Test with your camera/rosbag
2. Document any new features
3. Update configuration examples
4. Submit pull requests to the main branch

## Support

For issues:
1. Check the **Troubleshooting** section above
2. Verify COLMAP installation: `colmap --help`
3. Check rosbag topics: `ros2 bag info`
4. Review log messages for specific errors
