# ~/slam_test/src/var_mapping/var_mapping/utils/visualization.py

"""Visualization utilities for displaying RTAB-Map point clouds, camera poses, and sample frames.

Usage: ros2 run var_mapping visualize_point_cloud

Arguments:
    --data-dir: Path to data directory containing frames, point_cloud.pcd, and poses.txt
    --pcd: Path to point cloud file (.pcd)
    --poses: Path to camera poses file
    --frames: Path to frames directory
    --num-samples: Number of sample frames to display

Example:
    ros2 run var_mapping visualize_point_cloud --data-dir ~/slam_test/data/session1 --pcd ~/slam_test/data/session1/point_cloud.pcd --poses ~/slam_test/data/session1/poses.txt --frames ~/slam_test/data/session1/frames --num-samples 6
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_point_cloud(pcd_path):
    """Visualize point cloud using Open3D"""
    print(f"Loading point cloud from: {pcd_path}")
    
    # Load point cloud
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    
    print(f"Point cloud has {len(pcd.points)} points")
    
    # Visualize
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="RTAB-Map Point Cloud",
        width=1024,
        height=768,
        point_show_normal=False
    )

def visualize_camera_poses(poses_path):
    """Visualize camera trajectory"""
    print(f"Loading camera poses from: {poses_path}")
    
    # Load poses (format depends on export)
    # Typically: timestamp x y z qx qy qz qw
    poses = np.loadtxt(poses_path)
    
    # Extract positions
    positions = poses[:, 1:4]  # x, y, z columns
    
    # Plot trajectory
    fig = plt.figure(figsize=(12, 8))
    
    # 3D plot
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
               c='g', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
               c='r', s=100, label='End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Camera Trajectory (3D)')
    ax.legend()
    
    # Top-down view
    ax2 = fig.add_subplot(122)
    ax2.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
    ax2.scatter(positions[0, 0], positions[0, 1], c='g', s=100, label='Start')
    ax2.scatter(positions[-1, 0], positions[-1, 1], c='r', s=100, label='End')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Camera Trajectory (Top View)')
    ax2.axis('equal')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('camera_trajectory.png', dpi=300)
    plt.show()
    
    print(f"Trajectory has {len(positions)} poses")
    print(f"Total distance traveled: {np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)):.2f} m")

def display_sample_frames(frames_dir, num_samples=6):
    """Display sample extracted frames"""
    frames_dir = Path(frames_dir)
    frames = sorted(frames_dir.glob('*.jpg'))
    
    if len(frames) == 0:
        print("No frames found!")
        return
    
    # Sample evenly
    indices = np.linspace(0, len(frames)-1, num_samples, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, frame_idx in enumerate(indices):
        img = plt.imread(frames[frame_idx])
        axes[idx].imshow(img)
        axes[idx].set_title(f'Frame {frame_idx}/{len(frames)}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_frames.png', dpi=300)
    plt.show()
    
    print(f"Total frames extracted: {len(frames)}")

def main(args=None):
    """
    Main function for ROS 2 entry point.
    
    Can be run with: ros2 run var_mapping visualize_point_cloud
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualize RTAB-Map point clouds, camera poses, and sample frames'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=str(Path.home() / 'slam_test' / 'data' / 'session1'),
        help='Path to data directory containing frames, point_cloud.pcd, and poses.txt'
    )
    parser.add_argument(
        '--pcd',
        type=str,
        help='Path to point cloud file (.pcd)'
    )
    parser.add_argument(
        '--poses',
        type=str,
        help='Path to camera poses file'
    )
    parser.add_argument(
        '--frames',
        type=str,
        help='Path to frames directory'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=6,
        help='Number of sample frames to display'
    )
    
    # Parse arguments (handle ROS 2 args format)
    if args is not None:
        # Remove '--ros-args' and related ROS 2 arguments
        filtered_args = [arg for arg in args if not arg.startswith('--ros-args')]
        args = parser.parse_args(filtered_args)
    else:
        args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Display sample frames if frames directory exists or specified
    frames_dir = Path(args.frames) if args.frames else data_dir / 'frames'
    if frames_dir.exists():
        display_sample_frames(frames_dir, args.num_samples)
    
    # Visualize point cloud if file exists or specified
    pcd_path = Path(args.pcd) if args.pcd else data_dir / 'point_cloud.pcd'
    if pcd_path.exists():
        visualize_point_cloud(pcd_path)
    else:
        print(f"Point cloud file not found: {pcd_path}")
    
    # Visualize camera poses if file exists or specified
    poses_path = Path(args.poses) if args.poses else data_dir / 'poses.txt'
    if poses_path.exists():
        visualize_camera_poses(poses_path)
    else:
        print(f"Camera poses file not found: {poses_path}")


# Example usage
if __name__ == '__main__':
    main()