"""
Rosbag to image and camera info extractor for SfM.

This module extracts image frames and camera intrinsics from a ROS2 rosbag
and prepares them for Structure-from-Motion processing.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict
import json
import logging

try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge
    import cv2
    import yaml
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure rosbag2, cv_bridge, and OpenCV are installed:")
    print("  sudo apt install ros-humble-rosbag2 ros-humble-cv-bridge python3-opencv")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RosbagExtractor:
    """Extract images and camera info from ROS2 rosbags."""

    def __init__(self, bag_path: str, output_dir: str):
        """
        Initialize the extractor.
        
        Args:
            bag_path: Path to the ROS2 rosbag
            output_dir: Directory to save extracted data
        """
        self.bag_path = Path(bag_path)
        self.output_dir = Path(output_dir)
        self.bridge = CvBridge()
        self.frame_count = 0
        self.camera_info = None
        
        # Validate input
        if not self.bag_path.exists():
            raise FileNotFoundError(f"Rosbag not found: {bag_path}")
        
        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Extractor initialized: {bag_path} → {output_dir}")

    def extract(self, image_topic: str, camera_info_topic: Optional[str] = None) -> Dict:
        """
        Extract images and camera info from rosbag.
        
        Args:
            image_topic: ROS topic for image messages (e.g., "/camera/color/image_raw")
            camera_info_topic: ROS topic for camera info (e.g., "/camera/color/camera_info")
            
        Returns:
            Dictionary with extraction statistics
        """
        logger.info(f"Extracting images from topic: {image_topic}")
        if camera_info_topic:
            logger.info(f"Extracting camera info from topic: {camera_info_topic}")
        
        stats = {
            "frames_extracted": 0,
            "camera_info_found": False,
            "output_directory": str(self.output_dir)
        }
        
        try:
            reader = SequentialReader()
            reader.open(
                StorageOptions(uri=str(self.bag_path), storage_id='sqlite3'),
                ConverterOptions('', '')
            )
            
            # Extract camera topics info first
            topic_types = reader.get_all_topics_and_types()
            available_topics = {topic.name: topic.type for topic in topic_types}
            
            logger.info(f"Available topics: {list(available_topics.keys())}")
            
            # Validate requested topics exist
            if image_topic not in available_topics:
                logger.error(f"Image topic '{image_topic}' not found in rosbag")
                logger.info(f"Available topics: {list(available_topics.keys())}")
                return stats
            
            if camera_info_topic and camera_info_topic not in available_topics:
                logger.warning(f"Camera info topic '{camera_info_topic}' not found")
                camera_info_topic = None
            
            # Read messages
            while reader.has_next():
                topic, data, timestamp = reader.read_next()
                
                # Extract images
                if topic == image_topic:
                    try:
                        msg = Image()
                        msg.deserialize(data)
                        cv_img = self.bridge.imgmsg_to_cv2(msg)
                        
                        # Save image with timestamp as filename
                        output_path = self.images_dir / f"{timestamp:020d}.png"
                        cv2.imwrite(str(output_path), cv_img)
                        
                        self.frame_count += 1
                        if self.frame_count % 10 == 0:
                            logger.info(f"Extracted {self.frame_count} frames...")
                            
                    except Exception as e:
                        logger.warning(f"Failed to process image at {timestamp}: {e}")
                        continue
                
                # Extract camera info (only once)
                if camera_info_topic and topic == camera_info_topic and not self.camera_info:
                    try:
                        msg = CameraInfo()
                        msg.deserialize(data)
                        self.camera_info = msg
                        stats["camera_info_found"] = True
                        logger.info("Camera info extracted")
                    except Exception as e:
                        logger.warning(f"Failed to process camera info: {e}")
            
            reader.close()
            stats["frames_extracted"] = self.frame_count
            
            logger.info(f"✓ Extraction complete: {self.frame_count} frames saved")
            
            # Save camera info if found
            if self.camera_info:
                self._save_camera_info()
            
            return stats
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise

    def _save_camera_info(self) -> None:
        """Save camera intrinsics to YAML and JSON formats."""
        if not self.camera_info:
            logger.warning("No camera info to save")
            return
        
        # Save as YAML
        camera_info_dict = {
            'image_width': self.camera_info.width,
            'image_height': self.camera_info.height,
            'camera_name': self.camera_info.header.frame_id or 'camera',
            'camera_matrix': {
                'rows': 3,
                'cols': 3,
                'data': list(self.camera_info.k)
            },
            'distortion_coefficients': {
                'rows': 1,
                'cols': len(self.camera_info.d),
                'data': list(self.camera_info.d)
            },
            'rectification_matrix': {
                'rows': 3,
                'cols': 3,
                'data': list(self.camera_info.r) if self.camera_info.r else [1, 0, 0, 0, 1, 0, 0, 0, 1]
            },
            'projection_matrix': {
                'rows': 3,
                'cols': 4,
                'data': list(self.camera_info.p)
            }
        }
        
        yaml_path = self.output_dir / "camera_info.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(camera_info_dict, f, default_flow_style=False)
        logger.info(f"Camera info saved: {yaml_path}")
        
        # Save as JSON for easier parsing
        json_path = self.output_dir / "camera_info.json"
        with open(json_path, 'w') as f:
            json.dump(camera_info_dict, f, indent=2)
        logger.info(f"Camera info saved: {json_path}")
        
        # Save COLMAP format
        self._save_colmap_cameras(camera_info_dict)

    def _save_colmap_cameras(self, camera_info_dict: Dict) -> None:
        """Save camera intrinsics in COLMAP format."""
        # Extract intrinsics
        K = camera_info_dict['camera_matrix']['data']
        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]
        width = camera_info_dict['image_width']
        height = camera_info_dict['image_height']
        
        # COLMAP cameras.txt format:
        # IMAGE_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        colmap_path = self.output_dir / "cameras.txt"
        with open(colmap_path, 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")
        logger.info(f"COLMAP cameras.txt saved: {colmap_path}")


def main():
    """Main entry point for bag extraction."""
    parser = argparse.ArgumentParser(
        description="Extract images and camera info from ROS2 rosbag for SfM"
    )
    parser.add_argument(
        'bag_path',
        help='Path to the ROS2 rosbag'
    )
    parser.add_argument(
        '-o', '--output',
        default='./sfm_data',
        help='Output directory (default: ./sfm_data)'
    )
    parser.add_argument(
        '-i', '--image-topic',
        default='/camera/color/image_raw',
        help='Image topic (default: /camera/color/image_raw)'
    )
    parser.add_argument(
        '-c', '--camera-info-topic',
        default='/camera/color/camera_info',
        help='Camera info topic (default: /camera/color/camera_info)'
    )
    parser.add_argument(
        '--no-camera-info',
        action='store_true',
        help='Skip camera info extraction'
    )
    
    args = parser.parse_args()
    
    try:
        extractor = RosbagExtractor(args.bag_path, args.output)
        camera_info_topic = None if args.no_camera_info else args.camera_info_topic
        stats = extractor.extract(args.image_topic, camera_info_topic)
        
        logger.info("=" * 60)
        logger.info("EXTRACTION SUMMARY")
        logger.info("=" * 60)
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        logger.info("=" * 60)
        
        if stats['frames_extracted'] == 0:
            logger.warning("No frames extracted! Check your image topic.")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
