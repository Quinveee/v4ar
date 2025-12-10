"""
COLMAP-based SfM processor for reconstructing 3D point clouds.

This module handles feature extraction, matching, sparse reconstruction,
and dense reconstruction using COLMAP.
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from typing import Dict, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class COLMAPProcessor:
    """Handle COLMAP-based Structure-from-Motion processing."""

    def __init__(self, project_dir: str):
        """
        Initialize COLMAP processor.
        
        Args:
            project_dir: Path to the SfM project directory
        """
        self.project_dir = Path(project_dir)
        self.images_dir = self.project_dir / "images"
        self.sparse_dir = self.project_dir / "sparse"
        self.dense_dir = self.project_dir / "dense"
        self.database_path = self.project_dir / "database.db"
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        self.dense_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"COLMAP processor initialized: {project_dir}")
        self._check_colmap()

    def _check_colmap(self) -> bool:
        """Check if COLMAP is installed."""
        try:
            result = subprocess.run(['colmap'], capture_output=True, text=True)
            logger.info("✓ COLMAP is installed")
            return True
        except FileNotFoundError:
            logger.error("✗ COLMAP not found. Install with: sudo apt install colmap")
            return False

    def feature_extraction(self, camera_model: str = "PINHOLE", 
                          single_camera: bool = True) -> bool:
        """
        Extract SIFT features from images.
        
        Args:
            camera_model: Camera model (PINHOLE, SIMPLE_PINHOLE, etc.)
            single_camera: Assume all images from same camera
            
        Returns:
            Success status
        """
        logger.info("Starting feature extraction...")
        
        cmd = [
            'colmap', 'feature_extractor',
            '--database_path', str(self.database_path),
            '--image_path', str(self.images_dir),
            '--ImageReader.single_camera', '1' if single_camera else '0',
            '--ImageReader.camera_model', camera_model,
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                logger.info("✓ Feature extraction complete")
                return True
            else:
                logger.error(f"Feature extraction failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("Feature extraction timed out")
            return False
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return False

    def feature_matching(self, matcher_type: str = "exhaustive") -> bool:
        """
        Match features between images.
        
        Args:
            matcher_type: 'exhaustive' or 'sequential'
            
        Returns:
            Success status
        """
        logger.info(f"Starting feature matching ({matcher_type})...")
        
        cmd = [
            f'colmap {matcher_type}_matcher',
            '--database_path', str(self.database_path),
        ]
        
        if matcher_type == "sequential":
            cmd.extend(['--SequentialMatching.overlap', '4'])
        
        try:
            result = subprocess.run(' '.join(cmd), shell=True, capture_output=True, 
                                  text=True, timeout=3600)
            if result.returncode == 0:
                logger.info(f"✓ Feature matching complete")
                return True
            else:
                logger.error(f"Feature matching failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("Feature matching timed out")
            return False
        except Exception as e:
            logger.error(f"Feature matching error: {e}")
            return False

    def sparse_reconstruction(self) -> bool:
        """
        Perform sparse 3D reconstruction.
        
        Returns:
            Success status
        """
        logger.info("Starting sparse reconstruction (mapper)...")
        
        cmd = [
            'colmap', 'mapper',
            '--database_path', str(self.database_path),
            '--image_path', str(self.images_dir),
            '--output_path', str(self.sparse_dir),
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                logger.info("✓ Sparse reconstruction complete")
                logger.info(f"Sparse point cloud saved to: {self.sparse_dir}/0")
                return True
            else:
                logger.error(f"Sparse reconstruction failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("Sparse reconstruction timed out")
            return False
        except Exception as e:
            logger.error(f"Sparse reconstruction error: {e}")
            return False

    def undistort_images(self) -> bool:
        """
        Undistort images for dense reconstruction.
        
        Returns:
            Success status
        """
        logger.info("Undistorting images...")
        
        sparse_model = self.sparse_dir / "0"
        if not sparse_model.exists():
            logger.error(f"Sparse model not found: {sparse_model}")
            return False
        
        cmd = [
            'colmap', 'image_undistorter',
            '--image_path', str(self.images_dir),
            '--input_path', str(sparse_model),
            '--output_path', str(self.dense_dir),
            '--output_type', 'COLMAP',
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode == 0:
                logger.info("✓ Image undistortion complete")
                return True
            else:
                logger.error(f"Image undistortion failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("Image undistortion timed out")
            return False
        except Exception as e:
            logger.error(f"Image undistortion error: {e}")
            return False

    def dense_stereo(self) -> bool:
        """
        Perform patch match stereo for dense depth estimation.
        
        Returns:
            Success status
        """
        logger.info("Running patch match stereo...")
        
        cmd = [
            'colmap', 'patch_match_stereo',
            '--workspace_path', str(self.dense_dir),
            '--PatchMatchStereo.geom_consistency', 'true',
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                logger.info("✓ Patch match stereo complete")
                return True
            else:
                logger.error(f"Patch match stereo failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("Patch match stereo timed out")
            return False
        except Exception as e:
            logger.error(f"Patch match stereo error: {e}")
            return False

    def stereo_fusion(self, output_name: str = "fused.ply") -> bool:
        """
        Fuse depth maps into a point cloud.
        
        Args:
            output_name: Output point cloud filename
            
        Returns:
            Success status
        """
        logger.info("Fusing depth maps into point cloud...")
        
        output_path = self.dense_dir / output_name
        
        cmd = [
            'colmap', 'stereo_fusion',
            '--workspace_path', str(self.dense_dir),
            '--output_path', str(output_path),
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode == 0:
                logger.info(f"✓ Stereo fusion complete")
                logger.info(f"Dense point cloud saved to: {output_path}")
                return True
            else:
                logger.error(f"Stereo fusion failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("Stereo fusion timed out")
            return False
        except Exception as e:
            logger.error(f"Stereo fusion error: {e}")
            return False

    def full_pipeline(self, matcher_type: str = "exhaustive",
                      camera_model: str = "PINHOLE") -> Dict:
        """
        Run complete SfM pipeline.
        
        Args:
            matcher_type: 'exhaustive' or 'sequential'
            camera_model: Camera model type
            
        Returns:
            Dictionary with pipeline results
        """
        logger.info("=" * 60)
        logger.info("STARTING FULL SfM PIPELINE")
        logger.info("=" * 60)
        
        results = {
            "feature_extraction": False,
            "feature_matching": False,
            "sparse_reconstruction": False,
            "image_undistortion": False,
            "dense_stereo": False,
            "stereo_fusion": False,
            "output_point_cloud": None
        }
        
        # Feature Extraction
        if not self.feature_extraction(camera_model):
            logger.error("Pipeline failed at feature extraction")
            return results
        results["feature_extraction"] = True
        
        # Feature Matching
        if not self.feature_matching(matcher_type):
            logger.error("Pipeline failed at feature matching")
            return results
        results["feature_matching"] = True
        
        # Sparse Reconstruction
        if not self.sparse_reconstruction():
            logger.error("Pipeline failed at sparse reconstruction")
            return results
        results["sparse_reconstruction"] = True
        
        # Image Undistortion
        if not self.undistort_images():
            logger.error("Pipeline failed at image undistortion")
            return results
        results["image_undistortion"] = True
        
        # Dense Stereo
        if not self.dense_stereo():
            logger.error("Pipeline failed at dense stereo")
            return results
        results["dense_stereo"] = True
        
        # Stereo Fusion
        if not self.stereo_fusion():
            logger.error("Pipeline failed at stereo fusion")
            return results
        results["stereo_fusion"] = True
        
        # Check output
        output_cloud = self.dense_dir / "fused.ply"
        if output_cloud.exists():
            results["output_point_cloud"] = str(output_cloud)
            logger.info(f"✓ Point cloud generated: {output_cloud}")
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        
        return results


def main():
    """Main entry point for SfM processing."""
    parser = argparse.ArgumentParser(
        description="Run COLMAP-based Structure-from-Motion reconstruction"
    )
    parser.add_argument(
        'project_dir',
        help='Project directory with images subdirectory'
    )
    parser.add_argument(
        '-m', '--matcher',
        choices=['exhaustive', 'sequential'],
        default='exhaustive',
        help='Feature matcher type (default: exhaustive)'
    )
    parser.add_argument(
        '--sparse-only',
        action='store_true',
        help='Only perform sparse reconstruction (skip dense)'
    )
    parser.add_argument(
        '--dense-only',
        action='store_true',
        help='Skip to dense reconstruction (assumes sparse exists)'
    )
    parser.add_argument(
        '-c', '--camera-model',
        default='PINHOLE',
        help='Camera model (default: PINHOLE)'
    )
    
    args = parser.parse_args()
    
    try:
        processor = COLMAPProcessor(args.project_dir)
        
        if args.dense_only:
            logger.info("Running dense reconstruction phase...")
            success = processor.undistort_images() and \
                     processor.dense_stereo() and \
                     processor.stereo_fusion()
        else:
            results = processor.full_pipeline(args.matcher, args.camera_model)
            
            if args.sparse_only:
                logger.info("Sparse-only mode: skipping dense reconstruction")
                return 0 if results["sparse_reconstruction"] else 1
            
            success = all([
                results["feature_extraction"],
                results["feature_matching"],
                results["sparse_reconstruction"],
                results["stereo_fusion"]
            ])
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
