"""
Utility functions for SfM pipeline.
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def ply_to_pcd(ply_path: str, output_pcd: Optional[str] = None) -> Optional[str]:
    """
    Convert PLY point cloud to PCD format.
    
    Args:
        ply_path: Path to PLY file
        output_pcd: Output PCD path (default: replace .ply with .pcd)
        
    Returns:
        Path to PCD file if successful, None otherwise
    """
    ply_path = Path(ply_path)
    
    if not ply_path.exists():
        logger.error(f"PLY file not found: {ply_path}")
        return None
    
    if output_pcd is None:
        output_pcd = ply_path.with_suffix('.pcd')
    
    try:
        cmd = ['pcl_ply2pcd', str(ply_path), str(output_pcd)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info(f"✓ Converted to PCD: {output_pcd}")
            return str(output_pcd)
        else:
            logger.error(f"Conversion failed: {result.stderr}")
            return None
    except FileNotFoundError:
        logger.error("pcl_ply2pcd not found. Install with: sudo apt install pcl-tools")
        return None
    except Exception as e:
        logger.error(f"Conversion error: {e}")
        return None


def visualize_ply(ply_path: str) -> bool:
    """
    Visualize PLY point cloud with Meshlab.
    
    Args:
        ply_path: Path to PLY file
        
    Returns:
        Success status
    """
    ply_path = Path(ply_path)
    
    if not ply_path.exists():
        logger.error(f"PLY file not found: {ply_path}")
        return False
    
    try:
        subprocess.Popen(['meshlab', str(ply_path)])
        logger.info(f"Meshlab opened with: {ply_path}")
        return True
    except FileNotFoundError:
        logger.error("Meshlab not found. Install with: sudo apt install meshlab")
        return False
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        return False


def check_dependencies() -> Dict[str, bool]:
    """
    Check if all required tools are available.
    
    Returns:
        Dictionary with tool availability status
    """
    tools = {
        'colmap': 'colmap',
        'pcl_ply2pcd': 'pcl_ply2pcd',
        'meshlab': 'meshlab',
    }
    
    status = {}
    for name, cmd in tools.items():
        try:
            subprocess.run([cmd, '--help'], capture_output=True, timeout=5)
            status[name] = True
            logger.info(f"✓ {name} found")
        except FileNotFoundError:
            status[name] = False
            logger.warning(f"✗ {name} not found")
        except Exception:
            status[name] = False
    
    return status
