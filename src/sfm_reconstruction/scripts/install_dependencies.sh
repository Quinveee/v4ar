#!/bin/bash
# Install dependencies for SfM pipeline

set -e

echo "Installing SfM dependencies..."

# Check if running with sudo
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run with sudo"
   exit 1
fi

echo "Updating package manager..."
apt-get update

echo "Installing COLMAP..."
apt-get install -y colmap

echo "Installing point cloud tools..."
apt-get install -y pcl-tools

echo "Installing visualization tools..."
apt-get install -y meshlab

echo "Installing Python dependencies..."
pip3 install pyyaml opencv-python

echo -e "\n✓ All dependencies installed successfully!"
echo ""
echo "To verify installation, run:"
echo "  colmap --help"
echo "  pcl_ply2pcd"
echo "  meshlab"
