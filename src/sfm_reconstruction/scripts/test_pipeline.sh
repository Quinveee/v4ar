#!/bin/bash
# Quick test of SfM pipeline with synthetic data

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEST_DATA_DIR="${TEST_DATA_DIR:-./.test_data}"
TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR:-./.test_output}"

echo "Creating test data directory..."
mkdir -p "$TEST_DATA_DIR/images"

# Create some simple test images (requires ImageMagick)
if ! command -v convert &> /dev/null; then
    echo "ImageMagick not found. Install with: sudo apt install imagemagick"
    exit 1
fi

echo "Generating test images..."
for i in {1..5}; do
    convert -size 640x480 \
        xc:white \
        -draw "circle 320,240 340,240" \
        -draw "line 0,240 640,240" \
        -draw "line 320,0 320,480" \
        "$TEST_DATA_DIR/images/test_$i.png"
    echo "  Created test_$i.png"
done

echo ""
echo "Running SfM pipeline on test data..."
echo "  Input: $TEST_DATA_DIR"
echo "  Output: $TEST_OUTPUT_DIR"

ros2 run sfm_reconstruction sfm_processor "$TEST_OUTPUT_DIR" -m exhaustive

echo ""
echo "Test complete!"
