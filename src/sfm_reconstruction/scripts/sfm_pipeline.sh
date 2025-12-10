#!/bin/bash
# Complete SfM pipeline script: rosbag → images → SfM → point cloud

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Complete SfM pipeline: extract images from rosbag and reconstruct 3D model

Options:
    -b, --bag PATH              Path to ROS2 rosbag (required)
    -o, --output DIR            Output directory (default: ./sfm_output)
    -i, --image-topic TOPIC     Image topic (default: /camera/color/image_raw)
    -c, --camera-info TOPIC     Camera info topic (default: /camera/color/camera_info)
    --skip-camera-info          Skip camera info extraction
    -m, --matcher TYPE          Feature matcher: exhaustive|sequential (default: exhaustive)
    --sparse-only               Only run sparse reconstruction
    --help                      Show this help message

Example:
    $0 -b my_bag.db3 -o ./reconstruction -i /oak/rgb/image_raw
EOF
    exit 1
}

# Defaults
BAG_PATH=""
OUTPUT_DIR="./sfm_output"
IMAGE_TOPIC="/camera/color/image_raw"
CAMERA_INFO_TOPIC="/camera/color/camera_info"
SKIP_CAMERA_INFO=false
MATCHER="exhaustive"
SPARSE_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--bag)
            BAG_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -i|--image-topic)
            IMAGE_TOPIC="$2"
            shift 2
            ;;
        -c|--camera-info)
            CAMERA_INFO_TOPIC="$2"
            shift 2
            ;;
        --skip-camera-info)
            SKIP_CAMERA_INFO=true
            shift
            ;;
        -m|--matcher)
            MATCHER="$2"
            shift 2
            ;;
        --sparse-only)
            SPARSE_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate inputs
if [[ -z "$BAG_PATH" ]]; then
    echo -e "${RED}Error: Rosbag path required (-b)${NC}"
    usage
fi

if [[ ! -f "$BAG_PATH" && ! -d "$BAG_PATH" ]]; then
    echo -e "${RED}Error: Rosbag not found: $BAG_PATH${NC}"
    exit 1
fi

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}SfM RECONSTRUCTION PIPELINE${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}Rosbag:${NC} $BAG_PATH"
echo -e "${GREEN}Output:${NC} $OUTPUT_DIR"
echo -e "${GREEN}Image Topic:${NC} $IMAGE_TOPIC"
echo -e "${GREEN}Matcher:${NC} $MATCHER"
echo -e "${BLUE}============================================================${NC}"

# Step 1: Extract images
echo -e "\n${YELLOW}[1/3] Extracting images from rosbag...${NC}"
EXTRACT_CMD="ros2 run sfm_reconstruction bag_extractor '$BAG_PATH' -o '$OUTPUT_DIR'"
EXTRACT_CMD="$EXTRACT_CMD -i '$IMAGE_TOPIC'"

if [[ "$SKIP_CAMERA_INFO" == false ]]; then
    EXTRACT_CMD="$EXTRACT_CMD -c '$CAMERA_INFO_TOPIC'"
else
    EXTRACT_CMD="$EXTRACT_CMD --no-camera-info"
fi

eval "$EXTRACT_CMD"

if [[ ! -d "$OUTPUT_DIR/images" ]]; then
    echo -e "${RED}Error: Image extraction failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Images extracted${NC}"

# Step 2: Run SfM
echo -e "\n${YELLOW}[2/3] Running Structure-from-Motion reconstruction...${NC}"
SFM_CMD="ros2 run sfm_reconstruction sfm_processor '$OUTPUT_DIR' -m '$MATCHER'"

if [[ "$SPARSE_ONLY" == true ]]; then
    SFM_CMD="$SFM_CMD --sparse-only"
fi

eval "$SFM_CMD"

# Step 3: Summary
echo -e "\n${BLUE}============================================================${NC}"
echo -e "${GREEN}PIPELINE COMPLETE${NC}"
echo -e "${BLUE}============================================================${NC}"

if [[ "$SPARSE_ONLY" == true ]]; then
    POINT_CLOUD="$OUTPUT_DIR/sparse/0/points3D.txt"
else
    POINT_CLOUD="$OUTPUT_DIR/dense/fused.ply"
fi

if [[ -f "$POINT_CLOUD" ]]; then
    echo -e "${GREEN}✓ Point cloud generated: $POINT_CLOUD${NC}"
    echo -e "\nNext steps:"
    echo -e "  View in Meshlab:    ${BLUE}meshlab '$POINT_CLOUD'${NC}"
    echo -e "  Convert to PCD:     ${BLUE}pcl_ply2pcd '$POINT_CLOUD' output.pcd${NC}"
    echo -e "  View with PCL:      ${BLUE}pcl_viewer output.pcd${NC}"
else
    echo -e "${RED}✗ Point cloud not generated${NC}"
    exit 1
fi
