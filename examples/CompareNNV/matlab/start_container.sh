#!/bin/bash
#
# Build and run the NNV Docker container for verification experiments.
#
# This script:
# 1. Builds the Docker image (if needed)
# 2. Runs the container with mounted volumes for models, samples, scripts, and outputs
# 3. Executes the verification experiments
# 4. Copies results back to the host
#
# Usage:
#   ./start_container.sh              # Run all experiments
#   ./start_container.sh --build-only # Only build the image
#   ./start_container.sh --shell      # Start interactive shell instead of running experiments

set -e

# Configuration
IMAGE_NAME="nnv-verification"
CONTAINER_NAME="nnv-verification-run"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE_NNV_DIR="$(dirname "$SCRIPT_DIR")"

# Use sudo for docker commands (required on most servers)
DOCKER="sudo docker"

# Parse arguments
BUILD_ONLY=false
INTERACTIVE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --shell)
            INTERACTIVE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--build-only] [--shell]"
            exit 1
            ;;
    esac
done

# Build the Docker image
echo "========================================"
echo "Building Docker image: $IMAGE_NAME"
echo "========================================"
$DOCKER build -t "$IMAGE_NAME" "$SCRIPT_DIR"

if [ "$BUILD_ONLY" = true ]; then
    echo "Build complete. Exiting."
    exit 0
fi

# Create output directory if it doesn't exist and make it world-writable
# so the container's matlab user can write to it
mkdir -p "$COMPARE_NNV_DIR/outputs/nnv"
chmod 777 "$COMPARE_NNV_DIR/outputs/nnv"

# Remove existing container if it exists
$DOCKER rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo ""
echo "========================================"
echo "Starting container: $CONTAINER_NAME"
echo "========================================"

if [ "$INTERACTIVE" = true ]; then
    # Interactive mode - start a shell
    echo "Starting interactive shell..."
    $DOCKER run -it --rm \
        --name "$CONTAINER_NAME" \
        -v "$COMPARE_NNV_DIR/models:/home/matlab/CompareNNV/models:ro" \
        -v "$COMPARE_NNV_DIR/samples:/home/matlab/CompareNNV/samples:ro" \
        -v "$SCRIPT_DIR/scripts:/home/matlab/CompareNNV/matlab/scripts:ro" \
        -v "$SCRIPT_DIR/run_all_nnv.m:/home/matlab/CompareNNV/matlab/run_all_nnv.m:ro" \
        -v "$SCRIPT_DIR/utils:/home/matlab/CompareNNV/matlab/utils:ro" \
        -v "$COMPARE_NNV_DIR/outputs/nnv:/home/matlab/CompareNNV/outputs/nnv" \
        -w /home/matlab/CompareNNV \
        "$IMAGE_NAME" \
        /bin/bash
else
    # Run experiments
    echo "Running NNV verification experiments..."
    echo ""

    # Run the container with mounted volumes
    # Note: We run as the container's default matlab user (not host user) because
    # MATLAB needs write access to its home directory for licensing/temp files.
    # The output directory is made world-writable above so matlab can write to it.
    $DOCKER run --rm \
        --name "$CONTAINER_NAME" \
        -v "$COMPARE_NNV_DIR/models:/home/matlab/CompareNNV/models:ro" \
        -v "$COMPARE_NNV_DIR/samples:/home/matlab/CompareNNV/samples:ro" \
        -v "$SCRIPT_DIR/scripts:/home/matlab/CompareNNV/matlab/scripts:ro" \
        -v "$SCRIPT_DIR/run_all_nnv.m:/home/matlab/CompareNNV/matlab/run_all_nnv.m:ro" \
        -v "$SCRIPT_DIR/utils:/home/matlab/CompareNNV/matlab/utils:ro" \
        -v "$COMPARE_NNV_DIR/outputs/nnv:/home/matlab/CompareNNV/outputs/nnv" \
        -w /home/matlab/CompareNNV \
        "$IMAGE_NAME" \
        matlab -nodisplay -r "run('/home/matlab/CompareNNV/matlab/run_all_nnv.m'); exit()"

    echo ""
    echo "========================================"
    echo "Experiments complete!"
    echo "========================================"
    echo ""
    echo "Results saved to: $COMPARE_NNV_DIR/outputs/nnv/"
    echo ""

    # List the output files
    echo "Output files:"
    find "$COMPARE_NNV_DIR/outputs/nnv" -name "*.mat" -type f 2>/dev/null | sort
fi
