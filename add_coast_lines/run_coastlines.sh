#!/bin/bash
# =============================================================================
# Run script for adding coastlines to CARRA2 uncertainty plots
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ENV_NAME="coastline_plotting"
INPUT_DIR="/ec/res4/scratch/swe4281/DDPM_EVAL_JAN2021/OUTPUT_JAN2026/"
OUTPUT_DIR="../sample_data/output"

# Input mode: netcdf, netcdf_raw, or png
#   netcdf     - Upscaled netCDF files (2880x2880, grayscale normalized to [0,3])
#   netcdf_raw - Raw netCDF files (256x256, grayscale in [-1,1])
#   png        - PNG files with jet colormap (inverted to get true physical values)
MODE="png"

# Plot options (set to 1 to enable, 0 to disable)
SHOW_COLORBAR=1
SHOW_GRIDLINES=1

# Save processed data as netCDF (physical units, no padding)
SAVE_NETCDF=1

# NetCDF output resolution (only used with SAVE_NETCDF=1 and MODE=png)
# examples:
#   246  - ML model resolution (small files, ~240 KB)
#   2880 - Full CARRA2 resolution (large files, ~33 MB)
#   If left empty, it uses the native PNG resolution (754x754, ~2.3 MB)
OUTPUT_RESOLUTION="246"

# Color scale limits
VMIN=0.0
VMAX=3.0

# DateTime strings to process (format: YYYYMMDDHH)
DATETIMES=(
    "2019050100"
    "2019050106"
    "2019050112"
    "2019050118"
    "2019050200"
)

# -----------------------------------------------------------------------------
# Activate conda environment
# -----------------------------------------------------------------------------
echo "Activating conda environment: $ENV_NAME"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo "ERROR: Failed to activate conda environment '$ENV_NAME'"
    echo "Please run: conda env create -f environment.yml"
    exit 1
fi

echo "Environment activated successfully"
echo "Python: $(which python)"
echo ""

# -----------------------------------------------------------------------------
# Verify input directory and create output directory
# -----------------------------------------------------------------------------
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory does not exist: $INPUT_DIR"
    exit 1
fi
echo "Input directory: $INPUT_DIR"

mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# -----------------------------------------------------------------------------
# Build command-line options
# -----------------------------------------------------------------------------
OPTS="--mode $MODE"
[ "$SHOW_COLORBAR" -eq 1 ] && OPTS="$OPTS --colorbar"
[ "$SHOW_GRIDLINES" -eq 1 ] && OPTS="$OPTS --gridlines"
[ "$SAVE_NETCDF" -eq 1 ] && OPTS="$OPTS --save-netcdf"
[ -n "$OUTPUT_RESOLUTION" ] && OPTS="$OPTS --output-resolution $OUTPUT_RESOLUTION"

# -----------------------------------------------------------------------------
# Process each datetime
# -----------------------------------------------------------------------------
echo "Processing ${#DATETIMES[@]} datetime(s)..."
echo ""

for DT in "${DATETIMES[@]}"; do
    echo "=== Processing: $DT ==="
    python add_coastlines.py "$DT" \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --vmin "$VMIN" \
        --vmax "$VMAX" \
        $OPTS
    echo ""
done

echo "All done!"
