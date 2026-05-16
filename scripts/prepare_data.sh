#!/usr/bin/env bash
# ================================================================
# scripts/prepare_data.sh
#
# Runs data/preprocess.py to:
#   1. Split each defect class into reference (1/3) + target (2/3) sets.
#   2. Resize all images to 512×512 (SD2-inpainting input size).
#   3. Generate and cache binary mask PNGs from YOLO bounding boxes.
#   4. Write a summary JSON showing class split statistics.
#
# Output goes to: ./data_prepared/
#   ├── air-hole/reference/{images/, masks/}
#   ├── air-hole/target/{images/, masks/}
#   ├── bite-edge/reference/{images/, masks/}
#   └── ... (one folder per class)
#
# Usage:
#   bash scripts/prepare_data.sh
#   bash scripts/prepare_data.sh --config configs/config.yaml
#   bash scripts/prepare_data.sh --output_dir /custom/prepared
# ================================================================

set -e

CONFIG="configs/config.yaml"
OUTPUT_DIR="./data_prepared"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config)     CONFIG="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

echo ""
echo "================================================================"
echo "  DefectFill — Data Preparation"
echo "================================================================"
echo "  Config     : $CONFIG"
echo "  Output dir : $OUTPUT_DIR"
echo "================================================================"
echo ""

# Verify dataset exists
DATASET_ROOT=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg['dataset']['root'])
")

if [ ! -d "$DATASET_ROOT" ]; then
    echo "  [ERROR] Dataset not found at: $DATASET_ROOT"
    echo "  Run: bash scripts/download_dataset.sh first."
    exit 1
fi

echo "  Dataset root: $DATASET_ROOT  ✓"
echo ""

# Run preprocessing
python3 data/preprocess.py \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "================================================================"
echo "  Data preparation complete!"
echo "  Prepared data at: $OUTPUT_DIR"
echo ""
echo "  Next step:"
echo "    python train.py --config $CONFIG"
echo "================================================================"
echo ""
