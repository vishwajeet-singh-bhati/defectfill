#!/usr/bin/env bash
# ================================================================
# scripts/download_dataset.sh
#
# Downloads and unzips the Steel Pipe Weld Defect dataset from
# the official GitHub release.
#
# Dataset: https://github.com/huangyebiaoke/steel-pipe-weld-defect-detection
# Format:  YOLO + PASCAL VOC 2007 (we use YOLO labels)
# Size:    ~350 MB compressed
#
# Usage:
#   bash scripts/download_dataset.sh
#   bash scripts/download_dataset.sh --dir /custom/path
# ================================================================

set -e   # Exit immediately on any error

# ── Default destination ─────────────────────────────────────────────────────
DEST_DIR="."
ZIP_NAME="steel-tube-dataset-all.zip"
UNZIP_DIR="steel-tube-dataset-all"
DOWNLOAD_URL="https://github.com/huangyebiaoke/steel-pipe-weld-defect-detection/releases/download/1.0/steel-tube-dataset-all.zip"

# ── Parse arguments ─────────────────────────────────────────────────────────
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dir) DEST_DIR="$2"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

echo ""
echo "================================================================"
echo "  Steel Pipe Weld Defect Dataset Downloader"
echo "================================================================"
echo "  URL         : $DOWNLOAD_URL"
echo "  Destination : $(pwd)/$UNZIP_DIR"
echo "================================================================"
echo ""

# ── Check if already downloaded ─────────────────────────────────────────────
if [ -d "$UNZIP_DIR" ]; then
    echo "  [INFO] Dataset directory already exists: $(pwd)/$UNZIP_DIR"
    echo "  Skipping download. Delete the folder to re-download."
    exit 0
fi

# ── Check for wget / curl ────────────────────────────────────────────────────
if command -v wget &> /dev/null; then
    echo "  Downloading with wget..."
    wget --progress=bar:force -O "$ZIP_NAME" "$DOWNLOAD_URL"
elif command -v curl &> /dev/null; then
    echo "  Downloading with curl..."
    curl -L --progress-bar -o "$ZIP_NAME" "$DOWNLOAD_URL"
else
    echo "  [ERROR] Neither wget nor curl found. Please install one of them."
    exit 1
fi

# ── Unzip ────────────────────────────────────────────────────────────────────
echo ""
echo "  Unzipping $ZIP_NAME..."
unzip -q "$ZIP_NAME"
echo "  Done. Dataset at: $(pwd)/$UNZIP_DIR"

# ── Clean up zip ─────────────────────────────────────────────────────────────
rm -f "$ZIP_NAME"
echo "  Cleaned up zip file."

# ── Show structure ───────────────────────────────────────────────────────────
echo ""
echo "  Dataset structure:"
echo "  ────────────────────────────────────────────────────────────"
ls -lh "$UNZIP_DIR/" 2>/dev/null | head -20
echo ""

# ── Count annotations per class ─────────────────────────────────────────────
echo "  Class annotation counts (from YOLO labels):"
echo "  ────────────────────────────────────────────────────────────"
echo "  Label | Class Name    | Count"
echo "  ------|---------------|-------"

CLASS_NAMES=("air-hole" "bite-edge" "broken-arc" "crack" "hollow-bead" "overlap" "slag-inclusion" "unfused")
LABELS_DIR="$UNZIP_DIR/labels"

if [ -d "$LABELS_DIR" ]; then
    for i in "${!CLASS_NAMES[@]}"; do
        COUNT=$(grep -rh "^${i} " "$LABELS_DIR"/*.txt 2>/dev/null | wc -l)
        printf "  %-5s | %-13s | %d\n" "$i" "${CLASS_NAMES[$i]}" "$COUNT"
    done
else
    echo "  [WARN] Labels directory not found at: $LABELS_DIR"
    echo "  Please check the zip structure and update config.yaml accordingly."
fi

echo ""
echo "  ================================================================"
echo "  Download complete! Next step:"
echo "    bash scripts/prepare_data.sh"
echo "  ================================================================"
echo ""
