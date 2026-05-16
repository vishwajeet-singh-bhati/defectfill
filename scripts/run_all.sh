#!/usr/bin/env bash
# ================================================================
# scripts/run_all.sh
#
# End-to-end DefectFill pipeline for Steel Weld Defect dataset.
#
# Stages:
#   0. (Optional) Download dataset
#   1. Prepare data (split + mask generation)
#   2. Train DefectFill (fine-tune SD2-inpainting per defect class)
#   3. Generate synthetic defect images with LFS
#   4. Evaluate (KID, IC-LPIPS, classification accuracy, localisation)
#
# Estimated times (single A100 GPU):
#   Stage 2 (training)   : ~8 min/class × 8 classes = ~64 min
#   Stage 3 (generation) : ~15 min/class × 8 classes = ~120 min
#   Stage 4 (evaluation) : ~30-60 min (KID + classifier + UNet)
#   Total                : ~4-5 hours end-to-end
#
# Usage:
#   bash scripts/run_all.sh
#   bash scripts/run_all.sh --skip_download   # if dataset already downloaded
#   bash scripts/run_all.sh --class_id 1      # run only bite-edge class
#   bash scripts/run_all.sh --device cuda:0
#   bash scripts/run_all.sh --num_samples 100  # quick test with fewer samples
# ================================================================

set -e

# ── Defaults ─────────────────────────────────────────────────────────────────
CONFIG="configs/config.yaml"
DEVICE="auto"
CLASS_ARG=""
NUM_SAMPLES_ARG=""
SKIP_DOWNLOAD=false
SKIP_PREPARE=false
SKIP_TRAIN=false
SKIP_GENERATE=false
SKIP_EVALUATE=false
CUSTOM_MASKS=false

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config)         CONFIG="$2"; shift ;;
        --device)         DEVICE="$2"; shift ;;
        --class_id)       CLASS_ARG="--class_id $2"; shift ;;
        --num_samples)    NUM_SAMPLES_ARG="--num_samples $2"; shift ;;
        --skip_download)  SKIP_DOWNLOAD=true ;;
        --skip_prepare)   SKIP_PREPARE=true ;;
        --skip_train)     SKIP_TRAIN=true ;;
        --skip_generate)  SKIP_GENERATE=true ;;
        --skip_evaluate)  SKIP_EVALUATE=true ;;
        --custom_masks)   CUSTOM_MASKS=true ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

# ── Banner ───────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     DefectFill — Steel Pipe Weld Defect Dataset              ║"
echo "║     CVPR 2025 Paper Implementation                           ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Config   : $CONFIG"
echo "║  Device   : $DEVICE"
echo "║  Classes  : ${CLASS_ARG:-all}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
echo "  Started: $TIMESTAMP"
echo ""

# ── Stage 0: Download ────────────────────────────────────────────────────────
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "────────────────────────────────────────────────────────────────"
    echo "  Stage 0: Downloading Dataset"
    echo "────────────────────────────────────────────────────────────────"
    bash scripts/download_dataset.sh
else
    echo "  [SKIP] Dataset download (--skip_download)"
fi

# ── Stage 1: Prepare Data ────────────────────────────────────────────────────
if [ "$SKIP_PREPARE" = false ]; then
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "  Stage 1: Preparing Data (split + mask generation)"
    echo "────────────────────────────────────────────────────────────────"
    bash scripts/prepare_data.sh --config "$CONFIG"
else
    echo "  [SKIP] Data preparation (--skip_prepare)"
fi

# ── Stage 2: Train DefectFill ────────────────────────────────────────────────
if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "  Stage 2: Training DefectFill (LoRA fine-tuning per class)"
    echo "────────────────────────────────────────────────────────────────"
    echo "  Note: Training all 8 classes takes ~1 hour on a single GPU."
    echo "        Rarest classes (bite-edge, crack) are trained first."
    echo ""
    python3 train.py \
        --config "$CONFIG" \
        --device "$DEVICE" \
        $CLASS_ARG
else
    echo "  [SKIP] Training (--skip_train)"
fi

# ── Stage 3: Generate Defect Images ──────────────────────────────────────────
if [ "$SKIP_GENERATE" = false ]; then
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "  Stage 3: Generating Defect Images (DDIM + LFS)"
    echo "────────────────────────────────────────────────────────────────"
    echo "  Generating 1000 images per class with 8 LFS candidates each."
    echo ""

    CUSTOM_FLAG=""
    if [ "$CUSTOM_MASKS" = true ]; then
        CUSTOM_FLAG="--custom_masks"
        echo "  Using custom geometric mask shapes (star, square, circle)."
    fi

    python3 generate.py \
        --config "$CONFIG" \
        --device "$DEVICE" \
        $CLASS_ARG \
        $NUM_SAMPLES_ARG \
        $CUSTOM_FLAG
else
    echo "  [SKIP] Generation (--skip_generate)"
fi

# ── Stage 4: Evaluate ────────────────────────────────────────────────────────
if [ "$SKIP_EVALUATE" = false ]; then
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "  Stage 4: Evaluation (Generation + Classification + Localisation)"
    echo "────────────────────────────────────────────────────────────────"
    echo ""
    python3 evaluate.py \
        --config "$CONFIG" \
        --device "$DEVICE" \
        --stage all \
        $CLASS_ARG
else
    echo "  [SKIP] Evaluation (--skip_evaluate)"
fi

# ── Done ─────────────────────────────────────────────────────────────────────
END_TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Pipeline Complete!                                           ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Started  : $TIMESTAMP         ║"
echo "║  Finished : $END_TIMESTAMP         ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Outputs:                                                     ║"
echo "║    Generated images : outputs/generated/                      ║"
echo "║    Model checkpoints: outputs/checkpoints/                    ║"
echo "║    Evaluation results: outputs/results/                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
