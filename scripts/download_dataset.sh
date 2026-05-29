#!/usr/bin/env bash
set -e

HUGGINGFACE_MIRROR=true

if [ "$HUGGINGFACE_MIRROR" = true ]; then
    echo "Downloading MVTec AD from HuggingFace mirror..."
    pip install huggingface_hub -q

    python3 - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="TheoM55/mvtec_all_objects_split",
    repo_type="dataset",
    local_dir="mvtec_anomaly_detection",
    ignore_patterns=["*.md"],
)
print("Download complete.")
EOF

else
    DOWNLOAD_URL="https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/mvtec_anomaly_detection.tar.xz"
    echo "Downloading from MVTec official site..."
    wget -O mvtec_anomaly_detection.tar.xz "$DOWNLOAD_URL"
    echo "Extracting..."
    tar -xf mvtec_anomaly_detection.tar.xz
    rm -f mvtec_anomaly_detection.tar.xz
fi

echo ""
echo "Dataset structure:"
ls mvtec_anomaly_detection/
echo ""

echo "Defect image counts per object:"
for obj in bottle capsule carpet hazelnut leather pill tile toothbrush wood zipper; do
    if [ -d "mvtec_anomaly_detection/$obj/test" ]; then
        count=$(find "mvtec_anomaly_detection/$obj/test" -name "*.png" \
                ! -path "*/good/*" | wc -l)
        echo "  $obj: $count defect images"
    fi
done
