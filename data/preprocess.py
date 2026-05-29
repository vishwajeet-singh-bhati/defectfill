cat > data/preprocess.py << 'ENDOFFILE'
"""
data/preprocess.py — Resize MVTec images + masks to 512x512 and cache them.
Run: python data/preprocess.py --config configs/config.yaml
"""
import argparse
import json
import numpy as np
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from dataset import build_defect_samples, get_normal_images, gray_to_rgb


def save_split(samples, out_dir, split_name, img_size):
    img_out = out_dir / split_name / "images"
    msk_out = out_dir / split_name / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)

    for s in tqdm(samples, desc=f"  {split_name}", leave=False):
        stem = Path(s.image_path).stem

        img_bgr = cv2.imread(s.image_path)
        mask    = cv2.imread(s.mask_path, cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or mask is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)

        img_r  = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        mask_r = cv2.resize(mask,    (img_size, img_size), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(str(img_out / f"{stem}.png"),
                    cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(msk_out / f"{stem}.png"), mask_r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--output_dir", default="./data_prepared")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    root     = config["dataset"]["root"]
    objects  = config["dataset"]["objects"]
    dtypes   = config["dataset"]["defect_types"]
    img_size = config["dataset"]["img_size"]
    out_dir  = Path(args.output_dir)

    stats = {}
    for obj in objects:
        for defect_type in dtypes[obj]:
            key = f"{obj}/{defect_type}"
            print(f"\nProcessing: {key}")

            ref, tgt = build_defect_samples(root, obj, defect_type, config)
            if not ref:
                continue

            obj_out = out_dir / obj / defect_type
            save_split(ref, obj_out, "reference", img_size)
            save_split(tgt, obj_out, "target",    img_size)
            stats[key] = {"reference": len(ref), "target": len(tgt)}

    stats_path = out_dir / "split_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSummary saved to: {stats_path}")
    for k, v in stats.items():
        print(f"  {k:<35} ref={v['reference']:3d}  tgt={v['target']:3d}")


if __name__ == "__main__":
    main()
ENDOFFILE