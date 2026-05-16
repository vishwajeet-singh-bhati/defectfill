"""
data/preprocess.py
──────────────────
One-time preprocessing step run BEFORE training.

What this script does:
  1. Reads the raw YOLO-format dataset.
  2. For each of the 8 defect classes, splits images into:
       reference/  ← 1/3 of samples  →  DefectFill learns from these
       target/     ← 2/3 of samples  →  DefectFill generates onto these (backgrounds)
  3. Generates and saves pre-computed binary mask PNGs alongside each image
     so training does not recompute them on every step.
  4. Produces a summary JSON showing class statistics.

Why pre-generate masks?
  Mask creation involves bbox parsing + optional dilation.  Doing this once
  and caching the mask PNGs shaves significant time from training loops.

Run:
    python data/preprocess.py --config configs/config.yaml
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from dataset import (
    WeldDefectSample,
    build_class_samples,
    bbox_to_mask,
    yolo_bbox_to_pixel,
    gray_to_rgb,
)


def save_prepared_split(
    samples: list,
    out_dir: Path,
    class_id: int,
    class_name: str,
    config: dict,
    split_name: str,   # "reference" or "target"
):
    """
    Copy images + generate mask PNGs into the prepared output directory.

    Directory layout created:
        out_dir/
          {class_name}/
            {split_name}/
              images/  ← resized RGB images (PNG)
              masks/   ← binary masks (PNG, 255=defect, 0=background)
    """
    img_size = config["dataset"]["img_size"]
    dilation = config["dataset"].get("dilation_factor", 1.15)

    img_out = out_dir / class_name / split_name / "images"
    msk_out = out_dir / class_name / split_name / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)

    saved = 0
    for sample in tqdm(samples, desc=f"  {class_name}/{split_name}", leave=False):
        stem = Path(sample.image_path).stem

        # Load & convert
        img_bgr = cv2.imread(sample.image_path)
        if img_bgr is None:
            print(f"  [WARN] Cannot read: {sample.image_path}")
            continue
        img_h, img_w = img_bgr.shape[:2]
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_rgb = gray_to_rgb(img_gray)

        # Build mask for this specific class
        target_bboxes_px = [
            yolo_bbox_to_pixel(bbox_norm, img_w, img_h)
            for cid, bbox_norm in zip(sample.class_ids, sample.bboxes_norm)
            if cid == class_id
        ]
        mask = bbox_to_mask(target_bboxes_px, img_h, img_w, dilation)

        # Resize
        img_rgb_r = cv2.resize(img_rgb, (img_size, img_size),
                               interpolation=cv2.INTER_LINEAR)
        mask_r = cv2.resize(mask, (img_size, img_size),
                            interpolation=cv2.INTER_NEAREST)

        # Save (PNG for lossless storage of masks)
        cv2.imwrite(str(img_out / f"{stem}.png"), cv2.cvtColor(img_rgb_r, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(msk_out / f"{stem}.png"), mask_r)
        saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser(description="Preprocess Steel Weld Dataset for DefectFill")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output_dir", default="./data_prepared",
                        help="Where to write prepared splits")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_root = config["dataset"]["root"]
    class_names = config["dataset"]["class_names"]
    out_dir = Path(args.output_dir)

    print(f"\n{'='*60}")
    print("DefectFill — Steel Weld Dataset Preprocessing")
    print(f"{'='*60}")
    print(f"  Dataset root : {dataset_root}")
    print(f"  Output dir   : {out_dir}")
    print(f"  Image size   : {config['dataset']['img_size']}×{config['dataset']['img_size']}")
    print(f"  Mask mode    : {config['dataset']['mask_mode']} (dilation={config['dataset']['dilation_factor']})")
    print(f"  Ref split    : {config['dataset']['reference_split']:.0%} reference / "
          f"{1-config['dataset']['reference_split']:.0%} target\n")

    stats = {}
    for class_id, class_name in enumerate(class_names):
        print(f"Processing class {class_id}: [{class_name}]")
        reference, target = build_class_samples(dataset_root, class_id, config)

        if not reference:
            print(f"  [SKIP] No samples found for class {class_id} ({class_name})\n")
            continue

        n_ref = save_prepared_split(reference, out_dir, class_id, class_name,
                                    config, "reference")
        n_tgt = save_prepared_split(target, out_dir, class_id, class_name,
                                    config, "target")

        stats[class_name] = {
            "class_id": class_id,
            "total": len(reference) + len(target),
            "reference": n_ref,
            "target": n_tgt,
        }
        print(f"  ✓ reference={n_ref}, target={n_tgt}\n")

    # Write summary
    stats_path = out_dir / "split_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSummary saved to: {stats_path}")

    # Pretty print
    print("\n── Class Split Summary ──────────────────────────────────")
    print(f"{'Class':<20} {'Total':>7} {'Reference':>10} {'Target':>8}")
    print("-" * 50)
    for cname, s in stats.items():
        print(f"{cname:<20} {s['total']:>7} {s['reference']:>10} {s['target']:>8}")
    print("=" * 50)
    print("Preprocessing complete.  Run train.py next.\n")


if __name__ == "__main__":
    main()
