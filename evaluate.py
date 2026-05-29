"""
evaluate.py
───────────
Unified evaluation script for DefectFill on the MVTec AD dataset.
Reproduces Tables 1, 2, 3 from the paper.

Three evaluation stages:
─────────────────────────
Stage 1 — Generation Quality (per object × defect_type)
  • KID  (Kernel Inception Distance)  — lower is better
  • IC-LPIPS (Intra-Class LPIPS)      — higher is better

Stage 2 — Classification (per object, matching Table 3)
  • ResNet-34 trained on GENERATED images, tested on REAL images
  • Accuracy per object (classifying defect_type within object)

Stage 3 — Localisation (per object × defect_type, matching Table 2)
  • AUROC, AP, F1-max, PRO
  • Uses PIXEL-PERFECT MVTec ground-truth masks (not bbox rectangles)
  • This is why PRO is ~0.90 here vs ~0.10 on steel pipe

Run:
    python evaluate.py --config configs/config.yaml
    python evaluate.py --config configs/config.yaml --stage generation
    python evaluate.py --config configs/config.yaml --stage classification
    python evaluate.py --config configs/config.yaml --stage localization
    python evaluate.py --config configs/config.yaml --object hazelnut
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import build_defect_samples, gray_to_rgb
from utils.metrics import (
    summarise_generation,
    summarise_localisation,
    compute_accuracy,
    InceptionFeatureExtractor,
)
from utils.visualization import (
    plot_generated_grid,
    plot_training_curves,
    plot_classification_bar,
    plot_localisation_masks,
    plot_metrics_table,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_mvtec_test_images(dataset_root: str, object_name: str, defect_type: str) -> List[np.ndarray]:
    """Load real defect images from MVTec test split."""
    img_dir = Path(dataset_root) / object_name / "test" / defect_type
    if not img_dir.exists():
        return []
    imgs = []
    for p in sorted(img_dir.glob("*.png")):
        img = cv2.imread(str(p))
        if img is not None:
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return imgs


def load_mvtec_test_masks(dataset_root: str, object_name: str, defect_type: str) -> List[np.ndarray]:
    """Load pixel-level ground-truth masks from MVTec ground_truth split."""
    mask_dir = Path(dataset_root) / object_name / "ground_truth" / defect_type
    if not mask_dir.exists():
        return []
    masks = []
    for p in sorted(mask_dir.glob("*.png")):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            if m.max() <= 1:
                m = (m * 255).astype(np.uint8)
            masks.append(m)
    return masks


def load_generated_images(gen_dir: Path, max_n: int = 1000) -> List[np.ndarray]:
    """Load generated images from outputs/generated/{object}/{defect_type}/images/"""
    img_dir = gen_dir / "images"
    if not img_dir.exists():
        return []
    paths = sorted(img_dir.glob("*.png"))[:max_n]
    imgs  = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return imgs


def load_generated_masks(gen_dir: Path, max_n: int = 1000) -> List[np.ndarray]:
    """Load generated masks from outputs/generated/{object}/{defect_type}/masks/"""
    msk_dir = gen_dir / "masks"
    if not msk_dir.exists():
        return []
    paths = sorted(msk_dir.glob("*.png"))[:max_n]
    masks = []
    for p in paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            masks.append(m)
    return masks


def numpy_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(img.astype(np.uint8))


def img_to_tensor(img: np.ndarray, size: int = 256) -> torch.Tensor:
    img_pil = Image.fromarray(img.astype(np.uint8)).convert("RGB")
    import torchvision.transforms as T
    t = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])(img_pil)
    return t.unsqueeze(0)


# ─── Stage 1: Generation Quality ─────────────────────────────────────────────

def evaluate_generation(config: dict, device: torch.device,
                        objects: List[str], defect_types: dict) -> Dict:
    """
    Evaluate KID and IC-LPIPS for each (object, defect_type).
    real_images = MVTec test images for that defect_type
    fake_images = up to 1000 generated images from generate.py
    """
    gen_dir      = config["generation"]["output_dir"]
    dataset_root = config["dataset"]["root"]
    results_dir  = Path(config["evaluation"]["results_dir"]) / "generation"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Stage 1: Generation Quality (KID + IC-LPIPS)")
    print(f"{'='*60}")

    all_results = {}

    for object_name in objects:
        for defect_type in defect_types[object_name]:
            run_name = f"{object_name}_{defect_type}"
            print(f"\n  [{run_name}]")

            real_np  = load_mvtec_test_images(dataset_root, object_name, defect_type)
            fake_np  = load_generated_images(
                Path(gen_dir) / object_name / defect_type,
                max_n=config["evaluation"]["kid_num_samples"],
            )
            fake_msk = load_generated_masks(
                Path(gen_dir) / object_name / defect_type,
                max_n=config["evaluation"]["kid_num_samples"],
            )

            if not real_np:
                print(f"  [SKIP] No real test images for {run_name}")
                continue
            if not fake_np:
                print(f"  [SKIP] No generated images for {run_name} — run generate.py first")
                continue

            print(f"  Real test images  : {len(real_np)}")
            print(f"  Generated images  : {len(fake_np)}")

            real_pil = [numpy_to_pil(img) for img in real_np]
            fake_pil = [numpy_to_pil(img) for img in fake_np]
            fake_t   = [img_to_tensor(img) for img in fake_np[:200]]  # subsample for IC-LPIPS speed

            scores = summarise_generation(real_pil, fake_pil, fake_t, device=str(device))
            all_results[run_name] = scores

            print(f"  KID      : {scores['KID_mean']:.4f} ± {scores['KID_std']:.4f}")
            print(f"  IC-LPIPS : {scores['IC_LPIPS']:.4f}")

            if len(fake_np) >= 5:
                plot_generated_grid(
                    images=fake_np[:10],
                    masks=fake_msk[:10],
                    class_name=run_name,
                    save_path=str(results_dir / f"generated_grid_{run_name}.png"),
                    n_cols=5,
                )

    out_json = results_dir / "generation_metrics.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {out_json}")

    print(f"\n  ── Generation Quality Summary ──────────────────────────────────")
    print(f"  {'Run':<35} {'KID↓':>10} {'IC-LPIPS↑':>12}")
    print("  " + "-" * 59)
    for run_name, s in all_results.items():
        print(f"  {run_name:<35} {s['KID_mean']:>10.4f} {s['IC_LPIPS']:>12.4f}")

    plot_metrics_table(
        all_results,
        "Generation Quality Metrics (DefectFill — MVTec AD)",
        str(results_dir / "generation_metrics_table.png"),
    )
    return all_results


# ─── Stage 2: Classification ─────────────────────────────────────────────────

def evaluate_classification(config: dict, device: torch.device,
                             objects: List[str], defect_types: dict) -> Dict:
    """
    Train ResNet-34 on generated images per object, evaluate on real test images.
    Matches Table 3 in the paper — one classifier per object,
    classifying defect_type within that object.
    """
    print(f"\n{'='*60}")
    print("  Stage 2: Classification (ResNet-34, per object)")
    print(f"{'='*60}")

    from inspect_defect.classifier import train_classifier

    all_results = {}
    for object_name in objects:
        dtypes = defect_types[object_name]
        if len(dtypes) < 2:
            print(f"  [SKIP] {object_name} has only 1 defect type — skipping classification")
            continue
        print(f"\n  [{object_name}]  ({len(dtypes)} defect types: {dtypes})")
        results = train_classifier(config, device, object_name=object_name,
                                   defect_types=dtypes)
        all_results[object_name] = results

    results_dir = Path(config["evaluation"]["results_dir"]) / "classification"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_json = results_dir / "classification_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {out_json}")

    # Per-object accuracy bar chart
    per_obj_acc = {}
    for obj, res in all_results.items():
        last = res.get("per_class_acc", [{}])[-1]
        per_obj_acc[obj] = last
    if per_obj_acc:
        plot_classification_bar(
            {"DefectFill (Ours)": {k: np.mean(list(v.values())) for k, v in per_obj_acc.items()}},
            list(per_obj_acc.keys()),
            save_path=str(results_dir / "classification_accuracy_per_object.png"),
        )

    return all_results


# ─── Stage 3: Localisation ───────────────────────────────────────────────────

def evaluate_localisation(config: dict, device: torch.device,
                           objects: List[str], defect_types: dict) -> Dict:
    """
    Train UNet on generated images+masks, evaluate on real MVTec test images.
    Uses PIXEL-PERFECT ground-truth masks — this is what makes PRO ~0.90+.
    Matches Table 2 in the paper.
    """
    print(f"\n{'='*60}")
    print("  Stage 3: Localisation (UNet + Focal Loss)")
    print(f"{'='*60}")

    from inspect_defect.localizer import train_localizer, WeldUNet
    import torchvision.transforms as T

    model = train_localizer(config, device)
    model.eval()

    dataset_root = config["dataset"]["root"]
    img_size     = config["dataset"]["img_size"]
    results_dir  = Path(config["evaluation"]["results_dir"]) / "localization"
    results_dir.mkdir(parents=True, exist_ok=True)

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    all_class_metrics = {}

    for object_name in objects:
        for defect_type in defect_types[object_name]:
            run_name = f"{object_name}_{defect_type}"
            print(f"\n  Evaluating localisation for [{run_name}]...")

            # Load real test images and pixel-level GT masks from MVTec
            _, target_samples = build_defect_samples(
                dataset_root, object_name, defect_type, config
            )
            if not target_samples:
                print(f"  [SKIP] No test samples for {run_name}")
                continue

            pred_maps, gt_masks, vis_images = [], [], []

            for sample in tqdm(target_samples, desc=f"  [{run_name}]", leave=False):
                img_bgr = cv2.imread(sample.image_path)
                if img_bgr is None:
                    continue

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # Handle grayscale objects
                grayscale_objects = config["dataset"].get("grayscale_objects", [])
                if object_name in grayscale_objects:
                    if img_rgb.ndim == 2 or (img_rgb.ndim == 3 and img_rgb.shape[2] == 1):
                        img_rgb = gray_to_rgb(img_rgb)

                # ── Pixel-perfect ground-truth mask from MVTec ────────────
                gt_mask = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)
                if gt_mask is None:
                    continue
                if gt_mask.max() <= 1:
                    gt_mask = (gt_mask * 255).astype(np.uint8)
                gt_mask_rs = cv2.resize(gt_mask, (img_size, img_size),
                                        interpolation=cv2.INTER_NEAREST)

                # ── Model prediction ──────────────────────────────────────
                img_resized = cv2.resize(img_rgb, (img_size, img_size))
                img_pil     = Image.fromarray(img_resized.astype(np.uint8))
                img_t       = transform(img_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    logit = model(img_t)                            # (1, 1, H, W)
                    pred  = torch.sigmoid(logit).squeeze().cpu().numpy()  # (H, W)

                pred_maps.append(pred)
                gt_masks.append((gt_mask_rs > 127).astype(float))
                vis_images.append(img_resized)

            if not pred_maps:
                continue

            metrics = summarise_localisation(pred_maps, gt_masks)
            all_class_metrics[run_name] = metrics

            print(f"  AUROC  : {metrics['AUROC']:.4f}")
            print(f"  AP     : {metrics['AP']:.4f}")
            print(f"  F1-max : {metrics['F1-max']:.4f}")
            print(f"  PRO    : {metrics['PRO']:.4f}")

            if len(vis_images) >= 3:
                plot_localisation_masks(
                    vis_images[:6], pred_maps[:6], gt_masks[:6],
                    save_path=str(results_dir / f"localisation_{run_name}.png"),
                    class_name=run_name,
                    threshold=metrics["best_threshold"],
                )

    out_json = results_dir / "localisation_metrics.json"
    with open(out_json, "w") as f:
        json.dump(all_class_metrics, f, indent=2)
    print(f"\n  Saved: {out_json}")

    print(f"\n  ── Localisation Summary ────────────────────────────────────────")
    print(f"  {'Run':<35} {'AUROC':>8} {'AP':>8} {'F1-max':>8} {'PRO':>8}")
    print("  " + "-" * 67)
    for run_name, m in all_class_metrics.items():
        print(f"  {run_name:<35} {m['AUROC']:>8.4f} {m['AP']:>8.4f} "
              f"{m['F1-max']:>8.4f} {m['PRO']:>8.4f}")

    if all_class_metrics:
        plot_metrics_table(
            all_class_metrics,
            "Localisation Metrics (UNet on Generated Data — MVTec AD)",
            str(results_dir / "localisation_metrics_table.png"),
        )

    return all_class_metrics


# ─── Training Log Plots ───────────────────────────────────────────────────────

def plot_all_training_logs(config: dict, objects: List[str], defect_types: dict):
    """Load saved training logs and plot loss curves for all (object, defect_type) runs."""
    ckpt_dir  = Path(config["training"]["output_dir"])
    plots_dir = Path(config["evaluation"]["results_dir"]) / "training_curves"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for object_name in objects:
        for defect_type in defect_types[object_name]:
            run_name = f"{object_name}_{defect_type}"
            log_path = ckpt_dir / f"{run_name}_train_log.json"
            if not log_path.exists():
                continue
            with open(log_path) as f:
                log = json.load(f)
            plot_training_curves(
                log=log,
                save_path=str(plots_dir / f"loss_{run_name}.png"),
                class_name=run_name,
            )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate DefectFill on MVTec AD Dataset")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--stage",
                        choices=["all", "generation", "classification", "localization"],
                        default="all")
    parser.add_argument("--object", type=str, default=None,
                        help="Evaluate only this object, e.g. 'hazelnut'. Default: all.")
    parser.add_argument("--defect_type", type=str, default=None,
                        help="Evaluate only this defect type. Default: all.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    objects      = config["dataset"]["objects"]
    defect_types = config["dataset"]["defect_types"]

    # Filter by CLI flags
    if args.object:
        if args.object not in objects:
            raise ValueError(f"Unknown object '{args.object}'. Valid: {objects}")
        objects = [args.object]
    if args.defect_type:
        defect_types = {obj: [args.defect_type] for obj in objects}

    all_results = {}

    # Always plot training curves first (they exist independently)
    print("\nPlotting training loss curves...")
    plot_all_training_logs(config, objects, defect_types)

    # Stage 1
    if args.stage in ("all", "generation"):
        gen_results = evaluate_generation(config, device, objects, defect_types)
        all_results["generation"] = gen_results

    # Stage 2
    if args.stage in ("all", "classification"):
        cls_results = evaluate_classification(config, device, objects, defect_types)
        all_results["classification"] = cls_results

    # Stage 3
    if args.stage in ("all", "localization"):
        loc_results = evaluate_localisation(config, device, objects, defect_types)
        all_results["localization"] = loc_results

    # Combined results
    combined_path = Path(config["evaluation"]["results_dir"]) / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  All evaluation complete!")
    print(f"  Results saved to: {config['evaluation']['results_dir']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()