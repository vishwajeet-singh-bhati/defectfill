"""
evaluate.py
───────────
Unified evaluation script — runs all metrics from the DefectFill paper and
produces a comprehensive results report.

Three evaluation stages:
─────────────────────────
  Stage 1 — Generation Quality
    • KID  (Kernel Inception Distance)  — lower is better
    • IC-LPIPS (Intra-Class LPIPS)      — higher is better
    Compares: generated images vs real held-out test images, per class.

  Stage 2 — Classification (Downstream)
    • Accuracy per class
    Tests: ResNet-34 trained on GENERATED images, tested on REAL images.
    Shows that generated images carry enough information to train a real classifier.

  Stage 3 — Localisation (Downstream)
    • AUROC, AP, F1-max, PRO  (per class + overall)
    Tests: UNet trained on GENERATED images+masks, tested on REAL images.

All results are saved to outputs/results/ as JSON + PNG.

Run:
    python evaluate.py --config configs/config.yaml
    python evaluate.py --config configs/config.yaml --stage generation
    python evaluate.py --config configs/config.yaml --stage classification
    python evaluate.py --config configs/config.yaml --stage localization
    python evaluate.py --config configs/config.yaml --class_id 3
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from data.dataset import build_class_samples, yolo_bbox_to_pixel, bbox_to_mask, gray_to_rgb
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
    plot_class_distribution,
    plot_metrics_table,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_generated_images(gen_dir: str, class_name: str, max_n: int = 1000) -> List[np.ndarray]:
    """Load generated images for a class from disk."""
    img_dir = Path(gen_dir) / class_name / "images"
    if not img_dir.exists():
        return []
    paths = sorted(img_dir.glob("*.png"))[:max_n]
    imgs = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return imgs


def load_generated_masks(gen_dir: str, class_name: str, max_n: int = 1000) -> List[np.ndarray]:
    """Load generated masks for a class from disk."""
    msk_dir = Path(gen_dir) / class_name / "masks"
    if not msk_dir.exists():
        return []
    paths = sorted(msk_dir.glob("*.png"))[:max_n]
    masks = []
    for p in paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            masks.append(m)
    return masks


def load_real_test_images(
    dataset_root: str, class_id: int, config: dict
) -> List[np.ndarray]:
    """Load real test images (target split, not used in training)."""
    _, target_samples = build_class_samples(dataset_root, class_id, config)
    imgs = []
    for s in target_samples:
        img_bgr = cv2.imread(s.image_path)
        if img_bgr is not None:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            imgs.append(gray_to_rgb(gray))
    return imgs


def numpy_to_pil(img: np.ndarray) -> Image.Image:
    """Convert (H,W,3) uint8 numpy to PIL RGB."""
    return Image.fromarray(img.astype(np.uint8))


def img_to_tensor(img: np.ndarray, size: int = 256) -> torch.Tensor:
    """(H,W,3) uint8 → (1,3,H,W) float [-1,1] tensor."""
    img_pil = Image.fromarray(img.astype(np.uint8)).convert("RGB")
    import torchvision.transforms as T
    t = T.Compose([T.Resize((size, size)), T.ToTensor(),
                   T.Normalize([0.5]*3, [0.5]*3)])(img_pil)
    return t.unsqueeze(0)


# ─── Stage 1: Generation Quality ─────────────────────────────────────────────

def evaluate_generation(config: dict, device: torch.device, class_ids: List[int]) -> Dict:
    """
    Evaluate KID and IC-LPIPS for each defect class.

    For each class:
      real_images  = target split (real X-ray images, not seen during training)
      fake_images  = 1000 generated images from generate.py
    """
    gen_dir = config["generation"]["output_dir"]
    dataset_root = config["dataset"]["root"]
    class_names = config["dataset"]["class_names"]
    results_dir = Path(config["evaluation"]["results_dir"]) / "generation"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Stage 1: Generation Quality (KID + IC-LPIPS)")
    print(f"{'='*60}")

    all_results = {}

    for class_id in class_ids:
        cls_name = class_names[class_id]
        print(f"\n  [{cls_name}]")

        # Load images
        real_np  = load_real_test_images(dataset_root, class_id, config)
        fake_np  = load_generated_images(gen_dir, cls_name,
                                          config["evaluation"]["kid_num_samples"])
        fake_msk = load_generated_masks(gen_dir, cls_name,
                                         config["evaluation"]["kid_num_samples"])

        if len(real_np) == 0:
            print(f"  [SKIP] No real test images for {cls_name}")
            continue
        if len(fake_np) == 0:
            print(f"  [SKIP] No generated images for {cls_name} — run generate.py first")
            continue

        print(f"    Real test images : {len(real_np)}")
        print(f"    Generated images : {len(fake_np)}")

        # Convert to PIL for metric functions
        real_pil = [numpy_to_pil(img) for img in real_np]
        fake_pil = [numpy_to_pil(img) for img in fake_np]
        fake_t   = [img_to_tensor(img) for img in fake_np[:200]]  # subsample for IC-LPIPS speed

        scores = summarise_generation(real_pil, fake_pil, fake_t, device=str(device))
        all_results[cls_name] = scores
        print(f"    KID:       {scores['KID_mean']:.4f} ± {scores['KID_std']:.4f}")
        print(f"    IC-LPIPS:  {scores['IC_LPIPS']:.4f}")

        # Visualise a sample grid
        if len(fake_np) >= 5:
            plot_generated_grid(
                images=fake_np[:10],
                masks=fake_msk[:10],
                class_name=cls_name,
                save_path=str(results_dir / f"generated_grid_{cls_name}.png"),
                n_cols=5,
            )

    # Save JSON results
    out_json = results_dir / "generation_metrics.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {out_json}")

    # Print summary table
    print(f"\n  ── Generation Quality Summary ──────────────────────────")
    print(f"  {'Class':<20} {'KID↓':>10} {'IC-LPIPS↑':>12}")
    print("  " + "-" * 44)
    for cls_name, s in all_results.items():
        print(f"  {cls_name:<20} {s['KID_mean']:>10.4f} {s['IC_LPIPS']:>12.4f}")

    # Plot metrics table
    plot_metrics_table(
        all_results, "Generation Quality Metrics (DefectFill)",
        str(results_dir / "generation_metrics_table.png"),
    )
    return all_results


# ─── Stage 2: Classification ─────────────────────────────────────────────────

def evaluate_classification(config: dict, device: torch.device) -> Dict:
    """
    Train ResNet-34 on generated images, evaluate accuracy on real test images.
    Returns per-class accuracy dict.
    """
    print(f"\n{'='*60}")
    print("  Stage 2: Classification (ResNet-34)")
    print(f"{'='*60}")

    sys.path.insert(0, str(Path(__file__).parent))
    # FIX: renamed from 'inspect' to 'inspect_defect' to avoid shadowing stdlib inspect
    from inspect_defect.classifier import train_classifier
    results = train_classifier(config, device)

    # Visualise final per-class accuracies
    results_dir = Path(config["evaluation"]["results_dir"]) / "classification"
    last_per_class = results["per_class_acc"][-1] if results["per_class_acc"] else {}

    if last_per_class:
        cls_names = list(last_per_class.keys())
        accs = list(last_per_class.values())
        plot_classification_bar(
            {"DefectFill (Ours)": last_per_class},
            cls_names,
            save_path=str(results_dir / "classification_accuracy.png"),
        )
        # Plot dataset distribution for context
        raw_counts = [5191, 35, 458, 119, 229, 223, 120, 408]
        cls_all_names = config["dataset"]["class_names"]
        plot_class_distribution(
            cls_all_names, raw_counts,
            save_path=str(results_dir / "dataset_class_distribution.png"),
        )

    # Plot training loss curve
    results_dir.mkdir(parents=True, exist_ok=True)
    return results


# ─── Stage 3: Localisation ───────────────────────────────────────────────────

def evaluate_localisation(config: dict, device: torch.device, class_ids: List[int]) -> Dict:
    """
    Train UNet on generated images+masks, evaluate on real test images.
    Returns AUROC, AP, F1-max, PRO per class.
    """
    print(f"\n{'='*60}")
    print("  Stage 3: Localisation (UNet + Focal Loss)")
    print(f"{'='*60}")

    # FIX: renamed from 'inspect' to 'inspect_defect' to avoid shadowing stdlib inspect
    from inspect_defect.localizer import train_localizer, WeldUNet
    import torchvision.transforms as T

    # Train the localiser
    model = train_localizer(config, device)
    model.eval()

    dataset_root = config["dataset"]["root"]
    class_names  = config["dataset"]["class_names"]
    img_size     = config["dataset"]["img_size"]
    results_dir  = Path(config["evaluation"]["results_dir"]) / "localization"
    results_dir.mkdir(parents=True, exist_ok=True)

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    all_class_metrics = {}

    for class_id in class_ids:
        cls_name = class_names[class_id]
        print(f"\n  Evaluating localisation for [{cls_name}]...")

        # Load real test images + gt masks
        _, target_samples = build_class_samples(dataset_root, class_id, config)
        if not target_samples:
            print(f"  [SKIP] No test samples for {cls_name}")
            continue

        pred_maps, gt_masks, vis_images = [], [], []

        for sample in tqdm(target_samples, desc=f"  [{cls_name}]", leave=False):
            img_bgr = cv2.imread(sample.image_path)
            if img_bgr is None:
                continue
            img_h, img_w = img_bgr.shape[:2]
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_rgb  = gray_to_rgb(img_gray)

            # Ground-truth mask from YOLO bbox
            bboxes_px = [
                yolo_bbox_to_pixel(bn, img_w, img_h)
                for cid, bn in zip(sample.class_ids, sample.bboxes_norm)
                if cid == class_id
            ]
            if not bboxes_px:
                continue
            gt_mask = bbox_to_mask(bboxes_px, img_h, img_w,
                                   config["dataset"].get("dilation_factor", 1.15))
            gt_mask_rs = cv2.resize(gt_mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

            # Model prediction
            img_pil = Image.fromarray(cv2.resize(img_rgb, (img_size, img_size)).astype(np.uint8))
            img_t = transform(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                logit = model(img_t)  # (1, 1, H, W)
            pred = torch.sigmoid(logit).squeeze().cpu().numpy()  # (H, W)

            pred_maps.append(pred)
            gt_masks.append((gt_mask_rs > 127).astype(float))
            vis_images.append(cv2.resize(img_rgb, (img_size, img_size)))

        if not pred_maps:
            continue

        metrics = summarise_localisation(pred_maps, gt_masks)
        all_class_metrics[cls_name] = metrics
        print(f"    AUROC:  {metrics['AUROC']:.4f}")
        print(f"    AP:     {metrics['AP']:.4f}")
        print(f"    F1-max: {metrics['F1-max']:.4f}")
        print(f"    PRO:    {metrics['PRO']:.4f}")

        # Visualise some predictions
        if len(vis_images) >= 3:
            plot_localisation_masks(
                vis_images[:6], pred_maps[:6], gt_masks[:6],
                save_path=str(results_dir / f"localisation_{cls_name}.png"),
                class_name=cls_name,
                threshold=metrics["best_threshold"],
            )

    # Save JSON
    out_json = results_dir / "localisation_metrics.json"
    with open(out_json, "w") as f:
        json.dump(all_class_metrics, f, indent=2)
    print(f"\n  Saved: {out_json}")

    # Summary table
    print(f"\n  ── Localisation Summary ────────────────────────────────")
    print(f"  {'Class':<20} {'AUROC':>8} {'AP':>8} {'F1-max':>8} {'PRO':>8}")
    print("  " + "-" * 55)
    for cls_name, m in all_class_metrics.items():
        print(f"  {cls_name:<20} {m['AUROC']:>8.4f} {m['AP']:>8.4f} {m['F1-max']:>8.4f} {m['PRO']:>8.4f}")

    if all_class_metrics:
        plot_metrics_table(
            all_class_metrics,
            "Localisation Metrics (UNet on Generated Data)",
            str(results_dir / "localisation_metrics_table.png"),
        )

    return all_class_metrics


# ─── Training Log Plots ───────────────────────────────────────────────────────

def plot_all_training_logs(config: dict, class_ids: List[int]):
    """Load saved training logs and plot loss curves for all classes."""
    ckpt_dir = Path(config["training"]["output_dir"])
    class_names = config["dataset"]["class_names"]
    plots_dir = Path(config["evaluation"]["results_dir"]) / "training_curves"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for class_id in class_ids:
        cls_name = class_names[class_id]
        log_path = ckpt_dir / f"{cls_name}_train_log.json"
        if not log_path.exists():
            continue
        with open(log_path) as f:
            log = json.load(f)
        plot_training_curves(
            log=log,
            save_path=str(plots_dir / f"loss_{cls_name}.png"),
            class_name=cls_name,
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate DefectFill on Steel Weld Dataset")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--stage", choices=["all", "generation", "classification", "localization"],
                        default="all", help="Which evaluation stage to run.")
    parser.add_argument("--class_id", type=int, default=None,
                        help="Evaluate only one class ID (0-7). Default: all.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    class_names = config["dataset"]["class_names"]
    class_ids = [args.class_id] if args.class_id is not None else list(range(len(class_names)))

    all_results = {}

    # Plot training curves (always useful)
    print("\nPlotting training loss curves...")
    plot_all_training_logs(config, class_ids)

    # Stage 1
    if args.stage in ("all", "generation"):
        gen_results = evaluate_generation(config, device, class_ids)
        all_results["generation"] = gen_results

    # Stage 2
    if args.stage in ("all", "classification"):
        cls_results = evaluate_classification(config, device)
        all_results["classification"] = cls_results

    # Stage 3
    if args.stage in ("all", "localization"):
        loc_results = evaluate_localisation(config, device, class_ids)
        all_results["localization"] = loc_results

    # Final combined save
    combined_path = Path(config["evaluation"]["results_dir"]) / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  All evaluation complete!")
    print(f"  Results saved to: {config['evaluation']['results_dir']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()